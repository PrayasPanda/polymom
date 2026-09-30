"""Meeting use cases: upload (with dedupe), fetch, list, delete, results of a run.

Stage results are read from the meeting's **latest successful run** (falling back
to the latest run of any status), or from an explicit ``run_id``.
"""

import uuid
from typing import Any

import structlog
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.exceptions import (
    AnalyticsNotAvailableError,
    DiarizationNotAvailableError,
    LanguageSummaryNotAvailableError,
    MeetingNotFoundError,
    MeetingStateConflictError,
    NotFoundError,
    PolymomError,
    SummaryNotAvailableError,
    TranscriptNotAvailableError,
    ValidationError,
)
from app.core.logging import get_logger
from app.models.meeting import Meeting
from app.models.results import ProcessingRun
from app.repositories.artifacts import ArtifactStore, meeting_prefix, run_key
from app.repositories.meeting_repository import MeetingPage, MeetingQuery
from app.repositories.unit_of_work import UnitOfWork
from app.schemas.analytics import ConversationAnalytics
from app.schemas.asr import ASRResult
from app.schemas.audio import AudioQuality
from app.schemas.diarization import DiarizationResult
from app.schemas.language import LanguageSummary
from app.schemas.meeting import AudioMetadata, MeetingRead, MeetingStatus
from app.schemas.summary import MeetingSummary
from app.schemas.transcript import SpeakerTranscript
from app.services.audio.validator import AsyncReadable, MediaValidator
from app.services.stage_outputs import load_stage_output

logger = get_logger(__name__)


class RunNotFoundError(NotFoundError):
    code = "run_not_found"


def upload_key(meeting_id: uuid.UUID, extension: str) -> str:
    return f"{meeting_prefix(meeting_id)}upload/original.{extension}"


class MeetingService:
    """Services scope every read and write to the caller's ``owner_key_id``.

    A meeting whose ``owner_key_id`` does not match the caller is reported as
    ``meeting_not_found`` (never leak its existence to another tenant).
    ``owner_key_id=None`` means unscoped, used by workers and admin scripts.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        validator: MediaValidator,
        settings: Settings,
        store: ArtifactStore,
        owner_key_id: int | None = None,
    ) -> None:
        self.uow = uow
        self.store = store
        self.owner_key_id = owner_key_id
        self._validator = validator
        self._settings = settings

    def _check_owner(self, meeting: Meeting) -> None:
        if self.owner_key_id is not None and meeting.owner_key_id != self.owner_key_id:
            raise MeetingNotFoundError(
                f"Meeting {meeting.id} not found.", details={"meeting_id": str(meeting.id)}
            )

    # --- upload ---

    async def create(
        self,
        *,
        source: AsyncReadable,
        filename: str | None,
        title: str | None,
        expected_speakers: int | None,
        languages: list[str],
        allow_duplicate: bool = False,
    ) -> tuple[Meeting, bool]:
        """Validate and store an upload, then record it as ``queued``.

        Returns ``(meeting, duplicate)``: an identical file (same SHA-256) returns the
        existing meeting unless ``allow_duplicate``.
        """
        meeting_id = uuid.uuid4()
        tmp_dir = self._settings.storage_dir / "tmp"
        with structlog.contextvars.bound_contextvars(meeting_id=str(meeting_id)):
            logger.info("upload_started", filename=filename)
            try:
                media = await self._validator.validate(source, filename, tmp_dir, str(meeting_id))
            except PolymomError as exc:
                logger.warning("upload_rejected", code=exc.code, reason=exc.message)
                raise
            try:
                if not allow_duplicate:
                    existing = await self.uow.meetings.get_by_sha256(
                        media.sha256, owner_key_id=self.owner_key_id
                    )
                    if existing is not None:
                        logger.info("upload_duplicate", existing=str(existing.id))
                        return existing, True
                key = upload_key(meeting_id, media.extension)
                await self.store.put_file(key, media.path, media.mime_type)
            finally:
                await run_in_threadpool(media.path.unlink, missing_ok=True)
            logger.info(
                "upload_validated",
                size_bytes=media.size_bytes,
                mime_type=media.mime_type,
                duration_seconds=media.metadata.duration_seconds,
            )
            meeting = Meeting(
                id=meeting_id,
                title=title,
                original_filename=media.original_filename,
                owner_key_id=self.owner_key_id,
                upload_key=key,
                sha256=media.sha256,
                mime_type=media.mime_type,
                size_bytes=media.size_bytes,
                duration_seconds=media.metadata.duration_seconds,
                audio_metadata=media.metadata.model_dump(),
                languages_hint=languages,
                expected_speakers=expected_speakers,
                status=MeetingStatus.QUEUED,
            )
            try:
                await self.uow.meetings.add(meeting)
                await self.uow.commit()
            except BaseException:
                await self.store.delete(key)
                logger.exception("upload_persist_failed")
                raise
            logger.info("meeting_queued")
            return meeting, False

    # --- meetings ---

    async def get(self, meeting_id: uuid.UUID) -> Meeting:
        meeting = await self.uow.meetings.get(meeting_id)
        if meeting is None:
            raise MeetingNotFoundError(
                f"Meeting {meeting_id} not found.", details={"meeting_id": str(meeting_id)}
            )
        self._check_owner(meeting)
        return meeting

    async def page(self, query: MeetingQuery) -> MeetingPage:
        query.owner_key_id = self.owner_key_id
        return await self.uow.meetings.page(query)

    async def default_run(self, meeting_id: uuid.UUID) -> ProcessingRun | None:
        """The latest successful run, else the latest run of any status."""
        results = self.uow.results
        return await results.latest_run(meeting_id) or await results.latest_run(
            meeting_id, successful=False
        )

    async def to_read(self, meeting: Meeting) -> MeetingRead:
        run = await self.default_run(meeting.id)
        quality = (
            await load_stage_output(self.uow.results, self.store, run.id, "preprocess")
            if run
            else None
        )
        return MeetingRead(
            meeting_id=meeting.id,
            title=meeting.title,
            original_filename=meeting.original_filename,
            mime_type=meeting.mime_type,
            size_bytes=meeting.size_bytes,
            duration_seconds=meeting.duration_seconds,
            audio_metadata=AudioMetadata.model_validate(meeting.audio_metadata),
            audio_quality=AudioQuality.model_validate(quality) if quality else None,
            languages_hint=meeting.languages_hint,
            expected_speakers=meeting.expected_speakers,
            detected_languages=[x for x in (meeting.detected_languages or "").split(",") if x],
            num_speakers=meeting.num_speakers,
            sha256=meeting.sha256,
            raw_audio_purged_at=meeting.raw_audio_purged_at,
            status=meeting.status,
            error=meeting.error,
            created_at=meeting.created_at,
            updated_at=meeting.updated_at,
        )

    # --- runs and stage outputs ---

    async def run_for(
        self, meeting_id: uuid.UUID, run_id: uuid.UUID | None = None
    ) -> ProcessingRun | None:
        """``run_id`` if given (404 if unknown), else the latest successful, else the latest run."""
        await self.get(meeting_id)
        if run_id is not None:
            run = await self.uow.results.get_run(meeting_id, run_id)
            if run is None:
                raise RunNotFoundError(
                    f"Run {run_id} not found for this meeting.",
                    details={"meeting_id": str(meeting_id), "run_id": str(run_id)},
                )
            return run
        return await self.default_run(meeting_id)

    async def stage_output(
        self, meeting_id: uuid.UUID, stage: str, run_id: uuid.UUID | None = None
    ) -> Any | None:
        run = await self.run_for(meeting_id, run_id)
        if run is None:
            return None
        return await load_stage_output(self.uow.results, self.store, run.id, stage)

    def _details(self, meeting: Meeting) -> dict[str, str]:
        return {"meeting_id": str(meeting.id), "status": meeting.status.value}

    async def get_diarization(
        self, meeting_id: uuid.UUID, run_id: uuid.UUID | None = None
    ) -> DiarizationResult:
        output = await self.stage_output(meeting_id, "diarize", run_id)
        if output is None:
            raise DiarizationNotAvailableError(
                "Speaker diarization is not available yet. Run POST /meetings/{id}/process "
                "and wait for status 'completed'.",
                details=self._details(await self.get(meeting_id)),
            )
        return DiarizationResult.model_validate(output)

    async def get_transcript(self, meeting_id: uuid.UUID) -> ASRResult:
        output = await self.stage_output(meeting_id, "transcribe")
        if output is None:
            raise TranscriptNotAvailableError(
                "The transcript is not available yet. Run POST /meetings/{id}/process "
                "and wait for status 'completed'.",
                details=self._details(await self.get(meeting_id)),
            )
        return ASRResult.model_validate(output)

    async def get_language_summary(self, meeting_id: uuid.UUID) -> LanguageSummary:
        output = await self.stage_output(meeting_id, "identify_languages")
        if output is None:
            raise LanguageSummaryNotAvailableError(
                "Language information is not available. It is produced by "
                "POST /meetings/{id}/process when LANGUAGE_ROUTING_ENABLED=true.",
                details=self._details(await self.get(meeting_id)),
            )
        return LanguageSummary.model_validate(output)

    async def speaker_names(self, meeting_id: uuid.UUID) -> dict[str, str]:
        return {
            s.label: s.display_name
            for s in await self.uow.results.speakers(meeting_id)
            if s.display_name
        }

    async def get_speaker_transcript(
        self, meeting_id: uuid.UUID, run_id: uuid.UUID | None = None
    ) -> tuple[SpeakerTranscript, dict[str, str]]:
        """The aligned transcript with display names applied, plus the name mapping."""
        output = await self.stage_output(meeting_id, "align", run_id)
        if output is None:
            raise TranscriptNotAvailableError(
                "The speaker-attributed transcript is not available yet. Run "
                "POST /meetings/{id}/process and wait for status 'completed'; the raw ASR "
                "output may already be available with ?view=raw.",
                details=self._details(await self.get(meeting_id)),
            )
        names = await self.speaker_names(meeting_id)
        transcript = SpeakerTranscript.model_validate(output)
        for utterance in transcript.utterances:
            utterance.speaker_name = names.get(utterance.speaker)
        return transcript, names

    async def get_analytics(
        self, meeting_id: uuid.UUID, run_id: uuid.UUID | None = None
    ) -> tuple[ConversationAnalytics, dict[str, str]]:
        """Conversation analytics with display names applied, plus the name mapping."""
        output = await self.stage_output(meeting_id, "analytics", run_id)
        if output is None:
            raise AnalyticsNotAvailableError(
                "Analytics are not available yet. Run POST /meetings/{id}/process "
                "and wait for status 'completed'.",
                details=self._details(await self.get(meeting_id)),
            )
        names = await self.speaker_names(meeting_id)
        analytics = ConversationAnalytics.model_validate(output)
        for speaker in analytics.speakers:
            speaker.speaker_name = names.get(speaker.speaker)
        return analytics, names

    async def summary_error(self, run: ProcessingRun | None) -> str | None:
        if run is None:
            return None
        row = await self.uow.results.stage_result(run.id, "summarize")
        return row.error if row is not None and row.status == "failed" else None

    async def get_summary(
        self, meeting_id: uuid.UUID, run_id: uuid.UUID | None = None
    ) -> tuple[MeetingSummary, dict[str, str]]:
        """The latest minutes of the run (speaker labels; names applied at render time)."""
        run = await self.run_for(meeting_id, run_id)
        record = await self.uow.results.latest_summary(run.id) if run else None
        if record is None:
            meeting = await self.get(meeting_id)
            error = await self.summary_error(run)
            raise SummaryNotAvailableError(
                "Summarization failed; transcript and analytics are still available. "
                "Retry with POST /meetings/{id}/summary/regenerate."
                if error
                else "The summary is not available yet. Run POST /meetings/{id}/process "
                "and wait for status 'completed'.",
                details={**self._details(meeting), "summary_error": error},
            )
        return MeetingSummary.model_validate(record.content), await self.speaker_names(meeting_id)

    async def check_summary_regeneration(self, meeting_id: uuid.UUID) -> None:
        """Regeneration needs a speaker transcript and no pipeline run in progress."""
        meeting = await self.get(meeting_id)
        if meeting.status == MeetingStatus.PROCESSING:
            raise MeetingStateConflictError(
                "Meeting is still processing.", details=self._details(meeting)
            )
        run = await self.uow.results.latest_run(meeting_id)
        if run is None or await self.uow.results.stage_result(run.id, "align") is None:
            raise TranscriptNotAvailableError(
                "The speaker-attributed transcript is not available yet. Process the meeting "
                "first.",
                details=self._details(meeting),
            )

    # --- speakers ---

    async def rename_speakers(
        self, meeting_id: uuid.UUID, names: dict[str, str | None]
    ) -> dict[str, str]:
        """Set display names for diarization labels; ``None`` removes one.

        Original labels are never changed; names are applied when rendering. Cached
        exports and charts of the default run are invalidated.
        """
        meeting = await self.get(meeting_id)
        known = {s.label for s in await self.uow.results.speakers(meeting_id)}
        if not known:
            raise DiarizationNotAvailableError(
                "Speakers are not known yet. Process the meeting first.",
                details=self._details(meeting),
            )
        unknown = sorted(set(names) - known)
        if unknown:
            raise ValidationError(
                "Unknown speaker label(s).", details={"unknown": unknown, "speakers": sorted(known)}
            )
        cleaned = {label: ((name or "").strip()[:100] or None) for label, name in names.items()}
        current = await self.uow.results.set_display_names(meeting_id, cleaned)
        await self.uow.commit()
        await self.invalidate_rendered(meeting_id)
        return current

    async def invalidate_rendered(self, meeting_id: uuid.UUID) -> None:
        """Drop cached exports and charts of every run (they embed display names)."""
        for run in await self.uow.results.list_runs(meeting_id):
            for folder in ("exports/", "charts/"):
                await self.store.delete_prefix(run_key(meeting_id, run.id, folder))

    # --- lifecycle ---

    async def request_processing(self, meeting_id: uuid.UUID, *, force: bool = False) -> Meeting:
        """Mark a meeting as ``processing`` so the pipeline can be scheduled.

        Setting the status here, before the background task starts, makes a
        second request see ``processing`` and get a 409 instead of double-running.
        """
        meeting = await self.get(meeting_id)
        busy = (
            MeetingStatus.PROCESSING,
            MeetingStatus.COMPLETED,
            MeetingStatus.COMPLETED_WITH_ERRORS,
        )
        if meeting.status in busy and not force:
            raise MeetingStateConflictError(
                f"Meeting is already {meeting.status.value}. Use ?force=true to reprocess.",
                details=self._details(meeting),
            )
        if meeting.raw_audio_purged_at is not None:
            raise MeetingStateConflictError(
                "The raw audio was purged by the retention policy; the meeting cannot be "
                "reprocessed.",
                details=self._details(meeting),
            )
        meeting.status = MeetingStatus.PROCESSING
        meeting.error = None
        await self.uow.commit()
        logger.info("processing_requested", meeting_id=str(meeting_id), force=force)
        return meeting

    async def delete(self, meeting_id: uuid.UUID) -> None:
        """Delete rows in one transaction, then artifacts; failed deletions are logged.

        Rows go first so a crash never leaves a meeting pointing at missing files;
        orphaned files are retried by ``scripts/cleanup.py``.
        """
        meeting = await self.get(meeting_id)
        keys = [k for k in (meeting.upload_key, meeting.processed_key) if k]
        await self.uow.search.delete_meeting(meeting_id)
        await self.uow.meetings.delete(meeting)
        await self.uow.commit()
        await self.delete_artifacts([meeting_prefix(meeting_id), *keys])
        logger.info("meeting_deleted", meeting_id=str(meeting_id))

    async def delete_artifacts(self, keys: list[str]) -> None:
        failed = False
        for key in keys:
            try:
                if key.endswith("/"):
                    await self.store.delete_prefix(key)
                else:
                    await self.store.delete(key)
            except Exception as exc:
                failed = True
                logger.warning("artifact_delete_failed", key=key, error=str(exc))
                await self.uow.results.log_cleanup(key, str(exc))
        if failed:
            await self.uow.commit()
