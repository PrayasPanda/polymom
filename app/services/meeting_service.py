"""Meeting use cases: upload, fetch, list, delete, request processing."""

import uuid
from pathlib import Path

import structlog
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.exceptions import (
    DiarizationNotAvailableError,
    LanguageSummaryNotAvailableError,
    MeetingNotFoundError,
    MeetingStateConflictError,
    PolymomError,
    TranscriptNotAvailableError,
)
from app.core.logging import get_logger
from app.models.meeting import Meeting
from app.repositories.meeting_repository import MeetingRepository
from app.schemas.asr import ASRResult
from app.schemas.diarization import DiarizationResult
from app.schemas.language import LanguageSummary
from app.schemas.meeting import MeetingStatus
from app.services.audio.validator import AsyncReadable, MediaValidator

logger = get_logger(__name__)


class MeetingService:
    def __init__(
        self, repository: MeetingRepository, validator: MediaValidator, settings: Settings
    ) -> None:
        self._repo = repository
        self._validator = validator
        self._settings = settings

    async def create(
        self,
        *,
        source: AsyncReadable,
        filename: str | None,
        title: str | None,
        expected_speakers: int | None,
        languages: list[str],
    ) -> Meeting:
        """Validate and store an upload, then record it as ``queued``."""
        meeting_id = uuid.uuid4()
        with structlog.contextvars.bound_contextvars(meeting_id=str(meeting_id)):
            logger.info("upload_started", filename=filename)
            try:
                media = await self._validator.validate(
                    source, filename, self._settings.uploads_dir, str(meeting_id)
                )
            except PolymomError as exc:
                logger.warning("upload_rejected", code=exc.code, reason=exc.message)
                raise
            logger.info(
                "upload_validated",
                size_bytes=media.size_bytes,
                mime_type=media.mime_type,
                duration_seconds=media.metadata.duration_seconds,
                codec=media.metadata.codec,
            )

            meeting = Meeting(
                id=meeting_id,
                title=title,
                original_filename=media.original_filename,
                stored_path=str(media.path.resolve()),
                mime_type=media.mime_type,
                size_bytes=media.size_bytes,
                duration_seconds=media.metadata.duration_seconds,
                audio_metadata=media.metadata.model_dump(),
                languages_hint=languages,
                expected_speakers=expected_speakers,
                status=MeetingStatus.QUEUED,
            )
            try:
                await self._repo.add(meeting)
            except BaseException:
                media.path.unlink(missing_ok=True)
                logger.exception("upload_persist_failed")
                raise
            logger.info("meeting_queued")
            # TODO(prompt-3/10): enqueue the processing job here.
            return meeting

    async def get(self, meeting_id: uuid.UUID) -> Meeting:
        meeting = await self._repo.get(meeting_id)
        if meeting is None:
            raise MeetingNotFoundError(
                f"Meeting {meeting_id} not found.", details={"meeting_id": str(meeting_id)}
            )
        return meeting

    async def list(self, *, limit: int, offset: int) -> tuple[list[Meeting], int]:
        return await self._repo.list(limit=limit, offset=offset)

    async def get_diarization(self, meeting_id: uuid.UUID) -> DiarizationResult:
        meeting = await self.get(meeting_id)
        if meeting.diarization is None:
            raise DiarizationNotAvailableError(
                "Speaker diarization is not available yet. Run POST /meetings/{id}/process "
                "and wait for status 'completed'.",
                details={"meeting_id": str(meeting_id), "status": meeting.status.value},
            )
        return DiarizationResult.model_validate(meeting.diarization)

    async def get_transcript(self, meeting_id: uuid.UUID) -> ASRResult:
        meeting = await self.get(meeting_id)
        if meeting.transcript is None:
            raise TranscriptNotAvailableError(
                "The transcript is not available yet. Run POST /meetings/{id}/process "
                "and wait for status 'completed'.",
                details={"meeting_id": str(meeting_id), "status": meeting.status.value},
            )
        return ASRResult.model_validate(meeting.transcript)

    async def get_language_summary(self, meeting_id: uuid.UUID) -> LanguageSummary:
        meeting = await self.get(meeting_id)
        if meeting.language_summary is None:
            raise LanguageSummaryNotAvailableError(
                "Language information is not available. It is produced by "
                "POST /meetings/{id}/process when LANGUAGE_ROUTING_ENABLED=true.",
                details={"meeting_id": str(meeting_id), "status": meeting.status.value},
            )
        return LanguageSummary.model_validate(meeting.language_summary)

    async def request_processing(self, meeting_id: uuid.UUID, *, force: bool = False) -> Meeting:
        """Mark a meeting as ``processing`` so the pipeline can be scheduled.

        Setting the status here, before the background task starts, makes a
        second request see ``processing`` and get a 409 instead of double-running.
        """
        meeting = await self.get(meeting_id)
        busy = (MeetingStatus.PROCESSING, MeetingStatus.COMPLETED)
        if meeting.status in busy and not force:
            raise MeetingStateConflictError(
                f"Meeting is already {meeting.status.value}. Use ?force=true to reprocess.",
                details={"meeting_id": str(meeting_id), "status": meeting.status.value},
            )
        meeting.status = MeetingStatus.PROCESSING
        meeting.error = None
        meeting = await self._repo.save(meeting)
        logger.info("processing_requested", meeting_id=str(meeting_id), force=force)
        return meeting

    async def delete(self, meeting_id: uuid.UUID) -> None:
        """Remove the record, then its stored and processed files."""
        meeting = await self.get(meeting_id)
        await self._repo.delete(meeting)
        for path in (meeting.stored_path, meeting.processed_path):
            if path:
                await run_in_threadpool(Path(path).unlink, missing_ok=True)
        logger.info("meeting_deleted", meeting_id=str(meeting_id))
