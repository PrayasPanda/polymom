"""Minutes-of-Meeting pipeline: an ordered list of stages sharing a context.

Stages: preprocess -> diarize -> identify languages -> transcribe -> align -> analytics ->
summarize (language ID only when LANGUAGE_ROUTING_ENABLED).

Stages marked ``optional`` (summarization) may fail without failing the meeting:
earlier outputs are kept and the status becomes ``completed_with_errors``.
"""

import json
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import structlog
from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import PolymomError
from app.core.logging import get_logger
from app.db.base import utcnow
from app.repositories.artifacts import ArtifactStore, run_key
from app.repositories.unit_of_work import UnitOfWork, UnitOfWorkFactory
from app.schemas.analytics import ConversationAnalytics
from app.schemas.asr import ASRResult
from app.schemas.audio import PreprocessResult
from app.schemas.diarization import DiarizationResult
from app.schemas.language import LanguageSummary
from app.schemas.meeting import MeetingStatus
from app.schemas.summary import MeetingSummary
from app.schemas.transcript import SpeakerTranscript
from app.services.alignment.aligner import align_words
from app.services.alignment.utterances import build_transcript
from app.services.analytics.meeting_stats import build_analytics
from app.services.asr.router import ASRRouter
from app.services.asr.service import TranscriptionService
from app.services.audio.preprocessor import AudioPreprocessor
from app.services.diarization.base import DiarizationBackend
from app.services.diarization.service import DiarizationService
from app.services.language.base import LanguageIdentifier
from app.services.language.service import LanguageIdResult, LanguageIdService
from app.services.llm import build_llm_client
from app.services.llm.base import LLMClient
from app.services.llm.tracing import build_tracer
from app.services.stage_outputs import load_stage_output
from app.services.summarization.summarizer import Summarizer

logger = get_logger(__name__)


@dataclass
class PipelineContext:
    """State passed from stage to stage for one meeting."""

    meeting_id: uuid.UUID
    input_path: Path
    expected_speakers: int | None = None
    languages_hint: list[str] = field(default_factory=list)
    meeting_date: date | None = None
    processed_path: Path | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, int] = field(default_factory=dict)


class PipelineStage(ABC):
    """One step of the pipeline.

    ``run`` does the work and records results on the context; ``persist`` writes
    them through the run's :class:`RunWriter` (one unit of work per stage).
    """

    name: ClassVar[str]
    optional: ClassVar[bool] = False
    """An optional stage's failure is recorded but does not fail the meeting."""

    @abstractmethod
    async def run(self, context: PipelineContext) -> None: ...

    def output(self, context: PipelineContext) -> BaseModel | dict[str, Any] | None:
        """What to store as this stage's result. Default: ``context.outputs[name]``."""
        value = context.outputs.get(self.name)
        return value if isinstance(value, BaseModel | dict) else None

    def model_version(self, context: PipelineContext) -> str | None:
        """Model identifier recorded in ``processing_runs.model_versions``."""
        return None

    async def persist(self, context: PipelineContext, writer: "RunWriter") -> None:
        await writer.save_stage(
            self.name, self.output(context), duration_ms=context.timings_ms.get(self.name)
        )


class PreprocessStage(PipelineStage):
    name = "preprocess"

    def __init__(self, preprocessor: AudioPreprocessor) -> None:
        self._preprocessor = preprocessor

    async def run(self, context: PipelineContext) -> None:
        result = await self._preprocessor.process(context.meeting_id, context.input_path)
        context.processed_path = result.processed_path
        context.outputs[self.name] = result

    def output(self, context: PipelineContext) -> dict[str, Any]:
        result: PreprocessResult = context.outputs[self.name]
        return result.model_dump(mode="json", exclude={"processed_path"})


class DiarizationStage(PipelineStage):
    """Who spoke when, on the preprocessed audio. Requires :class:`PreprocessStage`."""

    name = "diarize"

    def __init__(self, service: DiarizationService) -> None:
        self._service = service

    async def run(self, context: PipelineContext) -> None:
        if context.processed_path is None:
            raise RuntimeError("DiarizationStage requires PreprocessStage to run first")
        context.outputs[self.name] = await self._service.diarize(
            context.meeting_id, context.processed_path, num_speakers=context.expected_speakers
        )

    def model_version(self, context: PipelineContext) -> str | None:
        result: DiarizationResult = context.outputs[self.name]
        return result.model_name

    async def persist(self, context: PipelineContext, writer: "RunWriter") -> None:
        await super().persist(context, writer)
        result: DiarizationResult = context.outputs[self.name]
        labels = sorted({t.speaker_label for t in result.turns})
        await writer.uow.results.upsert_speakers(writer.meeting_id, labels)


class LanguageIdentificationStage(PipelineStage):
    """Spoken language per diarization turn, smoothed into language regions.

    Uses the diarization turns when :class:`DiarizationStage` ran; otherwise
    fixed windows over the whole recording.
    """

    name = "identify_languages"

    def __init__(self, service: LanguageIdService) -> None:
        self._service = service

    async def run(self, context: PipelineContext) -> None:
        if context.processed_path is None:
            raise RuntimeError("LanguageIdentificationStage requires PreprocessStage to run first")
        diarization: DiarizationResult | None = context.outputs.get("diarize")
        context.outputs[self.name] = await self._service.identify(
            context.meeting_id,
            context.processed_path,
            diarization.turns if diarization else [],
            context.languages_hint,
        )

    def output(self, context: PipelineContext) -> LanguageSummary:
        result: LanguageIdResult = context.outputs[self.name]
        return result.summary

    def model_version(self, context: PipelineContext) -> str | None:
        return self.output(context).lid_model


class TranscriptionStage(PipelineStage):
    """Timestamped transcript of the preprocessed audio.

    With language regions from :class:`LanguageIdentificationStage` each region
    is routed to its language's backend; otherwise the single-pass Prompt 5
    strategy (hint or auto-detect) is used. Word-to-speaker alignment comes later.
    """

    name = "transcribe"

    def __init__(self, service: TranscriptionService) -> None:
        self._service = service

    async def run(self, context: PipelineContext) -> None:
        if context.processed_path is None:
            raise RuntimeError("TranscriptionStage requires PreprocessStage to run first")
        language_id: LanguageIdResult | None = context.outputs.get(LanguageIdentificationStage.name)
        if language_id is not None and language_id.regions:
            result = await self._service.transcribe_routed(
                context.meeting_id,
                context.processed_path,
                language_id.regions,
                context.languages_hint,
            )
        else:
            result = await self._service.transcribe(
                context.meeting_id, context.processed_path, context.languages_hint
            )
        context.outputs[self.name] = result

    def model_version(self, context: PipelineContext) -> str | None:
        result: ASRResult = context.outputs[self.name]
        return ",".join(result.model_names) or None

    async def persist(self, context: PipelineContext, writer: "RunWriter") -> None:
        await super().persist(context, writer)
        language_id: LanguageIdResult | None = context.outputs.get(LanguageIdentificationStage.name)
        if language_id is not None:
            result: ASRResult = context.outputs[self.name]
            language_id.summary.code_mixed_segments = sum(s.is_code_mixed for s in result.segments)
            await writer.save_stage(
                LanguageIdentificationStage.name,
                language_id.summary,
                duration_ms=context.timings_ms.get(LanguageIdentificationStage.name),
            )


class AlignmentStage(PipelineStage):
    """Speaker-attributed transcript: transcript words assigned to diarization turns.

    Speaker labels are taken as-is from diarization ("Person N"), never renumbered.
    """

    name = "align"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def run(self, context: PipelineContext) -> None:
        transcript: ASRResult | None = context.outputs.get(TranscriptionStage.name)
        if transcript is None:
            raise RuntimeError("AlignmentStage requires TranscriptionStage to run first")
        diarization: DiarizationResult | None = context.outputs.get(DiarizationStage.name)
        turns = diarization.turns if diarization else []
        settings = self._settings
        preprocess: PreprocessResult | None = context.outputs.get(PreprocessStage.name)
        words = align_words(transcript.segments, turns, settings.align_max_gap_seconds)
        context.outputs[self.name] = build_transcript(
            words,
            turns,
            total_duration=preprocess.duration_seconds if preprocess else 0.0,
            max_seconds=settings.utterance_max_seconds,
            min_words=settings.utterance_min_words,
            merge_gap=settings.align_merge_gap_seconds,
        )

    async def persist(self, context: PipelineContext, writer: "RunWriter") -> None:
        await super().persist(context, writer)
        result: SpeakerTranscript = context.outputs[self.name]
        uow = writer.uow
        await uow.results.replace_utterances(writer.meeting_id, writer.run_id, result.utterances)
        await uow.search.index_utterances(writer.meeting_id, writer.run_id, result.utterances)
        await uow.results.upsert_speakers(writer.meeting_id, result.speakers)


class AnalyticsStage(PipelineStage):
    """Speaker-wise and meeting-level conversation statistics. Requires :class:`AlignmentStage`."""

    name = "analytics"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def run(self, context: PipelineContext) -> None:
        transcript: SpeakerTranscript | None = context.outputs.get(AlignmentStage.name)
        if transcript is None:
            raise RuntimeError("AnalyticsStage requires AlignmentStage to run first")
        diarization: DiarizationResult | None = context.outputs.get(DiarizationStage.name)
        language_id: LanguageIdResult | None = context.outputs.get(LanguageIdentificationStage.name)
        settings = self._settings
        context.outputs[self.name] = build_analytics(
            diarization.turns if diarization else [],
            transcript,
            language_summary=language_id.summary if language_id else None,
            interruption_min_overlap=settings.interruption_min_overlap_seconds,
            bucket_seconds=settings.bucket_seconds,
            gini_balanced_max=settings.gini_balanced_max,
            gini_dominated_min=settings.gini_dominated_min,
        )

    async def persist(self, context: PipelineContext, writer: "RunWriter") -> None:
        await super().persist(context, writer)
        result: ConversationAnalytics = context.outputs[self.name]
        stats = {s.speaker: s.model_dump(mode="json") for s in result.speakers}
        await writer.uow.results.upsert_speakers(writer.meeting_id, list(stats), stats)


class SummarizationStage(PipelineStage):
    """LLM minutes (summary, decisions, action items) grounded in the aligned transcript.

    Analytics are passed as compact context only. Optional: a provider outage leaves
    the transcript and analytics available and marks the meeting ``completed_with_errors``.
    """

    name = "summarize"
    optional = True

    def __init__(self, settings: Settings, llm: LLMClient | None = None) -> None:
        self._settings = settings
        self._llm = llm

    async def summarize(
        self,
        meeting_id: uuid.UUID,
        transcript: SpeakerTranscript,
        analytics: ConversationAnalytics | None,
        *,
        meeting_date: date | None = None,
        output_language: str | None = None,
        model: str | None = None,
    ) -> MeetingSummary:
        llm = build_llm_client(self._settings, model) if model or self._llm is None else self._llm
        tracer = build_tracer(self._settings, str(meeting_id))
        return await Summarizer(llm, self._settings, tracer).summarize(
            transcript, analytics, output_language=output_language, meeting_date=meeting_date
        )

    async def run(self, context: PipelineContext) -> None:
        transcript: SpeakerTranscript | None = context.outputs.get(AlignmentStage.name)
        if transcript is None:
            raise RuntimeError("SummarizationStage requires AlignmentStage to run first")
        context.outputs[self.name] = await self.summarize(
            context.meeting_id,
            transcript,
            context.outputs.get(AnalyticsStage.name),
            meeting_date=context.meeting_date,
        )

    def model_version(self, context: PipelineContext) -> str | None:
        result: MeetingSummary = context.outputs[self.name]
        return f"{result.model_info.provider}/{result.model_info.model}"

    async def persist(self, context: PipelineContext, writer: "RunWriter") -> None:
        await save_summary(writer, context.outputs[self.name], context.timings_ms.get(self.name))


async def save_summary(
    writer: "RunWriter", summary: MeetingSummary, duration_ms: int | None
) -> None:
    """Summaries live in their own table; the stage row only records status and timing."""
    await writer.save_stage(SummarizationStage.name, None, duration_ms=duration_ms)
    await writer.uow.results.add_summary(writer.meeting_id, writer.run_id, summary)
    await writer.uow.search.index_summary(writer.meeting_id, writer.run_id, summary)


def _error_message(exc: Exception, stage: str) -> str:
    if isinstance(exc, PolymomError):
        return f"{exc.code}: {exc.message}"
    return f"internal_error: Unexpected failure in stage '{stage}'."


def config_snapshot(settings: Settings) -> dict[str, Any]:
    """Every setting, secrets masked (SecretStr dumps as ``**********``)."""
    return settings.model_dump(mode="json")


@dataclass
class RunWriter:
    """Writes one run's stage results through a unit of work.

    Outputs larger than ``STAGE_OUTPUT_INLINE_MAX_BYTES`` are stored as artifacts
    (``meetings/{id}/{run_id}/stages/{stage}.json``) and referenced by key.
    """

    uow: UnitOfWork
    store: ArtifactStore
    meeting_id: uuid.UUID
    run_id: uuid.UUID
    inline_max_bytes: int

    async def save_stage(
        self,
        name: str,
        output: BaseModel | dict[str, Any] | None,
        *,
        status: str = "completed",
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        data = output.model_dump(mode="json") if isinstance(output, BaseModel) else output
        ref = None
        if data is not None:
            raw = json.dumps(data, ensure_ascii=False).encode()
            if len(raw) > self.inline_max_bytes:
                ref = run_key(self.meeting_id, self.run_id, f"stages/{name}.json")
                await self.store.put(ref, raw, "application/json")
                data = None
        await self.uow.results.save_stage(
            self.run_id,
            name,
            status=status,
            output=data,
            output_ref=ref,
            duration_ms=duration_ms,
            error=error,
        )


class MoMPipeline:
    """Runs stages in order, records a :class:`ProcessingRun`, owns the status lifecycle.

    ``processing`` -> ``completed`` / ``completed_with_errors`` (an optional stage
    failed) / ``failed`` with ``"<error code>: <message>"``. Each stage's results are
    committed as soon as it finishes, so a later failure never discards them.
    """

    def __init__(
        self,
        stages: Sequence[PipelineStage],
        uow_factory: UnitOfWorkFactory,
        store: ArtifactStore,
        settings: Settings,
    ) -> None:
        if not stages:
            raise ValueError("a pipeline needs at least one stage")
        self.stages = list(stages)
        self._uow = uow_factory
        self._store = store
        self._settings = settings

    def _writer(self, uow: UnitOfWork, meeting_id: uuid.UUID, run_id: uuid.UUID) -> RunWriter:
        return RunWriter(
            uow, self._store, meeting_id, run_id, self._settings.stage_output_inline_max_bytes
        )

    async def run(self, meeting_id: uuid.UUID) -> MeetingStatus | None:
        """Process one meeting. Never raises for stage failures; returns the final status."""
        with structlog.contextvars.bound_contextvars(meeting_id=str(meeting_id)):
            async with self._uow() as uow:
                meeting = await uow.meetings.get(meeting_id)
                if meeting is None:
                    logger.warning("pipeline_meeting_missing")
                    return None
                meeting.status = MeetingStatus.PROCESSING
                meeting.error = None
                run = await uow.results.create_run(meeting_id, config_snapshot(self._settings))
                run_id, upload_key = run.id, meeting.upload_key
                context = PipelineContext(
                    meeting_id=meeting_id,
                    input_path=Path(),
                    expected_speakers=meeting.expected_speakers,
                    languages_hint=list(meeting.languages_hint or []),
                    meeting_date=meeting.created_at.date() if meeting.created_at else None,
                )
                await uow.commit()

            with structlog.contextvars.bound_contextvars(run_id=str(run_id)):
                status, error, versions = await self._run_stages(context, upload_key, run_id)
                processed_key = await self._store_processed_audio(context, run_id)
                await self._finish(context, run_id, status, error, versions, processed_key)
            return status

    async def _run_stages(
        self, context: PipelineContext, upload_key: str, run_id: uuid.UUID
    ) -> tuple[MeetingStatus, str | None, dict[str, str]]:
        logger.info("pipeline_started", stages=[s.name for s in self.stages])
        errors: list[str] = []
        versions: dict[str, str] = {}
        current = self.stages[0].name
        try:
            async with self._store.local_path(upload_key) as input_path:
                context.input_path = input_path
                for stage in self.stages:
                    current = stage.name
                    started = time.perf_counter()
                    try:
                        await stage.run(context)
                    except Exception as exc:
                        message = _error_message(exc, stage.name)
                        await self._save_failed_stage(context, run_id, stage.name, message)
                        if not stage.optional:
                            raise
                        errors.append(f"{stage.name}: {message}")
                        logger.warning("optional_stage_failed", stage=stage.name, error=message)
                        continue
                    context.timings_ms[stage.name] = int((time.perf_counter() - started) * 1000)
                    async with self._uow() as uow:
                        await stage.persist(context, self._writer(uow, context.meeting_id, run_id))
                        await uow.commit()
                    if version := stage.model_version(context):
                        versions[stage.name] = version
                    logger.info(
                        "stage_completed",
                        stage=stage.name,
                        duration_ms=context.timings_ms[stage.name],
                    )
        except FileNotFoundError:
            logger.warning("upload_missing", key=upload_key)
            return (
                MeetingStatus.FAILED,
                "upload_missing: The uploaded file is no longer stored.",
                versions,
            )
        except PolymomError as exc:
            logger.warning("stage_failed", stage=current, code=exc.code, error=exc.message)
            return MeetingStatus.FAILED, f"{exc.code}: {exc.message}", versions
        except Exception:
            logger.exception("stage_crashed", stage=current)
            return (
                MeetingStatus.FAILED,
                f"internal_error: Unexpected failure in stage '{current}'.",
                versions,
            )
        status = MeetingStatus.COMPLETED_WITH_ERRORS if errors else MeetingStatus.COMPLETED
        return status, "; ".join(errors) or None, versions

    async def _save_failed_stage(
        self, context: PipelineContext, run_id: uuid.UUID, stage: str, message: str
    ) -> None:
        try:
            async with self._uow() as uow:
                writer = self._writer(uow, context.meeting_id, run_id)
                await writer.save_stage(stage, None, status="failed", error=message)
                await uow.commit()
        except Exception:  # pragma: no cover - never mask the stage error
            logger.exception("stage_failure_not_recorded", stage=stage)

    async def _store_processed_audio(
        self, context: PipelineContext, run_id: uuid.UUID
    ) -> str | None:
        path = context.processed_path
        if path is None or not path.is_file():
            return None
        key = run_key(context.meeting_id, run_id, "processed.wav")
        try:
            await self._store.put_file(key, path, "audio/wav")
        except Exception:  # pragma: no cover - results matter more than the audio copy
            logger.exception("processed_audio_upload_failed")
            return None
        path.unlink(missing_ok=True)
        return key

    async def _finish(
        self,
        context: PipelineContext,
        run_id: uuid.UUID,
        status: MeetingStatus,
        error: str | None,
        versions: dict[str, str],
        processed_key: str | None,
    ) -> None:
        async with self._uow() as uow:
            meeting = await uow.meetings.get(context.meeting_id)
            run = await uow.results.get_run(context.meeting_id, run_id)
            if meeting is None or run is None:  # deleted while processing
                return
            meeting.status = status
            meeting.error = error
            if processed_key:
                meeting.processed_key = processed_key
            transcript: SpeakerTranscript | None = context.outputs.get(AlignmentStage.name)
            if status != MeetingStatus.FAILED and transcript is not None:
                langs = sorted(
                    {lang for u in transcript.utterances for lang in u.languages_present}
                )
                meeting.detected_languages = f",{','.join(langs)}," if langs else None
                meeting.num_speakers = len(transcript.speakers) or None
            run.status = status.value
            run.error = error
            run.finished_at = utcnow()
            run.timings_ms = dict(context.timings_ms)
            run.model_versions = versions
            await uow.commit()
        logger.info("pipeline_finished", status=status.value, timings_ms=context.timings_ms)


def build_pipeline(
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    store: ArtifactStore,
    diarization_backend: DiarizationBackend,
    asr_router: ASRRouter,
    language_identifier: LanguageIdentifier | None = None,
    llm: LLMClient | None = None,
) -> MoMPipeline:
    """Default stage registry."""
    stages: list[PipelineStage] = [
        PreprocessStage(AudioPreprocessor(settings)),
        DiarizationStage(DiarizationService(diarization_backend, settings)),
    ]
    if settings.language_routing_enabled and language_identifier is not None:
        stages.append(LanguageIdentificationStage(LanguageIdService(language_identifier, settings)))
    stages.append(TranscriptionStage(TranscriptionService(asr_router, settings)))
    stages.append(AlignmentStage(settings))
    stages.append(AnalyticsStage(settings))
    stages.append(SummarizationStage(settings, llm))
    return MoMPipeline(stages, uow_factory, store, settings)


class SummaryRegenerator:
    """Re-runs only summarization on the latest successful run (background task).

    The new summary is added to that run (older ones stay in ``summaries``) and the
    run's cached exports are invalidated.
    """

    def __init__(
        self,
        stage: SummarizationStage,
        uow_factory: UnitOfWorkFactory,
        store: ArtifactStore,
        settings: Settings,
    ) -> None:
        self._stage = stage
        self._uow = uow_factory
        self._store = store
        self._settings = settings

    async def run(
        self, meeting_id: uuid.UUID, *, output_language: str | None, model: str | None
    ) -> None:
        with structlog.contextvars.bound_contextvars(meeting_id=str(meeting_id)):
            async with self._uow() as uow:
                meeting = await uow.meetings.get(meeting_id)
                run = await uow.results.latest_run(meeting_id) if meeting else None
                raw = (
                    await load_stage_output(uow.results, self._store, run.id, AlignmentStage.name)
                    if run
                    else None
                )
                if meeting is None or run is None or raw is None:
                    logger.warning("summary_regenerate_skipped")
                    return
                analytics_raw = await load_stage_output(
                    uow.results, self._store, run.id, AnalyticsStage.name
                )
                run_id = run.id
                meeting_date = meeting.created_at.date() if meeting.created_at else None
            started = time.perf_counter()
            try:
                summary = await self._stage.summarize(
                    meeting_id,
                    SpeakerTranscript.model_validate(raw),
                    ConversationAnalytics.model_validate(analytics_raw) if analytics_raw else None,
                    meeting_date=meeting_date,
                    output_language=output_language,
                    model=model,
                )
            except Exception as exc:
                message = _error_message(exc, SummarizationStage.name)
                logger.warning("summary_regenerate_failed", error=message)
                async with self._uow() as uow:
                    writer = RunWriter(uow, self._store, meeting_id, run_id, 0)
                    await writer.save_stage(
                        SummarizationStage.name, None, status="failed", error=message
                    )
                    meeting = await uow.meetings.get(meeting_id)
                    if meeting is not None:
                        meeting.status = MeetingStatus.COMPLETED_WITH_ERRORS
                        meeting.error = f"{SummarizationStage.name}: {message}"
                    await uow.commit()
                return
            async with self._uow() as uow:
                writer = RunWriter(
                    uow,
                    self._store,
                    meeting_id,
                    run_id,
                    self._settings.stage_output_inline_max_bytes,
                )
                await save_summary(writer, summary, int((time.perf_counter() - started) * 1000))
                meeting = await uow.meetings.get(meeting_id)
                if (
                    meeting is not None
                    and meeting.status == MeetingStatus.COMPLETED_WITH_ERRORS
                    and (meeting.error or "").startswith(f"{SummarizationStage.name}:")
                ):
                    meeting.status = MeetingStatus.COMPLETED
                    meeting.error = None
                await uow.commit()
            await self._store.delete_prefix(run_key(meeting_id, run_id, "exports/"))
            logger.info("summary_regenerated")
