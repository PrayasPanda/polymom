"""Minutes-of-Meeting pipeline: an ordered list of stages sharing a context.

Stages: preprocess -> diarize -> identify languages -> transcribe -> align -> analytics ->
summarize (language ID only when LANGUAGE_ROUTING_ENABLED).

Stages marked ``optional`` (summarization) may fail without failing the meeting:
earlier outputs are kept and the status becomes ``completed_with_errors``.
"""

import asyncio
import hashlib
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import structlog
from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import PolymomError, StageTimeoutError, ValidationError
from app.core.logging import get_logger
from app.core.tracing import span
from app.db.base import utcnow
from app.repositories.artifacts import ArtifactStore, run_key
from app.repositories.unit_of_work import UnitOfWork, UnitOfWorkFactory

if TYPE_CHECKING:
    from app.pipelines.checkpoints import ChunkHooks
    from app.workers.progress import ProgressReporter


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

_STAGE_OUTPUT_SCHEMAS: dict[str, type[BaseModel]] = dict(
    {
        "preprocess": PreprocessResult,
        "diarize": DiarizationResult,
        "identify_languages": LanguageSummary,
        "transcribe": ASRResult,
        "align": SpeakerTranscript,
        "analytics": ConversationAnalytics,
        "summarize": MeetingSummary,
    }
)


def _canonical(value: Any) -> Any:
    """A JSON-stable form of a setting. Sets are sorted: their iteration order depends on
    the per-process hash seed, and workers on different queues must agree on fingerprints."""
    if isinstance(value, set | frozenset):
        return sorted(_canonical(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _reify_stage_output(name: str, output: Any) -> BaseModel | None:
    """Rebuild the Pydantic output of a reused stage (``None`` if the stage has none)."""
    schema = _STAGE_OUTPUT_SCHEMAS.get(name)
    if schema is None or output is None:
        return None
    try:
        return schema.model_validate(output)
    except Exception:  # pragma: no cover - a corrupt blob just means the stage re-runs
        return None


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
    fingerprints: dict[str, str] = field(default_factory=dict)
    hooks_for: "Callable[[str], ChunkHooks | None] | None" = None
    """Chunk checkpoint/stop hooks for long stages (diarize, transcribe)."""

    def hooks(self, stage: str) -> "ChunkHooks | None":
        return self.hooks_for(stage) if self.hooks_for else None


class PipelineStage(ABC):
    """One step of the pipeline.

    ``run`` does the work and records results on the context; ``persist`` writes
    them through the run's :class:`RunWriter` (one unit of work per stage).
    """

    name: ClassVar[str]
    optional: ClassVar[bool] = False
    """An optional stage's failure is recorded but does not fail the meeting."""
    queue: ClassVar[str] = "cpu"
    """Worker queue that runs this stage: cpu, gpu or llm."""
    config_keys: ClassVar[tuple[str, ...]] = ()
    """Settings that change this stage's output; part of its fingerprint."""

    def fingerprint(self, settings: Settings, upstream: str) -> str:
        """sha256 of the stage name, its config (incl. model names) and the upstream hash."""
        payload = {
            "stage": self.name,
            "upstream": upstream,
            "config": {k: _canonical(getattr(settings, k, None)) for k in self.config_keys},
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()

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
            self.name,
            self.output(context),
            duration_ms=context.timings_ms.get(self.name),
            fingerprint=writer.fingerprint,
        )


class PreprocessStage(PipelineStage):
    name = "preprocess"
    config_keys = (
        "target_sample_rate",
        "target_loudness_lufs",
        "enable_highpass",
        "highpass_cutoff_hz",
        "enable_denoise",
        "trim_silence",
        "silence_threshold_db",
        "silence_min_duration_seconds",
    )

    def __init__(self, preprocessor: AudioPreprocessor) -> None:
        self._preprocessor = preprocessor

    async def run(self, context: PipelineContext) -> None:
        result = await self._preprocessor.process(context.meeting_id, context.input_path)
        context.processed_path = result.processed_path
        context.outputs[self.name] = result

    def output(self, context: PipelineContext) -> dict[str, Any]:
        result: PreprocessResult = context.outputs[self.name]
        return result.model_dump(mode="json", exclude={"processed_path"})

    async def persist(self, context: PipelineContext, writer: "RunWriter") -> None:
        """Store the processed audio right away so a worker on another host can continue."""
        await super().persist(context, writer)
        if context.processed_path is not None and context.processed_path.is_file():
            key = run_key(writer.meeting_id, writer.run_id, "processed.wav")
            await writer.store.put_file(key, context.processed_path, "audio/wav")


class DiarizationStage(PipelineStage):
    """Who spoke when, on the preprocessed audio. Requires :class:`PreprocessStage`."""

    name = "diarize"
    queue = "gpu"
    config_keys = (
        "diarization_backend",
        "diarization_model",
        "merge_gap_seconds",
        "min_turn_seconds",
        "diarization_chunk_threshold_seconds",
        "speaker_similarity_threshold",
        "chunk_length_seconds",
        "chunk_overlap_seconds",
    )

    def __init__(self, service: DiarizationService) -> None:
        self._service = service

    async def run(self, context: PipelineContext) -> None:
        if context.processed_path is None:
            raise RuntimeError("DiarizationStage requires PreprocessStage to run first")
        context.outputs[self.name] = await self._service.diarize(
            context.meeting_id,
            context.processed_path,
            num_speakers=context.expected_speakers,
            hooks=context.hooks(self.name),
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
    queue = "gpu"
    config_keys = (
        "lid_backend",
        "lid_model_id",
        "lid_min_confidence",
        "lid_min_window_seconds",
        "lid_max_window_seconds",
        "supported_languages",
    )

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
    queue = "gpu"
    config_keys = (
        "asr_backend",
        "whisper_model_size",
        "whisper_compute_type",
        "odia_model_id",
        "odia_decoding",
        "asr_language_backends",
        "asr_beam_size",
        "asr_vad_filter",
        "asr_low_confidence_threshold",
        "asr_compression_ratio_threshold",
        "asr_no_speech_threshold",
        "chunk_length_seconds",
        "chunk_overlap_seconds",
    )

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
                context.meeting_id,
                context.processed_path,
                context.languages_hint,
                hooks=context.hooks(self.name),
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
                fingerprint=context.fingerprints.get(LanguageIdentificationStage.name),
            )


class AlignmentStage(PipelineStage):
    """Speaker-attributed transcript: transcript words assigned to diarization turns.

    Speaker labels are taken as-is from diarization ("Person N"), never renumbered.
    """

    name = "align"
    config_keys = (
        "align_max_gap_seconds",
        "align_merge_gap_seconds",
        "utterance_max_seconds",
        "utterance_min_words",
    )

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
    config_keys = (
        "interruption_min_overlap_seconds",
        "bucket_seconds",
        "gini_balanced_max",
        "gini_dominated_min",
    )

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
    queue = "llm"
    config_keys = (
        "llm_provider",
        "llm_model",
        "llm_temperature",
        "summary_output_language",
        "summary_single_pass_tokens",
        "summary_chunk_tokens",
        "evidence_match_threshold",
    )

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
    await writer.save_stage(
        SummarizationStage.name, None, duration_ms=duration_ms, fingerprint=writer.fingerprint
    )
    await writer.uow.results.add_summary(writer.meeting_id, writer.run_id, summary)
    from app.core.metrics import LLM_COST, LLM_TOKENS

    usage = summary.model_info.usage
    LLM_TOKENS.labels("prompt").inc(usage.prompt_tokens)
    LLM_TOKENS.labels("completion").inc(usage.completion_tokens)
    LLM_COST.inc(usage.cost_usd)
    await writer.uow.search.index_summary(writer.meeting_id, writer.run_id, summary)


def _error_message(exc: Exception, stage: str) -> str:
    if isinstance(exc, PolymomError):
        return f"{exc.code}: {exc.message}"
    return f"internal_error: Unexpected failure in stage '{stage}'."


@dataclass
class _Outcome:
    """How a pass over the stages ended: finished, handed off to another queue, or interrupted."""

    kind: Literal["finished", "handoff", "interrupted"]
    status: MeetingStatus | None = None
    error: str | None = None
    queue: str | None = None


class TransientStageError(Exception):
    """A required stage failed with a retryable error; the job queue should retry.

    Completed stages are already committed, so the retry resumes at this stage.
    """

    def __init__(self, stage: str, cause: PolymomError) -> None:
        super().__init__(f"{stage}: {cause.code}: {cause.message}")
        self.stage = stage
        self.cause = cause
        self.run_id: uuid.UUID | None = None


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
    fingerprint: str | None = None

    async def save_stage(
        self,
        name: str,
        output: BaseModel | dict[str, Any] | None,
        *,
        status: str = "completed",
        duration_ms: int | None = None,
        error: str | None = None,
        fingerprint: str | None = None,
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
            fingerprint=fingerprint,
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

    async def run(
        self,
        meeting_id: uuid.UUID,
        *,
        run_id: uuid.UUID | None = None,
        from_stage: str | None = None,
        cancel_check: Callable[[uuid.UUID], Awaitable[bool]] | None = None,
        progress: "ProgressReporter | None" = None,
        queue: str | None = None,
        handoff: Callable[[uuid.UUID, str], Awaitable[None]] | None = None,
        shutdown_check: Callable[[], bool] | None = None,
    ) -> MeetingStatus | None:
        """Process one meeting, or continue an existing run.

        - ``run_id``: continue that run (a hand-off, a retry after a crash, or a
          re-queue by the stuck-job reaper). Stages already completed in the run are
          skipped when their stored fingerprint still matches; the rest run.
        - ``from_stage``: new run that reuses the latest successful run's outputs
          for every stage before this one (e.g. rerun only summarization).
        - ``queue`` + ``handoff``: this worker only runs stages of its own queue
          (cpu, gpu or llm); at the first stage of another queue it calls
          ``handoff(run_id, that_queue)`` and returns ``PROCESSING``.
        - ``cancel_check`` / ``shutdown_check``: checked between stages and chunks.
          Cancel marks the meeting ``cancelled``; shutdown leaves the run
          ``processing`` so the job can be retried and resume where it stopped.

        Never raises for stage failures; returns the meeting status (``None`` if the
        meeting does not exist).
        """
        from app.workers.progress import NullProgressReporter

        reporter = progress or NullProgressReporter()
        with structlog.contextvars.bound_contextvars(meeting_id=str(meeting_id)):
            async with self._uow() as uow:
                meeting = await uow.meetings.get(meeting_id)
                if meeting is None:
                    logger.warning("pipeline_meeting_missing")
                    return None
                meeting.status = MeetingStatus.PROCESSING
                meeting.error = None
                context = PipelineContext(
                    meeting_id=meeting_id,
                    input_path=Path(),
                    expected_speakers=meeting.expected_speakers,
                    languages_hint=list(meeting.languages_hint or []),
                    meeting_date=meeting.created_at.date() if meeting.created_at else None,
                )
                self._compute_fingerprints(context, meeting.sha256 or meeting.upload_key)
                upload_key, audio_seconds = meeting.upload_key, meeting.duration_seconds
                skipped: set[str] = set()
                reused_label = "skipped"  # outputs copied from an earlier run
                run = await uow.results.get_run(meeting_id, run_id) if run_id else None
                if run is not None:
                    skipped = await self._load_completed_stages(uow, run.id, context)
                    reused_label = "completed"  # finished by an earlier job of this run
                else:
                    run = await uow.results.create_run(meeting_id, config_snapshot(self._settings))
                    if from_stage is not None:
                        skipped = await self._seed_from_previous_run(
                            uow, meeting_id, run.id, from_stage, context
                        )
                run.status = "processing"
                current_run_id = run.id
                run_started = run.started_at.timestamp()
                await uow.commit()

            await reporter.start(
                current_run_id, [s.name for s in self.stages], audio_seconds, run_started
            )
            for name in skipped:
                await reporter.stage(name, reused_label)
            with structlog.contextvars.bound_contextvars(run_id=str(current_run_id)):
                attempt = 0
                while True:
                    try:
                        outcome = await self._run_stages(
                            context,
                            upload_key,
                            current_run_id,
                            skipped,
                            cancel_check,
                            reporter,
                            queue,
                            handoff,
                            shutdown_check,
                        )
                        break
                    except TransientStageError as transient:
                        from app.core.metrics import STAGE_RETRIES

                        STAGE_RETRIES.labels(transient.stage).inc()
                        transient.run_id = current_run_id
                        attempt += 1
                        if queue is not None:
                            raise  # the job queue retries with backoff (see app.workers.main)
                        if attempt > self._settings.max_retries:
                            outcome = _Outcome(
                                "finished",
                                MeetingStatus.FAILED,
                                f"{transient.cause.code}: {transient.cause.message}",
                            )
                            break
                        logger.warning("stage_retry_inline", stage=transient.stage, attempt=attempt)
                        await asyncio.sleep(
                            self._settings.retry_backoff_seconds * 2 ** (attempt - 1)
                        )
                        async with self._uow() as uow:
                            skipped = await self._load_completed_stages(
                                uow, current_run_id, context
                            )
                if outcome.kind != "finished":
                    self._cleanup_work_file(context)
                    logger.info("pipeline_paused", reason=outcome.kind, next_queue=outcome.queue)
                    return MeetingStatus.PROCESSING
                assert outcome.status is not None  # noqa: S101 - finished always has a status
                await self._finish(context, current_run_id, outcome.status, outcome.error)
                self._cleanup_work_file(context)
            await reporter.finish(outcome.status, outcome.error)
            return outcome.status

    def _compute_fingerprints(self, context: PipelineContext, root: str) -> None:
        upstream = hashlib.sha256(
            json.dumps(
                {
                    "input": root,
                    "languages_hint": context.languages_hint,
                    "expected_speakers": context.expected_speakers,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        for stage in self.stages:
            upstream = stage.fingerprint(self._settings, upstream)
            context.fingerprints[stage.name] = upstream

    async def _load_completed_stages(
        self, uow: UnitOfWork, run_id: uuid.UUID, context: PipelineContext
    ) -> set[str]:
        """Rehydrate stages this run already completed, if their fingerprint still matches.

        A mismatch (settings changed since the crash) makes that stage and every
        later one run again: fingerprints chain, so downstream hashes differ too.
        """
        done: set[str] = set()
        for stage in self.stages:
            row = await uow.results.stage_result(run_id, stage.name)
            if (
                row is None
                or row.status not in ("completed", "skipped")
                or row.fingerprint != context.fingerprints.get(stage.name)
            ):
                break
            output = await load_stage_output(uow.results, self._store, run_id, stage.name)
            if row.status == "completed" and output is None and stage.name != "summarize":
                break
            reified: Any = _reify_stage_output(stage.name, output)
            if stage.name == LanguageIdentificationStage.name and reified is not None:
                assert isinstance(reified, LanguageSummary)  # noqa: S101
                reified = LanguageIdResult(
                    regions=reified.regions, summary=reified, processing_time_ms=0
                )
            if reified is not None:
                context.outputs[stage.name] = reified
            if row.duration_ms is not None:
                context.timings_ms[stage.name] = row.duration_ms
            done.add(stage.name)
        if done:
            logger.info("pipeline_resumed", completed=sorted(done))
        return done

    async def _seed_from_previous_run(
        self,
        uow: UnitOfWork,
        meeting_id: uuid.UUID,
        run_id: uuid.UUID,
        from_stage: str,
        context: PipelineContext,
    ) -> set[str]:
        """Copy every stage before ``from_stage`` from the latest successful run."""
        stage_names = [s.name for s in self.stages]
        if from_stage not in stage_names:
            raise ValidationError(
                f"Unknown stage {from_stage!r}.",
                details={"stages": stage_names, "from_stage": from_stage},
            )
        prior = await uow.results.latest_run(meeting_id, successful=True)
        if prior is None:
            raise ValidationError(
                "from_stage requires a previous successful run.",
                details={"meeting_id": str(meeting_id)},
            )
        writer = self._writer(uow, meeting_id, run_id)
        skipped: set[str] = set()
        for name in stage_names[: stage_names.index(from_stage)]:
            output = await load_stage_output(uow.results, self._store, prior.id, name)
            if output is None:
                continue
            reified: Any = _reify_stage_output(name, output)
            if name == LanguageIdentificationStage.name and isinstance(reified, LanguageSummary):
                reified = LanguageIdResult(
                    regions=reified.regions, summary=reified, processing_time_ms=0
                )
            if reified is not None:
                context.outputs[name] = reified
            await writer.save_stage(
                name, output, status="skipped", fingerprint=context.fingerprints.get(name)
            )
            skipped.add(name)
        prior_audio = run_key(meeting_id, prior.id, "processed.wav")
        if PreprocessStage.name in skipped and await self._store.exists(prior_audio):
            await self._store.put(
                run_key(meeting_id, run_id, "processed.wav"), await self._store.get(prior_audio)
            )
        return skipped

    async def _run_stages(
        self,
        context: PipelineContext,
        upload_key: str,
        run_id: uuid.UUID,
        skipped: set[str],
        cancel_check: Callable[[uuid.UUID], Awaitable[bool]] | None,
        reporter: "ProgressReporter",
        queue: str | None,
        handoff: Callable[[uuid.UUID, str], Awaitable[None]] | None,
        shutdown_check: Callable[[], bool] | None,
    ) -> "_Outcome":
        from app.core.exceptions import JobCancelledError
        from app.core.metrics import AUDIO_MINUTES, STAGE_DURATION, STAGE_RTF
        from app.pipelines.checkpoints import ChunkHooks, StopReason, StopRequested

        async def should_stop() -> StopReason | None:
            if shutdown_check is not None and shutdown_check():
                return "shutdown"
            if cancel_check is not None and await cancel_check(context.meeting_id):
                return "cancelled"
            return None

        def hooks_for(stage: str) -> ChunkHooks:
            return ChunkHooks(
                self._store,
                context.meeting_id,
                run_id,
                stage,
                context.fingerprints.get(stage, "nofingerprint"),
                reporter,
                should_stop,
            )

        context.hooks_for = hooks_for
        remaining = [s for s in self.stages if s.name not in skipped]
        logger.info(
            "pipeline_started",
            stages=[s.name for s in remaining],
            skipped=sorted(skipped) or None,
            queue=queue,
        )
        errors: list[str] = []
        current = remaining[0].name if remaining else ""
        current_queue = remaining[0].queue if remaining else "cpu"
        audio_seconds = 0.0
        try:
            if not remaining:
                return _Outcome("finished", MeetingStatus.COMPLETED)
            async with self._inputs(context, upload_key, run_id, skipped):
                for stage in remaining:
                    if queue is not None and handoff is not None and stage.queue != queue:
                        await handoff(run_id, stage.queue)
                        return _Outcome("handoff", queue=stage.queue)
                    if reason := await should_stop():
                        raise StopRequested(reason)
                    current, current_queue = stage.name, stage.queue
                    await reporter.stage(stage.name, "running")
                    started = time.perf_counter()
                    try:
                        with span(f"stage.{stage.name}", meeting_id=context.meeting_id):
                            await asyncio.wait_for(
                                stage.run(context),
                                timeout=self._settings.stage_timeout(stage.name),
                            )
                    except StopRequested:
                        raise
                    except Exception as caught:
                        failure: Exception = caught
                        if isinstance(caught, TimeoutError):
                            failure = StageTimeoutError(
                                f"Stage '{stage.name}' exceeded its time limit.",
                                details={
                                    "timeout_seconds": self._settings.stage_timeout(stage.name)
                                },
                            )
                        message = _error_message(failure, stage.name)
                        await self._save_failed_stage(context, run_id, stage.name, message)
                        await reporter.stage(stage.name, "failed")
                        STAGE_DURATION.labels(stage.name, "failed").observe(
                            time.perf_counter() - started
                        )
                        if not stage.optional:
                            if isinstance(failure, PolymomError) and failure.retryable:
                                raise TransientStageError(stage.name, failure) from caught
                            raise failure from None
                        errors.append(f"{stage.name}: {message}")
                        logger.warning("optional_stage_failed", stage=stage.name, error=message)
                        continue
                    elapsed = time.perf_counter() - started
                    context.timings_ms[stage.name] = int(elapsed * 1000)
                    STAGE_DURATION.labels(stage.name, "completed").observe(elapsed)
                    preprocess = context.outputs.get(PreprocessStage.name)
                    if isinstance(preprocess, PreprocessResult):
                        audio_seconds = preprocess.duration_seconds
                        if audio_seconds > 0:
                            STAGE_RTF.labels(stage.name).observe(elapsed / audio_seconds)
                    async with self._uow() as uow:
                        writer = self._writer(uow, context.meeting_id, run_id)
                        writer.fingerprint = context.fingerprints.get(stage.name)
                        await stage.persist(context, writer)
                        await uow.commit()
                    await reporter.stage(stage.name, "completed")
                    logger.info(
                        "stage_completed",
                        stage=stage.name,
                        duration_ms=context.timings_ms[stage.name],
                    )
        except StopRequested as stop:
            if stop.reason == "shutdown":
                logger.info("pipeline_interrupted_for_shutdown", stage=current)
                if handoff is not None:
                    # Resume on another (or the restarted) worker of the same queue;
                    # chunks finished so far are read back from their checkpoints.
                    await handoff(run_id, current_queue)
                return _Outcome("interrupted", queue=current_queue)
            cancelled = JobCancelledError(
                f"Processing was cancelled during stage '{current}'.",
                details={"stage": current},
            )
            cancel_message = f"{cancelled.code}: {cancelled.message}"
            await self._save_failed_stage(context, run_id, current, cancel_message)
            return _Outcome("finished", MeetingStatus.CANCELLED, cancel_message)
        except TransientStageError:
            raise
        except FileNotFoundError:
            logger.warning("upload_missing", key=upload_key)
            return _Outcome(
                "finished",
                MeetingStatus.FAILED,
                "upload_missing: The uploaded file is no longer stored.",
            )
        except PolymomError as exc:
            logger.warning("stage_failed", stage=current, code=exc.code, error=exc.message)
            return _Outcome("finished", MeetingStatus.FAILED, f"{exc.code}: {exc.message}")
        except Exception:
            logger.exception("stage_crashed", stage=current)
            return _Outcome(
                "finished",
                MeetingStatus.FAILED,
                f"internal_error: Unexpected failure in stage '{current}'.",
            )
        if audio_seconds > 0:
            AUDIO_MINUTES.inc(audio_seconds / 60)
        status = MeetingStatus.COMPLETED_WITH_ERRORS if errors else MeetingStatus.COMPLETED
        return _Outcome("finished", status, "; ".join(errors) or None)

    @asynccontextmanager
    async def _inputs(
        self, context: PipelineContext, upload_key: str, run_id: uuid.UUID, skipped: set[str]
    ) -> AsyncIterator[None]:
        """Make the upload (and, after preprocessing, the processed WAV) available locally."""
        async with self._store.local_path(upload_key) as input_path:
            context.input_path = input_path
            if PreprocessStage.name in skipped and context.processed_path is None:
                key = run_key(context.meeting_id, run_id, "processed.wav")
                async with self._store.local_path(key) as processed:
                    context.processed_path = processed
                    yield
                return
            yield

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

    async def fail_run(self, meeting_id: uuid.UUID, run_id: uuid.UUID, error: str) -> None:
        """Mark a run and its meeting failed (the job queue ran out of retries)."""
        await self._finish(
            PipelineContext(meeting_id=meeting_id, input_path=Path()),
            run_id,
            MeetingStatus.FAILED,
            error,
        )

    def _cleanup_work_file(self, context: PipelineContext) -> None:
        """Remove the local processed WAV; the artifact store keeps the durable copy."""
        path = context.processed_path
        if path is not None and path.parent == self._settings.processed_dir:
            path.unlink(missing_ok=True)

    async def _finish(
        self,
        context: PipelineContext,
        run_id: uuid.UUID,
        status: MeetingStatus,
        error: str | None,
    ) -> None:
        versions = {
            stage.name: version
            for stage in self.stages
            if stage.name in context.outputs and (version := stage.model_version(context))
        }
        processed_key = run_key(context.meeting_id, run_id, "processed.wav")
        if not await self._store.exists(processed_key):
            processed_key = ""
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
            run.timings_ms = {**(run.timings_ms or {}), **context.timings_ms}
            run.model_versions = {**(run.model_versions or {}), **versions}
            await uow.commit()
        from app.core.metrics import JOBS

        JOBS.labels(status.value).inc()
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
