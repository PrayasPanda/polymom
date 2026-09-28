"""Minutes-of-Meeting pipeline: an ordered list of stages sharing a context.

Stages (planned): preprocess -> diarize -> transcribe -> align -> analytics -> summarize.
Registered so far: preprocess -> diarize -> transcribe.
"""

import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import structlog

from app.core.config import Settings
from app.core.exceptions import PolymomError
from app.core.logging import get_logger
from app.models.meeting import Meeting
from app.repositories.meeting_repository import MeetingRepository
from app.schemas.asr import ASRResult
from app.schemas.audio import PreprocessResult
from app.schemas.diarization import DiarizationResult
from app.schemas.meeting import MeetingStatus
from app.services.asr.router import ASRRouter
from app.services.asr.service import TranscriptionService
from app.services.audio.preprocessor import AudioPreprocessor
from app.services.diarization.base import DiarizationBackend
from app.services.diarization.service import DiarizationService

logger = get_logger(__name__)

RepositoryFactory = Callable[[], AbstractAsyncContextManager[MeetingRepository]]


@dataclass
class PipelineContext:
    """State passed from stage to stage for one meeting."""

    meeting_id: uuid.UUID
    input_path: Path
    expected_speakers: int | None = None
    languages_hint: list[str] = field(default_factory=list)
    processed_path: Path | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, int] = field(default_factory=dict)


class PipelineStage(ABC):
    """One step of the pipeline.

    ``run`` does the work and records results on the context; ``apply`` copies
    whatever should be persisted onto the meeting entity.
    """

    name: ClassVar[str]

    @abstractmethod
    async def run(self, context: PipelineContext) -> None: ...

    def apply(self, context: PipelineContext, meeting: Meeting) -> None:  # noqa: B027
        """Persist stage outputs on the meeting. Default: nothing to persist."""


class PreprocessStage(PipelineStage):
    name = "preprocess"

    def __init__(self, preprocessor: AudioPreprocessor) -> None:
        self._preprocessor = preprocessor

    async def run(self, context: PipelineContext) -> None:
        result = await self._preprocessor.process(context.meeting_id, context.input_path)
        context.processed_path = result.processed_path
        context.outputs[self.name] = result

    def apply(self, context: PipelineContext, meeting: Meeting) -> None:
        result: PreprocessResult = context.outputs[self.name]
        meeting.processed_path = str(result.processed_path.resolve())
        meeting.audio_quality = result.model_dump(mode="json", exclude={"processed_path"})


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

    def apply(self, context: PipelineContext, meeting: Meeting) -> None:
        result: DiarizationResult = context.outputs[self.name]
        meeting.diarization = result.model_dump(mode="json")


class TranscriptionStage(PipelineStage):
    """Timestamped transcript of the preprocessed audio.

    Independent of diarization for now; word-to-speaker alignment comes later.
    """

    name = "transcribe"

    def __init__(self, service: TranscriptionService) -> None:
        self._service = service

    async def run(self, context: PipelineContext) -> None:
        if context.processed_path is None:
            raise RuntimeError("TranscriptionStage requires PreprocessStage to run first")
        context.outputs[self.name] = await self._service.transcribe(
            context.meeting_id, context.processed_path, context.languages_hint
        )

    def apply(self, context: PipelineContext, meeting: Meeting) -> None:
        result: ASRResult = context.outputs[self.name]
        meeting.transcript = result.model_dump(mode="json")


class MoMPipeline:
    """Runs stages in order and owns the meeting status lifecycle.

    ``processing`` -> ``completed`` on success, or ``failed`` with
    ``"<error code>: <message>"`` when a stage raises.
    """

    def __init__(self, stages: Sequence[PipelineStage], repositories: RepositoryFactory) -> None:
        if not stages:
            raise ValueError("a pipeline needs at least one stage")
        self.stages = list(stages)
        self._repositories = repositories

    async def run(self, meeting_id: uuid.UUID) -> MeetingStatus | None:
        """Process one meeting. Never raises for stage failures; returns the final status."""
        with structlog.contextvars.bound_contextvars(meeting_id=str(meeting_id)):
            async with self._repositories() as repo:
                meeting = await repo.get(meeting_id)
                if meeting is None:
                    logger.warning("pipeline_meeting_missing")
                    return None
                meeting.status = MeetingStatus.PROCESSING
                meeting.error = None
                meeting = await repo.save(meeting)

            context = PipelineContext(
                meeting_id=meeting_id,
                input_path=Path(meeting.stored_path),
                expected_speakers=meeting.expected_speakers,
                languages_hint=list(meeting.languages_hint or []),
            )
            logger.info("pipeline_started", stages=[s.name for s in self.stages])
            started = time.perf_counter()
            current = self.stages[0].name
            try:
                for stage in self.stages:
                    current = stage.name
                    stage_started = time.perf_counter()
                    await stage.run(context)
                    elapsed = int((time.perf_counter() - stage_started) * 1000)
                    context.timings_ms[stage.name] = elapsed
                    logger.info("stage_completed", stage=stage.name, duration_ms=elapsed)
                    stage.apply(context, meeting)
                meeting.status = MeetingStatus.COMPLETED
            except PolymomError as exc:
                meeting.status = MeetingStatus.FAILED
                meeting.error = f"{exc.code}: {exc.message}"
                logger.warning("stage_failed", stage=current, code=exc.code, error=exc.message)
            except Exception:
                meeting.status = MeetingStatus.FAILED
                meeting.error = f"internal_error: Unexpected failure in stage '{current}'."
                logger.exception("stage_crashed", stage=current)

            async with self._repositories() as repo:
                await repo.save(meeting)
            logger.info(
                "pipeline_finished",
                status=meeting.status.value,
                duration_ms=int((time.perf_counter() - started) * 1000),
                timings_ms=context.timings_ms,
            )
            return meeting.status


def build_pipeline(
    settings: Settings,
    repositories: RepositoryFactory,
    diarization_backend: DiarizationBackend,
    asr_router: ASRRouter,
) -> MoMPipeline:
    """Default stage registry. Later prompts append alignment, analytics, ..."""
    return MoMPipeline(
        [
            PreprocessStage(AudioPreprocessor(settings)),
            DiarizationStage(DiarizationService(diarization_backend, settings)),
            TranscriptionStage(TranscriptionService(asr_router, settings)),
        ],
        repositories,
    )
