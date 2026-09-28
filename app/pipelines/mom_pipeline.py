"""Minutes-of-Meeting pipeline: an ordered list of stages sharing a context.

Stages (planned): preprocess -> diarize -> transcribe -> align -> analytics -> summarize.
Only ``preprocess`` is registered so far.
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
from app.schemas.audio import PreprocessResult
from app.schemas.meeting import MeetingStatus
from app.services.audio.preprocessor import AudioPreprocessor

logger = get_logger(__name__)

RepositoryFactory = Callable[[], AbstractAsyncContextManager[MeetingRepository]]


@dataclass
class PipelineContext:
    """State passed from stage to stage for one meeting."""

    meeting_id: uuid.UUID
    input_path: Path
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

            context = PipelineContext(meeting_id=meeting_id, input_path=Path(meeting.stored_path))
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


def build_pipeline(settings: Settings, repositories: RepositoryFactory) -> MoMPipeline:
    """Default stage registry. Later prompts append diarization, ASR, ..."""
    return MoMPipeline([PreprocessStage(AudioPreprocessor(settings))], repositories)
