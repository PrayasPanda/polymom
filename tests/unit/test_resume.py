"""Resumable runs: skip completed stages and chunks, hand-off, cancel, retry, timeout."""

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.core.exceptions import LLMError, StageTimeoutError, ValidationError
from app.models.meeting import Meeting
from app.pipelines.checkpoints import ChunkHooks, StopRequested
from app.pipelines.mom_pipeline import MoMPipeline, PipelineContext, TransientStageError
from app.repositories.artifacts import LocalArtifactStore
from app.repositories.unit_of_work import UnitOfWorkFactory
from app.schemas.meeting import MeetingStatus
from app.services.diarization.mock_backend import MockDiarizationBackend
from app.services.diarization.service import DiarizationService
from tests.conftest import make_wav
from tests.unit.test_pipeline import Output, RecordingStage, create_meeting


class CountingStage(RecordingStage):
    """Counts runs; optionally fails the first ``fail_times`` runs with ``error``."""

    def __init__(
        self,
        name: str,
        *,
        queue: str = "cpu",
        error: Exception | None = None,
        fail_times: int = 0,
        config_keys: tuple[str, ...] = (),
        sleep: float = 0.0,
    ) -> None:
        super().__init__(name)
        self.queue = queue  # type: ignore[misc]
        self.config_keys = config_keys  # type: ignore[misc]
        self.failure = error
        self.fail_times = fail_times
        self.sleep = sleep
        self.calls = 0

    async def run(self, context: PipelineContext) -> None:
        self.calls += 1
        if self.sleep:
            await asyncio.sleep(self.sleep)
        if self.failure is not None and self.calls <= self.fail_times:
            raise self.failure
        context.outputs[self.name] = Output(value=f"{self.name}-{self.calls}")


@pytest.fixture
async def meeting(uow_factory: UnitOfWorkFactory, store: LocalArtifactStore) -> Meeting:
    return await create_meeting(uow_factory, store)


def build(
    stages: list[Any], uow: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> MoMPipeline:
    return MoMPipeline(stages, uow, store, settings)


async def statuses(uow_factory: UnitOfWorkFactory, meeting_id: uuid.UUID) -> dict[str, str]:
    async with uow_factory() as uow:
        run = (await uow.results.list_runs(meeting_id))[0]
        return {r.stage_name: r.status for r in await uow.results.stage_results(run.id)}


async def test_interrupted_run_resumes_and_skips_completed_stages(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    one, two = CountingStage("one"), CountingStage("two")
    pipeline = build([one, two], uow_factory, store, settings)
    stop_after_first = iter([False, True])  # checked before each stage

    first = await pipeline.run(meeting.id, shutdown_check=lambda: next(stop_after_first))

    assert first == MeetingStatus.PROCESSING  # "worker killed" after stage one
    async with uow_factory() as uow:
        run_id = (await uow.results.list_runs(meeting.id))[0].id

    resumed = await pipeline.run(meeting.id, run_id=run_id)

    assert resumed == MeetingStatus.COMPLETED
    assert (one.calls, two.calls) == (1, 1)  # stage one was not recomputed
    async with uow_factory() as uow:
        runs = await uow.results.list_runs(meeting.id)
    assert [r.id for r in runs] == [run_id]  # same run, not a new one
    assert await statuses(uow_factory, meeting.id) == {"one": "completed", "two": "completed"}


async def test_changed_config_invalidates_the_fingerprint(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    one = CountingStage("one", config_keys=("align_max_gap_seconds",))
    two = CountingStage("two")
    stops = iter([False, True])
    await build([one, two], uow_factory, store, settings).run(
        meeting.id, shutdown_check=lambda: next(stops)
    )
    async with uow_factory() as uow:
        run_id = (await uow.results.list_runs(meeting.id))[0].id

    changed = settings.model_copy(update={"align_max_gap_seconds": 9.0})
    await build([one, two], uow_factory, store, changed).run(meeting.id, run_id=run_id)

    assert one.calls == 2  # config changed since the crash: recomputed


async def test_handoff_between_queues(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    stages = [
        CountingStage("pre"),
        CountingStage("asr", queue="gpu"),
        CountingStage("post"),
        CountingStage("llm", queue="llm"),
    ]
    pipeline = build(stages, uow_factory, store, settings)
    handoffs: list[tuple[uuid.UUID, str]] = []

    async def handoff(run_id: uuid.UUID, queue: str) -> None:
        handoffs.append((run_id, queue))

    assert await pipeline.run(meeting.id, queue="cpu", handoff=handoff) == MeetingStatus.PROCESSING
    run_id = handoffs[0][0]
    for queue in ("gpu", "cpu"):
        status = await pipeline.run(meeting.id, run_id=run_id, queue=queue, handoff=handoff)
        assert status == MeetingStatus.PROCESSING
    final = await pipeline.run(meeting.id, run_id=run_id, queue="llm", handoff=handoff)

    assert final == MeetingStatus.COMPLETED
    assert [q for _, q in handoffs] == ["gpu", "cpu", "llm"]
    assert [s.calls for s in stages] == [1, 1, 1, 1]  # every stage ran exactly once


async def test_cancel_marks_meeting_cancelled(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    one, two = CountingStage("one"), CountingStage("two")
    answers = iter([False, True])

    async def cancel_check(_: uuid.UUID) -> bool:
        return next(answers)

    status = await build([one, two], uow_factory, store, settings).run(
        meeting.id, cancel_check=cancel_check
    )

    assert status == MeetingStatus.CANCELLED
    assert two.calls == 0
    async with uow_factory() as uow:
        stored = await uow.meetings.get(meeting.id)
    assert stored is not None
    assert (stored.error or "").startswith("cancelled:")


async def test_transient_errors_retry_inline_then_succeed(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    one = CountingStage("one")
    flaky = CountingStage("flaky", error=LLMError("rate limited"), fail_times=2)

    status = await build([one, flaky], uow_factory, store, settings).run(meeting.id)

    assert status == MeetingStatus.COMPLETED
    assert (one.calls, flaky.calls) == (1, 3)  # retries resume at the failed stage


async def test_non_retryable_errors_fail_immediately(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    bad = CountingStage("bad", error=ValidationError("corrupt"), fail_times=99)

    status = await build([bad], uow_factory, store, settings).run(meeting.id)

    assert status == MeetingStatus.FAILED
    assert bad.calls == 1


async def test_retries_are_bounded(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    always = CountingStage("always", error=LLMError("down"), fail_times=99)
    limited = settings.model_copy(update={"max_retries": 2})

    status = await build([always], uow_factory, store, limited).run(meeting.id)

    assert status == MeetingStatus.FAILED
    assert always.calls == 3  # first try + 2 retries


async def test_queue_mode_raises_for_the_job_queue_to_retry(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    flaky = CountingStage("flaky", error=LLMError("rate limited"), fail_times=1)

    with pytest.raises(TransientStageError) as info:
        await build([flaky], uow_factory, store, settings).run(meeting.id, queue="cpu")

    assert info.value.stage == "flaky"
    assert info.value.run_id is not None


async def test_stage_timeout(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    slow = CountingStage("slow", sleep=0.5)
    tight = settings.model_copy(update={"stage_timeouts": {"slow": 0.05}, "max_retries": 0})

    status = await build([slow], uow_factory, store, tight).run(meeting.id)

    assert status == MeetingStatus.FAILED
    async with uow_factory() as uow:
        stored = await uow.meetings.get(meeting.id)
    assert stored is not None
    assert (stored.error or "").startswith(StageTimeoutError.code)


# --- chunk checkpoints ---


class CountingDiarization(MockDiarizationBackend):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.chunks = 0

    async def diarize_raw(self, *args: Any, **kwargs: Any) -> Any:
        self.chunks += 1
        return await super().diarize_raw(*args, **kwargs)


async def test_chunk_checkpoints_resume_mid_stage(
    tmp_path: Path, store: LocalArtifactStore, settings: Settings
) -> None:
    chunked = settings.model_copy(
        update={
            "diarization_chunk_threshold_seconds": 5.0,
            "chunk_length_seconds": 4.0,
            "chunk_overlap_seconds": 0.0,
        }
    )
    audio = tmp_path / "long.wav"
    audio.write_bytes(make_wav(seconds=16))  # 4 chunks
    backend = CountingDiarization(chunked)
    service = DiarizationService(backend, chunked)
    meeting_id, run_id = uuid.uuid4(), uuid.uuid4()
    stop_after = {"n": 2}

    async def should_stop() -> Any:
        stop_after["n"] -= 1
        return "shutdown" if stop_after["n"] == 0 else None

    hooks = ChunkHooks(store, meeting_id, run_id, "diarize", "f" * 64, None, should_stop)
    with pytest.raises(StopRequested):
        await service.diarize(meeting_id, audio, hooks=hooks)
    assert backend.chunks == 2  # crashed after chunk 2 of 4

    resumed = ChunkHooks(store, meeting_id, run_id, "diarize", "f" * 64)
    result = await service.diarize(meeting_id, audio, hooks=resumed)

    assert backend.chunks == 4  # only chunks 3 and 4 were computed on resume
    reference = await DiarizationService(CountingDiarization(chunked), chunked).diarize(
        meeting_id, audio
    )
    assert [(t.speaker_label, t.start) for t in result.turns] == [
        (t.speaker_label, t.start) for t in reference.turns
    ]
    other = ChunkHooks(store, meeting_id, run_id, "diarize", "0" * 64)
    assert await other.load(0) is None  # a different fingerprint never reuses checkpoints
