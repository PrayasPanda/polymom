import uuid
from pathlib import Path

import pytest
from pydantic import BaseModel

from app.core.config import Settings
from app.core.exceptions import FFmpegTimeoutError
from app.models.meeting import Meeting
from app.pipelines.mom_pipeline import MoMPipeline, PipelineContext, PipelineStage
from app.repositories.artifacts import LocalArtifactStore
from app.repositories.unit_of_work import UnitOfWorkFactory
from app.schemas.meeting import MeetingStatus
from app.services.stage_outputs import load_stage_output


class Output(BaseModel):
    value: str


class RecordingStage(PipelineStage):
    def __init__(self, name: str, error: Exception | None = None) -> None:
        self.name = name  # type: ignore[misc]
        self.error = error
        self.ran = False

    async def run(self, context: PipelineContext) -> None:
        self.ran = True
        if self.error:
            raise self.error
        context.outputs[self.name] = Output(value=f"{self.name}-output")


class OptionalStage(RecordingStage):
    optional = True


@pytest.fixture
async def meeting(uow_factory: UnitOfWorkFactory, store: LocalArtifactStore) -> Meeting:
    return await create_meeting(uow_factory, store)


async def create_meeting(uow_factory: UnitOfWorkFactory, store: LocalArtifactStore) -> Meeting:
    meeting_id = uuid.uuid4()
    key = f"meetings/{meeting_id}/upload/original.wav"
    await store.put(key, b"RIFF")
    m = Meeting(
        id=meeting_id,
        title="t",
        original_filename="a.wav",
        upload_key=key,
        mime_type="audio/x-wav",
        size_bytes=4,
        audio_metadata={},
        languages_hint=[],
        status=MeetingStatus.QUEUED,
    )
    async with uow_factory() as uow:
        await uow.meetings.add(m)
        await uow.commit()
    return m


def pipeline(
    stages: list[PipelineStage],
    uow_factory: UnitOfWorkFactory,
    store: LocalArtifactStore,
    settings: Settings,
) -> MoMPipeline:
    return MoMPipeline(stages, uow_factory, store, settings)


async def stored(uow_factory: UnitOfWorkFactory, meeting_id: uuid.UUID) -> tuple[Meeting, object]:
    async with uow_factory() as uow:
        meeting = await uow.meetings.get(meeting_id)
        runs = await uow.results.list_runs(meeting_id)
        assert meeting is not None
        return meeting, runs[0] if runs else None


async def test_runs_stages_in_order_records_run_and_outputs(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    stages = [RecordingStage("one"), RecordingStage("two")]

    status = await pipeline(stages, uow_factory, store, settings).run(meeting.id)

    assert status == MeetingStatus.COMPLETED
    m, run = await stored(uow_factory, meeting.id)
    assert (m.status, m.error) == (MeetingStatus.COMPLETED, None)
    assert run.status == "completed"  # type: ignore[attr-defined]
    assert set(run.timings_ms) == {"one", "two"}  # type: ignore[attr-defined]
    assert run.config_snapshot["llm_provider"] == "mock"  # type: ignore[attr-defined]
    assert run.finished_at is not None  # type: ignore[attr-defined]
    async with uow_factory() as uow:
        output = await load_stage_output(uow.results, store, run.id, "two")  # type: ignore[attr-defined]
    assert output == {"value": "two-output"}


async def test_large_outputs_go_to_the_artifact_store(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    small = settings.model_copy(update={"stage_output_inline_max_bytes": 5})

    await pipeline([RecordingStage("big")], uow_factory, store, small).run(meeting.id)

    _, run = await stored(uow_factory, meeting.id)
    async with uow_factory() as uow:
        row = await uow.results.stage_result(run.id, "big")  # type: ignore[attr-defined]
        assert row is not None
        assert row.output is None
        assert row.output_ref == f"meetings/{meeting.id}/{run.id}/stages/big.json"  # type: ignore[attr-defined]
        assert await load_stage_output(uow.results, store, run.id, "big") == {  # type: ignore[attr-defined]
            "value": "big-output"
        }


async def test_typed_error_marks_failed_and_stops(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    failing = RecordingStage("preprocess", FFmpegTimeoutError("Audio processing timed out."))
    after = RecordingStage("later")

    status = await pipeline([failing, after], uow_factory, store, settings).run(meeting.id)

    assert status == MeetingStatus.FAILED
    m, run = await stored(uow_factory, meeting.id)
    assert m.error == "ffmpeg_timeout: Audio processing timed out."
    assert run.status == "failed"  # type: ignore[attr-defined]
    assert not after.ran
    async with uow_factory() as uow:
        row = await uow.results.stage_result(run.id, "preprocess")  # type: ignore[attr-defined]
    assert row is not None
    assert (row.status, row.error) == ("failed", "ffmpeg_timeout: Audio processing timed out.")


async def test_unexpected_error_marks_failed_with_internal_error(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    stage = RecordingStage("preprocess", RuntimeError("boom"))

    status = await pipeline([stage], uow_factory, store, settings).run(meeting.id)

    assert status == MeetingStatus.FAILED
    m, _ = await stored(uow_factory, meeting.id)
    assert m.error == "internal_error: Unexpected failure in stage 'preprocess'."


async def test_missing_upload_fails_cleanly(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    await store.delete(meeting.upload_key)
    stage = RecordingStage("one")

    status = await pipeline([stage], uow_factory, store, settings).run(meeting.id)

    assert status == MeetingStatus.FAILED
    assert not stage.ran
    m, _ = await stored(uow_factory, meeting.id)
    assert (m.error or "").startswith("upload_missing")


async def test_missing_meeting_is_a_no_op(
    uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    runner = pipeline([RecordingStage("x")], uow_factory, store, settings)
    assert await runner.run(uuid.uuid4()) is None


def test_pipeline_requires_stages(
    uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        pipeline([], uow_factory, store, settings)


async def test_optional_stage_failure_keeps_earlier_outputs(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    stages = [
        RecordingStage("one"),
        OptionalStage("summarize", RuntimeError("provider down")),
        RecordingStage("after"),
    ]

    status = await pipeline(stages, uow_factory, store, settings).run(meeting.id)

    assert status == MeetingStatus.COMPLETED_WITH_ERRORS
    m, run = await stored(uow_factory, meeting.id)
    assert m.error == "summarize: internal_error: Unexpected failure in stage 'summarize'."
    async with uow_factory() as uow:
        statuses = {
            r.stage_name: r.status
            for r in await uow.results.stage_results(run.id)  # type: ignore[attr-defined]
        }
    assert statuses == {"one": "completed", "summarize": "failed", "after": "completed"}


async def test_summarization_stage_requires_alignment() -> None:
    from app.pipelines.mom_pipeline import SummarizationStage

    stage = SummarizationStage(Settings(_env_file=None, llm_provider="mock"))
    with pytest.raises(RuntimeError, match="AlignmentStage"):
        await stage.run(PipelineContext(meeting_id=uuid.uuid4(), input_path=Path("x")))
