import uuid
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import pytest

from app.core.exceptions import FFmpegTimeoutError
from app.models.meeting import Meeting
from app.pipelines.mom_pipeline import (
    MoMPipeline,
    PipelineContext,
    PipelineStage,
    RepositoryFactory,
)
from app.repositories.meeting_repository import MeetingRepository
from app.schemas.meeting import MeetingStatus


class DictRepository(MeetingRepository):
    def __init__(self) -> None:
        self.items: dict[uuid.UUID, Meeting] = {}
        self.saved_statuses: list[MeetingStatus] = []

    async def add(self, meeting: Meeting) -> Meeting:
        self.items[meeting.id] = meeting
        return meeting

    async def get(self, meeting_id: uuid.UUID) -> Meeting | None:
        return self.items.get(meeting_id)

    async def list(self, *, limit: int, offset: int) -> tuple[list[Meeting], int]:
        return list(self.items.values()), len(self.items)

    async def save(self, meeting: Meeting) -> Meeting:
        self.saved_statuses.append(meeting.status)
        self.items[meeting.id] = meeting
        return meeting

    async def delete(self, meeting: Meeting) -> None:
        self.items.pop(meeting.id, None)


class RecordingStage(PipelineStage):
    def __init__(self, name: str, error: Exception | None = None) -> None:
        self.name = name  # type: ignore[misc]
        self.error = error
        self.ran = False

    async def run(self, context: PipelineContext) -> None:
        self.ran = True
        if self.error:
            raise self.error
        context.outputs[self.name] = f"{self.name}-output"

    def apply(self, context: PipelineContext, meeting: Meeting) -> None:
        meeting.title = f"{meeting.title}+{context.outputs[self.name]}"


@pytest.fixture
def repo() -> DictRepository:
    return DictRepository()


@pytest.fixture
def meeting(repo: DictRepository) -> Meeting:
    m = Meeting(
        id=uuid.uuid4(),
        title="t",
        original_filename="a.wav",
        stored_path="/data/a.wav",
        mime_type="audio/x-wav",
        size_bytes=1,
        audio_metadata={},
        languages_hint=[],
        status=MeetingStatus.QUEUED,
    )
    repo.items[m.id] = m
    return m


def _factory(repo: DictRepository) -> RepositoryFactory:
    @asynccontextmanager
    async def _cm() -> AsyncIterator[MeetingRepository]:
        yield repo

    def _make() -> AbstractAsyncContextManager[MeetingRepository]:
        return _cm()

    return _make


async def test_runs_stages_in_order_and_completes(repo: DictRepository, meeting: Meeting) -> None:
    stages = [RecordingStage("one"), RecordingStage("two")]

    status = await MoMPipeline(stages, _factory(repo)).run(meeting.id)

    assert status == MeetingStatus.COMPLETED
    assert repo.saved_statuses == [MeetingStatus.PROCESSING, MeetingStatus.COMPLETED]
    stored = repo.items[meeting.id]
    assert stored.title == "t+one-output+two-output"
    assert stored.error is None


async def test_typed_error_marks_failed_and_stops(repo: DictRepository, meeting: Meeting) -> None:
    failing = RecordingStage("preprocess", FFmpegTimeoutError("Audio processing timed out."))
    after = RecordingStage("later")

    status = await MoMPipeline([failing, after], _factory(repo)).run(meeting.id)

    assert status == MeetingStatus.FAILED
    assert repo.items[meeting.id].error == "ffmpeg_timeout: Audio processing timed out."
    assert not after.ran


async def test_unexpected_error_marks_failed_with_internal_error(
    repo: DictRepository, meeting: Meeting
) -> None:
    stage = RecordingStage("preprocess", RuntimeError("boom"))

    status = await MoMPipeline([stage], _factory(repo)).run(meeting.id)

    assert status == MeetingStatus.FAILED
    assert repo.items[meeting.id].error == (
        "internal_error: Unexpected failure in stage 'preprocess'."
    )


async def test_missing_meeting_is_a_no_op(repo: DictRepository) -> None:
    assert await MoMPipeline([RecordingStage("x")], _factory(repo)).run(uuid.uuid4()) is None


def test_pipeline_requires_stages(repo: DictRepository) -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        MoMPipeline([], _factory(repo))
