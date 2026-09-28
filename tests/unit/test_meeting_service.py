import uuid
from pathlib import Path

import pytest

from app.core.config import Settings
from app.models.meeting import Meeting
from app.repositories.meeting_repository import MeetingRepository
from app.services.audio.validator import MediaValidator
from app.services.meeting_service import MeetingService
from tests.conftest import requires_ffmpeg
from tests.unit.test_validator import BytesSource


class FailingRepository(MeetingRepository):
    async def add(self, meeting: Meeting) -> Meeting:
        raise RuntimeError("database is down")

    async def get(self, meeting_id: uuid.UUID) -> Meeting | None:
        return None

    async def list(self, *, limit: int, offset: int) -> tuple[list[Meeting], int]:
        return [], 0

    async def delete(self, meeting: Meeting) -> None:
        return None


@requires_ffmpeg
async def test_stored_file_removed_when_persistence_fails(
    settings: Settings, wav_bytes: bytes
) -> None:
    service = MeetingService(FailingRepository(), MediaValidator(settings), settings)

    with pytest.raises(RuntimeError, match="database is down"):
        await service.create(
            source=BytesSource(wav_bytes),
            filename="a.wav",
            title=None,
            expected_speakers=None,
            languages=[],
        )

    uploads: Path = settings.uploads_dir
    assert not any(uploads.iterdir())
