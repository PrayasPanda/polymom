import pytest

from app.core.config import Settings
from app.repositories.artifacts import LocalArtifactStore
from app.repositories.unit_of_work import UnitOfWork, UnitOfWorkFactory
from app.services.audio.validator import MediaValidator
from app.services.meeting_service import MeetingService
from tests.conftest import requires_ffmpeg
from tests.unit.test_validator import BytesSource


class FailingUnitOfWork(UnitOfWork):
    async def commit(self) -> None:
        raise RuntimeError("database is down")


@requires_ffmpeg
async def test_stored_file_removed_when_persistence_fails(
    settings: Settings, wav_bytes: bytes, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore
) -> None:
    async with uow_factory() as uow:
        service = MeetingService(
            FailingUnitOfWork(uow.session), MediaValidator(settings), settings, store
        )
        with pytest.raises(RuntimeError, match="database is down"):
            await service.create(
                source=BytesSource(wav_bytes),
                filename="a.wav",
                title=None,
                expected_speakers=None,
                languages=[],
            )

    leftovers = [p for p in settings.storage_dir.rglob("*") if p.is_file() and p.suffix != ".db"]
    assert leftovers == []
