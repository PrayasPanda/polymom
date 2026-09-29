"""Retention: purge old raw audio, keep results; retry failed artifact deletions."""

import uuid
from datetime import timedelta

import pytest

from app.core.config import Settings
from app.db.base import utcnow
from app.models.meeting import Meeting
from app.repositories.artifacts import LocalArtifactStore, run_key
from app.repositories.unit_of_work import UnitOfWorkFactory
from app.schemas.meeting import MeetingStatus
from scripts import cleanup


async def add_meeting(
    uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, age_days: int
) -> tuple[Meeting, uuid.UUID, list[str]]:
    meeting_id = uuid.uuid4()
    upload = f"meetings/{meeting_id}/upload/original.wav"
    async with uow_factory() as uow:
        meeting = Meeting(
            id=meeting_id,
            original_filename="a.wav",
            upload_key=upload,
            mime_type="audio/x-wav",
            size_bytes=1,
            audio_metadata={},
            languages_hint=[],
            status=MeetingStatus.COMPLETED,
            created_at=utcnow() - timedelta(days=age_days),
        )
        await uow.meetings.add(meeting)
        run = await uow.results.create_run(meeting_id, {})
        await uow.results.save_stage(run.id, "align", status="completed", output={"kept": True})
        await uow.commit()
    processed = run_key(meeting_id, run.id, "processed.wav")
    export = run_key(meeting_id, run.id, "exports/minutes.pdf")
    for key in (upload, processed, export):
        await store.put(key, b"x")
    return meeting, run.id, [upload, processed, export]


async def test_purges_old_raw_audio_but_keeps_results(
    uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    old, old_run, (upload, processed, export) = await add_meeting(uow_factory, store, 120)
    _recent, _, recent_keys = await add_meeting(uow_factory, store, 5)
    policy = settings.model_copy(update={"retention_days": 90, "keep_raw_audio": False})

    dry = await cleanup.purge_raw_audio(uow_factory, store, policy, apply=False)
    assert dry.purged_meetings == [str(old.id)]
    assert await store.exists(upload)

    report = await cleanup.purge_raw_audio(uow_factory, store, policy, apply=True)

    assert report.purged_meetings == [str(old.id)]
    assert sorted(report.deleted_keys) == sorted([upload, processed])
    assert not await store.exists(upload)
    assert not await store.exists(processed)
    assert await store.exists(export)  # results are kept
    assert all([await store.exists(k) for k in recent_keys])
    async with uow_factory() as uow:
        meeting = await uow.meetings.get(old.id)
        assert meeting is not None
        assert meeting.raw_audio_purged_at is not None
        assert await uow.results.stage_result(old_run, "align") is not None
    again = await cleanup.purge_raw_audio(uow_factory, store, policy, apply=True)
    assert again.purged_meetings == []


@pytest.mark.parametrize("update", [{"keep_raw_audio": True}, {"retention_days": 0}])
async def test_retention_disabled(
    uow_factory: UnitOfWorkFactory,
    store: LocalArtifactStore,
    settings: Settings,
    update: dict[str, object],
) -> None:
    await add_meeting(uow_factory, store, 400)
    report = await cleanup.purge_raw_audio(
        uow_factory, store, settings.model_copy(update=update), apply=True
    )
    assert report.purged_meetings == []


class FlakyStore(LocalArtifactStore):
    fail = True

    async def delete(self, key: str) -> None:
        if self.fail:
            raise OSError("bucket unavailable")
        await super().delete(key)


async def test_failed_deletions_are_logged_and_retried(
    uow_factory: UnitOfWorkFactory, settings: Settings
) -> None:
    from app.services.audio.validator import MediaValidator
    from app.services.meeting_service import MeetingService

    store = FlakyStore(settings.storage_dir)
    meeting, _, keys = await add_meeting(uow_factory, store, 1)
    async with uow_factory() as uow:
        await MeetingService(uow, MediaValidator(settings), settings, store).delete(meeting.id)
    async with uow_factory() as uow:
        assert await uow.meetings.get(meeting.id) is None  # rows go even if files can't
        pending = await uow.results.pending_cleanups()
        assert [p.key for p in pending] == [keys[0]]

    store.fail = False
    report = cleanup.CleanupReport()
    await cleanup.retry_failed_deletions(uow_factory, store, report, apply=True)
    assert (report.retried, report.still_failing) == (1, 0)
    async with uow_factory() as uow:
        assert await uow.results.pending_cleanups() == []


async def test_cli_dry_run(
    settings: Settings, uow_factory: UnitOfWorkFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STORAGE_DIR", str(settings.storage_dir))
    monkeypatch.setenv("RETENTION_DAYS", "30")
    report = await cleanup.run(Settings(_env_file=None), apply=False)
    assert report.purged_meetings == []
