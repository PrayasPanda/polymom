"""Retention and artifact cleanup.

    uv run python scripts/cleanup.py            # dry run: report what would be purged
    uv run python scripts/cleanup.py --apply    # purge

1. Purges raw audio (the upload and every run's ``processed.wav``) of meetings older
   than ``RETENTION_DAYS``, unless ``KEEP_RAW_AUDIO=true`` or ``RETENTION_DAYS=0``.
   Results (transcripts, analytics, summaries, exports) are kept; the meeting gets
   ``raw_audio_purged_at`` and can no longer be reprocessed.
2. Retries artifact deletions that failed after a meeting was deleted
   (``artifact_cleanup_log``).
"""

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from datetime import timedelta

from app.core.config import Settings
from app.db.base import utcnow
from app.db.session import create_engine, create_sessionmaker
from app.repositories.artifacts import ArtifactStore, build_artifact_store, run_key
from app.repositories.unit_of_work import UnitOfWorkFactory, unit_of_work_factory


@dataclass
class CleanupReport:
    purged_meetings: list[str] = field(default_factory=list)
    deleted_keys: list[str] = field(default_factory=list)
    retried: int = 0
    still_failing: int = 0


async def purge_raw_audio(
    uow_factory: UnitOfWorkFactory, store: ArtifactStore, settings: Settings, *, apply: bool
) -> CleanupReport:
    report = CleanupReport()
    if settings.keep_raw_audio or settings.retention_days == 0:
        return report
    cutoff = utcnow() - timedelta(days=settings.retention_days)
    async with uow_factory() as uow:
        for meeting in await uow.meetings.older_than(cutoff):
            keys = [meeting.upload_key, meeting.processed_key]
            keys += [
                run_key(meeting.id, run.id, "processed.wav")
                for run in await uow.results.list_runs(meeting.id)
            ]
            keys = sorted({k for k in keys if k})
            report.purged_meetings.append(str(meeting.id))
            report.deleted_keys += keys
            if apply:
                for key in keys:
                    await store.delete(key)
                meeting.raw_audio_purged_at = utcnow()
                meeting.processed_key = None
        if apply:
            await uow.commit()
    return report


async def retry_failed_deletions(
    uow_factory: UnitOfWorkFactory, store: ArtifactStore, report: CleanupReport, *, apply: bool
) -> None:
    async with uow_factory() as uow:
        for entry in await uow.results.pending_cleanups():
            report.retried += 1
            if not apply:
                continue
            try:
                if entry.key.endswith("/"):
                    await store.delete_prefix(entry.key)
                else:
                    await store.delete(entry.key)
            except Exception as exc:
                entry.attempts += 1
                entry.error = str(exc)[:2000]
                report.still_failing += 1
            else:
                await uow.results.resolve_cleanup(entry)
        if apply:
            await uow.commit()


async def run(settings: Settings, *, apply: bool) -> CleanupReport:
    engine = create_engine(settings.resolved_database_url)
    try:
        uow_factory = unit_of_work_factory(create_sessionmaker(engine))
        store = build_artifact_store(settings)
        report = await purge_raw_audio(uow_factory, store, settings, apply=apply)
        await retry_failed_deletions(uow_factory, store, report, apply=apply)
        return report
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> CleanupReport:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="delete (default: dry run)")
    args = parser.parse_args(argv)
    settings = Settings()
    report = asyncio.run(run(settings, apply=args.apply))
    mode = "purged" if args.apply else "would purge"
    print(
        f"{mode} raw audio of {len(report.purged_meetings)} meeting(s) older than "
        f"{settings.retention_days} days ({len(report.deleted_keys)} artifacts); "
        f"failed deletions retried: {report.retried}, still failing: {report.still_failing}"
    )
    return report


if __name__ == "__main__":
    main(sys.argv[1:])
