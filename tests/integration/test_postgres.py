"""The same repositories and migrations on Postgres (testcontainers).

Marked ``postgres``; skipped when Docker is unavailable. JSONB columns, the
``simple`` tsvector search index and ON DELETE CASCADE are Postgres-specific paths.
"""

import uuid
from collections.abc import AsyncIterator, Iterator

import pytest
import sqlalchemy as sa

from app.db.migrate import downgrade_to, upgrade_to_head
from app.db.session import create_engine, create_sessionmaker
from app.models.meeting import Meeting
from app.repositories.meeting_repository import MeetingQuery
from app.repositories.results_repository import UtteranceQuery
from app.repositories.unit_of_work import UnitOfWorkFactory, unit_of_work_factory
from app.schemas.meeting import MeetingStatus
from app.schemas.transcript import Utterance

pytestmark = pytest.mark.postgres

TEXTS = [
    ("Person 1", "en", "The quarterly budget is approved."),
    ("Person 2", "hi", "हम बजट पर सहमत हैं।"),
    ("Person 1", "or", "ଆମେ ବଜେଟ୍ ପାଇଁ ରାଜି।"),
]


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    try:
        from testcontainers.postgres import PostgresContainer

        container = PostgresContainer("postgres:16-alpine", driver="asyncpg")
        container.start()
    except Exception as exc:  # Docker not running, image pull blocked, ...
        pytest.skip(f"Postgres container unavailable: {exc}")
    try:
        yield container.get_connection_url()
    finally:
        container.stop()


@pytest.fixture
async def pg_uow(postgres_url: str) -> AsyncIterator[UnitOfWorkFactory]:
    engine = create_engine(postgres_url)
    async with engine.begin() as conn:
        await conn.execute(sa.text("DROP SCHEMA public CASCADE"))
        await conn.execute(sa.text("CREATE SCHEMA public"))
    await upgrade_to_head(engine)
    yield unit_of_work_factory(create_sessionmaker(engine))
    await engine.dispose()


def utterance(i: int, speaker: str, lang: str, text: str) -> Utterance:
    return Utterance(
        id=i,
        speaker=speaker,
        start=i * 5.0,
        end=i * 5.0 + 4,
        duration=4,
        text=text,
        words=[],
        primary_language=lang,
        languages_present=[lang],
        is_code_mixed=False,
        avg_confidence=None,
        has_overlap=False,
        overlapping_speakers=[],
        alignment_precision="word",
    )


async def seed(uow_factory: UnitOfWorkFactory) -> tuple[uuid.UUID, uuid.UUID]:
    meeting_id = uuid.uuid4()
    utterances = [utterance(i, *t) for i, t in enumerate(TEXTS)]
    async with uow_factory() as uow:
        await uow.meetings.add(
            Meeting(
                id=meeting_id,
                original_filename="a.wav",
                upload_key=f"meetings/{meeting_id}/upload/original.wav",
                sha256="ab" * 32,
                mime_type="audio/x-wav",
                size_bytes=1,
                audio_metadata={"codec": "pcm"},
                languages_hint=["hi"],
                detected_languages=",en,hi,or,",
                num_speakers=2,
                status=MeetingStatus.COMPLETED,
            )
        )
        run = await uow.results.create_run(meeting_id, {"llm_provider": "mock"})
        run.status = "completed"
        await uow.results.save_stage(run.id, "align", status="completed", output={"ok": ["नमस्ते"]})
        await uow.results.replace_utterances(meeting_id, run.id, utterances)
        await uow.search.index_utterances(meeting_id, run.id, utterances)
        await uow.results.upsert_speakers(meeting_id, ["Person 1", "Person 2"])
        await uow.commit()
    return meeting_id, run.id


async def test_repositories_on_postgres(pg_uow: UnitOfWorkFactory) -> None:
    meeting_id, run_id = await seed(pg_uow)

    async with pg_uow() as uow:
        assert (await uow.meetings.get_by_sha256("ab" * 32)).id == meeting_id  # type: ignore[union-attr]
        page = await uow.meetings.page(MeetingQuery(language="or", min_speakers=2))
        assert [m.id for m in page.items] == [meeting_id]
        assert (await uow.results.latest_run(meeting_id)).id == run_id  # type: ignore[union-attr]
        row = await uow.results.stage_result(run_id, "align")
        assert row is not None
        assert row.output == {"ok": ["नमस्ते"]}
        rows, total = await uow.results.query_utterances(run_id, UtteranceQuery(language="hi"))
        assert (total, rows[0].speaker) == (1, "Person 2")
        for query in ("budget", "बजट", "ବଜେଟ୍"):
            hits = await uow.search.search(query, limit=5)
            assert [h.meeting_id for h in hits] == [meeting_id], query
        names = await uow.results.set_display_names(meeting_id, {"Person 1": "Ravi"})
        assert names == {"Person 1": "Ravi"}
        await uow.commit()

    async with pg_uow() as uow:
        meeting = await uow.meetings.get(meeting_id)
        await uow.meetings.delete(meeting)  # type: ignore[arg-type]
        await uow.commit()
        count = await uow.session.scalar(sa.text("SELECT count(*) FROM utterances"))
        index = await uow.session.scalar(sa.text("SELECT count(*) FROM search_index"))
        assert (count, index) == (0, 0)  # ON DELETE CASCADE


async def test_migrations_round_trip_on_postgres(
    postgres_url: str, pg_uow: UnitOfWorkFactory
) -> None:
    meeting_id, _ = await seed(pg_uow)
    engine = create_engine(postgres_url)
    try:
        await downgrade_to(engine, "0008")
        async with engine.connect() as conn:
            row = (
                await conn.execute(sa.text("SELECT stored_path, speaker_transcript FROM meetings"))
            ).one()
            assert row.stored_path == f"meetings/{meeting_id}/upload/original.wav"
            assert row.speaker_transcript == {"ok": ["नमस्ते"]}
        await upgrade_to_head(engine)
        async with engine.connect() as conn:
            assert (await conn.execute(sa.text("SELECT count(*) FROM utterances"))).scalar() == 0
            assert (await conn.execute(sa.text("SELECT count(*) FROM stage_results"))).scalar() == 1
    finally:
        await engine.dispose()
