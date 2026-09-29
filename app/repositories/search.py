"""Full-text search over utterances and summaries of each meeting's latest results.

- SQLite: an FTS5 table with the ``trigram`` tokenizer. It needs no word segmentation,
  so Devanagari and Odia (whose vowel signs confuse word tokenizers) match as
  substrings. Queries shorter than 3 characters fall back to ``LIKE``.
- Postgres: a ``tsvector`` generated with the ``simple`` config (no stemming, no
  stop words, script-agnostic) plus a GIN index, OR-ed with ``ILIKE`` so partial
  words and Indic vowel-sign boundaries still match.

The index is replaced whenever a run's utterances or a summary are written.
"""

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.summary import MeetingSummary
from app.schemas.transcript import Utterance

TRIGRAM_MIN_CHARS = 3


@dataclass
class SearchRow:
    kind: Literal["utterance", "summary"]
    meeting_id: uuid.UUID
    run_id: uuid.UUID
    ref: str
    start: float | None
    speaker: str | None
    text: str


class SearchRepository(ABC):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _id(self, value: uuid.UUID) -> object:
        return value

    async def _insert(self, rows: Sequence[SearchRow]) -> None:
        if not rows:
            return
        await self._session.execute(
            text(
                "INSERT INTO search_index (text, kind, meeting_id, run_id, ref, start, speaker) "
                "VALUES (:text, :kind, :meeting_id, :run_id, :ref, :start, :speaker)"
            ),
            [
                {
                    "text": r.text,
                    "kind": r.kind,
                    "meeting_id": self._id(r.meeting_id),
                    "run_id": self._id(r.run_id),
                    "ref": r.ref,
                    "start": r.start,
                    "speaker": r.speaker,
                }
                for r in rows
            ],
        )

    async def _delete(self, meeting_id: uuid.UUID, kind: str | None = None) -> None:
        sql = "DELETE FROM search_index WHERE meeting_id = :m"
        params: dict[str, object] = {"m": self._id(meeting_id)}
        if kind:
            sql += " AND kind = :k"
            params["k"] = kind
        await self._session.execute(text(sql), params)

    async def index_utterances(
        self, meeting_id: uuid.UUID, run_id: uuid.UUID, utterances: Sequence[Utterance]
    ) -> None:
        await self._delete(meeting_id, "utterance")
        await self._insert(
            [
                SearchRow("utterance", meeting_id, run_id, str(u.id), u.start, u.speaker, u.text)
                for u in utterances
            ]
        )

    async def index_summary(
        self, meeting_id: uuid.UUID, run_id: uuid.UUID, summary: MeetingSummary
    ) -> None:
        await self._delete(meeting_id, "summary")
        parts = [summary.title, summary.executive_summary]
        parts += [k.title + ". " + k.description for k in summary.key_points]
        parts += [d.decision for d in summary.decisions]
        parts += [a.task for a in summary.action_items]
        await self._insert(
            [SearchRow("summary", meeting_id, run_id, "summary", None, None, "\n".join(parts))]
        )

    async def delete_meeting(self, meeting_id: uuid.UUID) -> None:
        await self._delete(meeting_id)

    @abstractmethod
    async def search(
        self, q: str, *, limit: int, meeting_id: uuid.UUID | None = None
    ) -> list[SearchRow]: ...

    @staticmethod
    def _rows(result: object) -> list[SearchRow]:
        return [
            SearchRow(
                kind=r.kind,
                meeting_id=uuid.UUID(str(r.meeting_id)),
                run_id=uuid.UUID(str(r.run_id)),
                ref=r.ref,
                start=r.start,
                speaker=r.speaker,
                text=r.text,
            )
            for r in result  # type: ignore[attr-defined]
        ]


class SqliteSearchRepository(SearchRepository):
    def _id(self, value: uuid.UUID) -> object:
        return str(value)

    async def search(
        self, q: str, *, limit: int, meeting_id: uuid.UUID | None = None
    ) -> list[SearchRow]:
        cols = "kind, meeting_id, run_id, ref, start, speaker, text"
        scope = " AND meeting_id = :m" if meeting_id else ""
        params: dict[str, object] = {"limit": limit, "m": str(meeting_id) if meeting_id else None}
        if len(q) >= TRIGRAM_MIN_CHARS:
            params["q"] = '"' + q.replace('"', '""') + '"'  # one phrase, no FTS operators
            sql = (
                f"SELECT {cols} FROM search_index WHERE search_index MATCH :q{scope} "  # noqa: S608
                "ORDER BY rank LIMIT :limit"
            )
        else:
            params["q"] = f"%{q}%"
            sql = f"SELECT {cols} FROM search_index WHERE text LIKE :q{scope} LIMIT :limit"  # noqa: S608
        return self._rows(await self._session.execute(text(sql), params))


class PostgresSearchRepository(SearchRepository):
    async def search(
        self, q: str, *, limit: int, meeting_id: uuid.UUID | None = None
    ) -> list[SearchRow]:  # pragma: no cover - runs in the Postgres (testcontainers) suite
        scope = " AND meeting_id = :m" if meeting_id else ""
        sql = (
            "SELECT kind, meeting_id, run_id, ref, start, speaker, text, "  # noqa: S608
            "ts_rank(tsv, plainto_tsquery('simple', :q)) AS score FROM search_index "
            "WHERE (tsv @@ plainto_tsquery('simple', :q) OR text ILIKE :like)"
            f"{scope} ORDER BY score DESC, start NULLS FIRST LIMIT :limit"
        )
        params = {"q": q, "like": f"%{q}%", "limit": limit, "m": meeting_id}
        return self._rows(await self._session.execute(text(sql), params))


def build_search_repository(session: AsyncSession) -> SearchRepository:
    if session.get_bind().dialect.name == "postgresql":
        return PostgresSearchRepository(session)
    return SqliteSearchRepository(session)
