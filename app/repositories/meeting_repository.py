"""Meeting repository: CRUD, duplicate lookup, filtered keyset pagination.

Repositories never commit; the :class:`~app.repositories.unit_of_work.UnitOfWork` does.
"""

import base64
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import ColumnElement, and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ValidationError
from app.models.meeting import Meeting
from app.schemas.meeting import MeetingStatus

SortField = Literal["created_at", "duration_seconds"]


@dataclass
class MeetingQuery:
    status: MeetingStatus | None = None
    language: str | None = None
    created_from: datetime | None = None
    created_to: datetime | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    sort: SortField = "created_at"
    order: Literal["asc", "desc"] = "desc"
    cursor: str | None = None
    limit: int = 20


@dataclass
class MeetingPage:
    items: list[Meeting]
    total: int
    next_cursor: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def encode_cursor(value: Any, meeting_id: uuid.UUID) -> str:
    raw = json.dumps([value, str(meeting_id)], default=str).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[Any, uuid.UUID]:
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        value, meeting_id = json.loads(raw)
        return value, uuid.UUID(meeting_id)
    except (ValueError, TypeError) as exc:
        raise ValidationError("Invalid cursor.", details={"cursor": cursor}) from exc


class MeetingRepository(ABC):
    @abstractmethod
    async def add(self, meeting: Meeting) -> Meeting: ...

    @abstractmethod
    async def get(self, meeting_id: uuid.UUID) -> Meeting | None: ...

    @abstractmethod
    async def get_by_sha256(self, sha256: str) -> Meeting | None:
        """Oldest meeting with this upload hash."""

    @abstractmethod
    async def page(self, query: MeetingQuery) -> MeetingPage: ...

    @abstractmethod
    async def older_than(self, cutoff: datetime) -> list[Meeting]:
        """Meetings created before ``cutoff`` whose raw audio has not been purged."""

    @abstractmethod
    async def delete(self, meeting: Meeting) -> None: ...


class SqlAlchemyMeetingRepository(MeetingRepository):
    """Works with any SQLAlchemy async backend (SQLite, Postgres, ...)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, meeting: Meeting) -> Meeting:
        self._session.add(meeting)
        await self._session.flush()
        return meeting

    async def get(self, meeting_id: uuid.UUID) -> Meeting | None:
        return await self._session.get(Meeting, meeting_id)

    async def get_by_sha256(self, sha256: str) -> Meeting | None:
        return await self._session.scalar(
            select(Meeting).where(Meeting.sha256 == sha256).order_by(Meeting.created_at).limit(1)
        )

    async def page(self, query: MeetingQuery) -> MeetingPage:
        filters: list[ColumnElement[bool]] = []
        if query.status:
            filters.append(Meeting.status == query.status)
        if query.language:
            filters.append(Meeting.detected_languages.like(f"%,{query.language},%"))
        if query.created_from:
            filters.append(Meeting.created_at >= query.created_from)
        if query.created_to:
            filters.append(Meeting.created_at <= query.created_to)
        if query.min_speakers is not None:
            filters.append(Meeting.num_speakers >= query.min_speakers)
        if query.max_speakers is not None:
            filters.append(Meeting.num_speakers <= query.max_speakers)
        total = await self._session.scalar(
            select(func.count()).select_from(Meeting).where(*filters)
        )

        sort_col: Any = (
            Meeting.created_at
            if query.sort == "created_at"
            else func.coalesce(Meeting.duration_seconds, -1.0)
        )
        desc = query.order == "desc"
        page_filters = list(filters)
        if query.cursor:
            value, last_id = decode_cursor(query.cursor)
            if query.sort == "created_at":
                value = datetime.fromisoformat(value)
            beyond = sort_col < value if desc else sort_col > value
            tie = Meeting.id < last_id if desc else Meeting.id > last_id
            page_filters.append(or_(beyond, and_(sort_col == value, tie)))
        order = (sort_col.desc(), Meeting.id.desc()) if desc else (sort_col.asc(), Meeting.id.asc())
        rows = list(
            await self._session.scalars(
                select(Meeting).where(*page_filters).order_by(*order).limit(query.limit + 1)
            )
        )
        next_cursor = None
        if len(rows) > query.limit:
            rows = rows[: query.limit]
            last = rows[-1]
            value = (
                last.created_at.isoformat()
                if query.sort == "created_at"
                else (last.duration_seconds if last.duration_seconds is not None else -1.0)
            )
            next_cursor = encode_cursor(value, last.id)
        return MeetingPage(items=rows, total=total or 0, next_cursor=next_cursor)

    async def older_than(self, cutoff: datetime) -> list[Meeting]:
        return list(
            await self._session.scalars(
                select(Meeting).where(
                    Meeting.created_at < cutoff, Meeting.raw_audio_purged_at.is_(None)
                )
            )
        )

    async def delete(self, meeting: Meeting) -> None:
        await self._session.delete(meeting)
        await self._session.flush()
