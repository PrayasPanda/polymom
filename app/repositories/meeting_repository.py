"""Meeting repository interface and its SQLAlchemy implementation."""

import uuid
from abc import ABC, abstractmethod

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.meeting import Meeting


class MeetingRepository(ABC):
    """Persistence port for meetings. Services depend on this, not on a backend."""

    @abstractmethod
    async def add(self, meeting: Meeting) -> Meeting: ...

    @abstractmethod
    async def get(self, meeting_id: uuid.UUID) -> Meeting | None: ...

    @abstractmethod
    async def list(self, *, limit: int, offset: int) -> tuple[list[Meeting], int]:
        """Return one page of meetings, newest first, and the total count."""

    @abstractmethod
    async def save(self, meeting: Meeting) -> Meeting:
        """Persist changes to an existing meeting."""

    @abstractmethod
    async def delete(self, meeting: Meeting) -> None: ...


class SqlAlchemyMeetingRepository(MeetingRepository):
    """Works with any SQLAlchemy async backend (SQLite, Postgres, ...)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, meeting: Meeting) -> Meeting:
        self._session.add(meeting)
        await self._session.commit()
        return meeting

    async def get(self, meeting_id: uuid.UUID) -> Meeting | None:
        return await self._session.get(Meeting, meeting_id)

    async def list(self, *, limit: int, offset: int) -> tuple[list[Meeting], int]:
        total = await self._session.scalar(select(func.count()).select_from(Meeting))
        rows = await self._session.scalars(
            select(Meeting)
            .order_by(Meeting.created_at.desc(), Meeting.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(rows), total or 0

    async def save(self, meeting: Meeting) -> Meeting:
        merged = await self._session.merge(meeting)
        await self._session.commit()
        return merged

    async def delete(self, meeting: Meeting) -> None:
        await self._session.delete(meeting)
        await self._session.commit()
