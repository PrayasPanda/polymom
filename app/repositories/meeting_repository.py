"""Meeting repository interface and an in-memory stub implementation."""

from typing import Protocol

from app.models.meeting import Meeting


class MeetingRepository(Protocol):
    """Persistence port for meetings. Services depend on this, not on a backend."""

    async def add(self, meeting: Meeting) -> None: ...

    async def get(self, meeting_id: str) -> Meeting | None: ...

    async def list_all(self) -> list[Meeting]: ...


class InMemoryMeetingRepository:
    """Process-local repository for development and tests.

    TODO(prompt-2+): add a durable implementation (DB + file storage).
    """

    def __init__(self) -> None:
        self._items: dict[str, Meeting] = {}

    async def add(self, meeting: Meeting) -> None:
        self._items[meeting.id] = meeting

    async def get(self, meeting_id: str) -> Meeting | None:
        return self._items.get(meeting_id)

    async def list_all(self) -> list[Meeting]:
        return list(self._items.values())
