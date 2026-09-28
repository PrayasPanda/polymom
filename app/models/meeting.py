"""Meeting persistence model (placeholder dataclass until an ORM is chosen)."""

from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.schemas.meeting import MeetingStatus


@dataclass(slots=True)
class Meeting:
    """A meeting recording and its processing state. TODO: replace with an ORM entity."""

    id: str
    filename: str
    status: MeetingStatus = MeetingStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
