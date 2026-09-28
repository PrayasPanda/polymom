"""Meeting request/response schemas.

TODO(prompt-2+): add transcript segments, speaker stats, summary, decisions, action items.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class MeetingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class MeetingCreateResponse(BaseModel):
    meeting_id: str
    status: MeetingStatus


class MeetingRead(BaseModel):
    meeting_id: str
    status: MeetingStatus
    filename: str | None = None
    created_at: datetime | None = None
