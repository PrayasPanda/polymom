"""Meeting request/response schemas."""

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

Language = Literal["en", "hi", "or"]
SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"en", "hi", "or"})

_EXAMPLE_ID = "3f8b6f0e-2c1d-4d6a-9d3e-6c2b1a0f9e7d"
_EXAMPLE_TS = "2026-09-28T10:15:00Z"


class MeetingStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AudioMetadata(BaseModel):
    """Stream details reported by ffprobe for the first audio stream."""

    duration_seconds: float | None = None
    codec: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    format_name: str | None = None
    bit_rate: int | None = None


class MeetingCreateResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"meeting_id": _EXAMPLE_ID, "status": "queued", "created_at": _EXAMPLE_TS}]
        }
    )

    meeting_id: UUID
    status: MeetingStatus
    created_at: datetime


class MeetingRead(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "meeting_id": _EXAMPLE_ID,
                    "title": "Weekly sync",
                    "original_filename": "standup.m4a",
                    "mime_type": "audio/mp4",
                    "size_bytes": 1843200,
                    "duration_seconds": 312.4,
                    "audio_metadata": {
                        "duration_seconds": 312.4,
                        "codec": "aac",
                        "sample_rate": 44100,
                        "channels": 2,
                        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                        "bit_rate": 128000,
                    },
                    "languages_hint": ["en", "hi"],
                    "expected_speakers": 4,
                    "status": "queued",
                    "error": None,
                    "created_at": _EXAMPLE_TS,
                    "updated_at": _EXAMPLE_TS,
                }
            ]
        }
    )

    meeting_id: UUID
    title: str | None
    original_filename: str
    mime_type: str
    size_bytes: int
    duration_seconds: float | None
    audio_metadata: AudioMetadata
    languages_hint: list[Language]
    expected_speakers: int | None
    status: MeetingStatus
    error: str | None
    created_at: datetime
    updated_at: datetime


class MeetingList(BaseModel):
    items: list[MeetingRead]
    total: int = Field(description="Total number of meetings, ignoring pagination.")
    limit: int
    offset: int
