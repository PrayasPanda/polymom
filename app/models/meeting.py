"""Meeting ORM entity."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, Enum, Float, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UTCDateTime, utcnow
from app.schemas.meeting import MeetingStatus


class Meeting(Base):
    """A meeting recording, its media metadata and processing state."""

    __tablename__ = "meetings"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    title: Mapped[str | None] = mapped_column(String(200))
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(1024))
    mime_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    audio_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    languages_hint: Mapped[list[str]] = mapped_column(JSON, default=list)
    expected_speakers: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[MeetingStatus] = mapped_column(
        Enum(
            MeetingStatus,
            native_enum=False,
            length=16,
            values_callable=lambda e: [m.value for m in e],
        ),
        default=MeetingStatus.QUEUED,
        index=True,
    )
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)
