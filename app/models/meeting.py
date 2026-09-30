"""Meeting ORM entity: metadata, status and where its audio lives in the artifact store."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, Enum, Float, ForeignKey, Integer, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, JSONDocument, UTCDateTime, utcnow
from app.schemas.meeting import MeetingStatus

if TYPE_CHECKING:
    from app.models.results import ProcessingRun, Speaker


class Meeting(Base):
    """A meeting recording, its media metadata and processing state.

    Processed results live in run-scoped tables (:mod:`app.models.results`);
    ``detected_languages`` and ``num_speakers`` are denormalized from the latest
    successful run so meetings can be filtered cheaply.
    """

    __tablename__ = "meetings"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    title: Mapped[str | None] = mapped_column(String(200))
    original_filename: Mapped[str] = mapped_column(String(255))
    owner_key_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("api_keys.id", ondelete="SET NULL"), index=True
    )
    callback_url: Mapped[str | None] = mapped_column(
        String(2048), doc="Webhook POSTed with the final status (validated against SSRF)."
    )
    upload_key: Mapped[str] = mapped_column(String(1024))
    processed_key: Mapped[str | None] = mapped_column(String(1024))
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    mime_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    audio_metadata: Mapped[dict[str, Any]] = mapped_column(JSONDocument, default=dict)
    languages_hint: Mapped[list[str]] = mapped_column(JSONDocument, default=list)
    expected_speakers: Mapped[int | None] = mapped_column(Integer)
    detected_languages: Mapped[str | None] = mapped_column(
        String(64), doc='Comma-wrapped codes, e.g. ",en,hi,", for LIKE filtering.'
    )
    num_speakers: Mapped[int | None] = mapped_column(Integer)
    raw_audio_purged_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    status: Mapped[MeetingStatus] = mapped_column(
        Enum(
            MeetingStatus,
            native_enum=False,
            length=32,
            values_callable=lambda e: [m.value for m in e],
        ),
        default=MeetingStatus.QUEUED,
        index=True,
    )
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)

    runs: Mapped[list["ProcessingRun"]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan", passive_deletes=True
    )
    speakers: Mapped[list["Speaker"]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True
    )
