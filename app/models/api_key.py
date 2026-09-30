"""API keys: hashed tokens presented via ``X-API-Key``.

Keys are stored as SHA-256 hex digests (never in plaintext). The plaintext is
shown once when the key is created (``scripts/create_api_key.py``) and never
again. Every meeting stores ``owner_key_id`` so one client cannot read another's
data (object-level authorization, aka OWASP API1:2023).
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UTCDateTime, utcnow


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    prefix: Mapped[str] = mapped_column(String(16), index=True, doc="First 8 chars, for display.")
    label: Mapped[str] = mapped_column(String(200))
    owner: Mapped[str | None] = mapped_column(String(200))
    scopes: Mapped[str] = mapped_column(String(200), default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class IdempotencyRecord(Base):
    """Per-key record of processed Idempotency-Keys, so retries return the first result."""

    __tablename__ = "idempotency_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    api_key_id: Mapped[int] = mapped_column(index=True)
    idempotency_key: Mapped[str] = mapped_column(String(200), index=True)
    method: Mapped[str] = mapped_column(String(16))
    path: Mapped[str] = mapped_column(String(512))
    status_code: Mapped[int] = mapped_column()
    response_body: Mapped[str] = mapped_column(String(8192))
    meeting_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
