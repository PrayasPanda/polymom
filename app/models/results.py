"""Run-scoped processing results.

One :class:`ProcessingRun` per pipeline execution keeps reprocessing history and
reproducibility (config snapshot, model versions). The latest successful run is
the default everywhere; older runs stay retrievable by ``run_id``.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, JSONDocument, UTCDateTime, utcnow

if TYPE_CHECKING:
    from app.models.meeting import Meeting


def _meeting_fk() -> ForeignKey:
    return ForeignKey("meetings.id", ondelete="CASCADE")


def _run_fk() -> ForeignKey:
    return ForeignKey("processing_runs.id", ondelete="CASCADE")


SUCCESSFUL_RUN_STATUSES = ("completed", "completed_with_errors")


class ProcessingRun(Base):
    __tablename__ = "processing_runs"
    __table_args__ = (Index("ix_processing_runs_meeting_started", "meeting_id", "started_at"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    meeting_id: Mapped[uuid.UUID] = mapped_column(Uuid, _meeting_fk())
    status: Mapped[str] = mapped_column(String(32), default="processing")
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONDocument, default=dict)
    model_versions: Mapped[dict[str, Any]] = mapped_column(JSONDocument, default=dict)
    timings_ms: Mapped[dict[str, int]] = mapped_column(JSONDocument, default=dict)
    error: Mapped[str | None] = mapped_column(Text)

    meeting: Mapped["Meeting"] = relationship(back_populates="runs")
    stages: Mapped[list["StageResult"]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True, order_by="StageResult.id"
    )


class StageResult(Base):
    __tablename__ = "stage_results"
    __table_args__ = (UniqueConstraint("run_id", "stage_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(Uuid, _run_fk(), index=True)
    stage_name: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    output: Mapped[Any | None] = mapped_column(JSONDocument)
    fingerprint: Mapped[str | None] = mapped_column(
        String(64), doc="sha256 of (input artifact, stage config, upstream fingerprints)."
    )
    output_ref: Mapped[str | None] = mapped_column(
        String(1024), doc="Artifact key when the output is too large to store inline."
    )
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Speaker(Base):
    """Meeting-wide speaker: the stable label, its display name and latest stats."""

    __tablename__ = "speakers"
    __table_args__ = (UniqueConstraint("meeting_id", "label"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[uuid.UUID] = mapped_column(Uuid, _meeting_fk(), index=True)
    label: Mapped[str] = mapped_column(String(64))
    display_name: Mapped[str | None] = mapped_column(String(100))
    stats: Mapped[dict[str, Any] | None] = mapped_column(JSONDocument)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class UtteranceRecord(Base):
    __tablename__ = "utterances"
    __table_args__ = (
        Index("ix_utterances_meeting_start", "meeting_id", "start"),
        UniqueConstraint("run_id", "utterance_index"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[uuid.UUID] = mapped_column(Uuid, _meeting_fk())
    run_id: Mapped[uuid.UUID] = mapped_column(Uuid, _run_fk(), index=True)
    utterance_index: Mapped[int] = mapped_column(Integer, doc="The utterance id in the transcript.")
    speaker: Mapped[str] = mapped_column(String(64))
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    primary_language: Mapped[str | None] = mapped_column(String(16))
    is_code_mixed: Mapped[bool] = mapped_column(Boolean, default=False)
    has_overlap: Mapped[bool] = mapped_column(Boolean, default=False)
    alignment_precision: Mapped[str] = mapped_column(String(16), default="word")


class SummaryRecord(Base):
    __tablename__ = "summaries"
    __table_args__ = (Index("ix_summaries_run_created", "run_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(Uuid, _run_fk())
    meeting_id: Mapped[uuid.UUID] = mapped_column(Uuid, _meeting_fk(), index=True)
    output_language: Mapped[str] = mapped_column(String(8))
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(200))
    prompt_version: Mapped[str] = mapped_column(String(200))
    content: Mapped[dict[str, Any]] = mapped_column(JSONDocument)
    verification_report: Mapped[dict[str, Any]] = mapped_column(JSONDocument)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class ArtifactCleanup(Base):
    """Artifact deletions that failed after their rows were deleted; retried by cleanup.py."""

    __tablename__ = "artifact_cleanup_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(1024))
    error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
