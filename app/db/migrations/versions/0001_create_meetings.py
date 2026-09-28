"""create meetings table

Revision ID: 0001
Revises:
Create Date: 2026-09-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.db.base import UTCDateTime

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "meetings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=True),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("stored_path", sa.String(length=1024), nullable=False),
        sa.Column("mime_type", sa.String(length=100), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("audio_metadata", sa.JSON(), nullable=False),
        sa.Column("languages_hint", sa.JSON(), nullable=False),
        sa.Column("expected_speakers", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_meetings")),
    )
    op.create_index(op.f("ix_meetings_created_at"), "meetings", ["created_at"])
    op.create_index(op.f("ix_meetings_status"), "meetings", ["status"])


def downgrade() -> None:
    op.drop_index(op.f("ix_meetings_status"), table_name="meetings")
    op.drop_index(op.f("ix_meetings_created_at"), table_name="meetings")
    op.drop_table("meetings")
