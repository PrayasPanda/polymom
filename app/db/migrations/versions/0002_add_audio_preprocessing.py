"""add processed_path and audio_quality to meetings

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("meetings") as batch:
        batch.add_column(sa.Column("processed_path", sa.String(length=1024), nullable=True))
        batch.add_column(sa.Column("audio_quality", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("meetings") as batch:
        batch.drop_column("audio_quality")
        batch.drop_column("processed_path")
