"""add speaker_transcript and speaker_names JSON to meetings

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("meetings") as batch:
        batch.add_column(sa.Column("speaker_transcript", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("speaker_names", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("meetings") as batch:
        batch.drop_column("speaker_names")
        batch.drop_column("speaker_transcript")
