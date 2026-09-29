"""add summary, summary_error; widen status for completed_with_errors

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("meetings") as batch:
        batch.add_column(sa.Column("summary", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("summary_error", sa.Text(), nullable=True))
        batch.alter_column("status", existing_type=sa.String(length=16), type_=sa.String(length=32))


def downgrade() -> None:
    op.execute("UPDATE meetings SET status = 'completed' WHERE status = 'completed_with_errors'")
    with op.batch_alter_table("meetings") as batch:
        batch.alter_column("status", existing_type=sa.String(length=32), type_=sa.String(length=16))
        batch.drop_column("summary_error")
        batch.drop_column("summary")
