"""api_keys, idempotency_records, meetings.owner_key_id; add cancelled status

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.db.base import UTCDateTime

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("key_hash", sa.String(64), nullable=False),
        sa.Column("prefix", sa.String(16), nullable=False),
        sa.Column("label", sa.String(200), nullable=False),
        sa.Column("owner", sa.String(200), nullable=True),
        sa.Column("scopes", sa.String(200), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.Column("last_used_at", UTCDateTime(), nullable=True),
        sa.UniqueConstraint("key_hash"),
    )
    op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"], unique=True)
    op.create_index("ix_api_keys_prefix", "api_keys", ["prefix"])

    op.create_table(
        "idempotency_records",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("api_key_id", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(200), nullable=False),
        sa.Column("method", sa.String(16), nullable=False),
        sa.Column("path", sa.String(512), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("response_body", sa.String(8192), nullable=False),
        sa.Column("meeting_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.UniqueConstraint("api_key_id", "idempotency_key", name="uq_idempotency_key_per_api_key"),
    )
    op.create_index("ix_idempotency_records_api_key_id", "idempotency_records", ["api_key_id"])
    op.create_index(
        "ix_idempotency_records_idempotency_key", "idempotency_records", ["idempotency_key"]
    )
    op.create_index("ix_idempotency_records_created_at", "idempotency_records", ["created_at"])

    # Add owner_key_id (with an inline FK on Postgres; SQLite ignores the FK, which
    # is fine here: cross-tenant scoping is enforced in the service layer, not the DB).
    dialect = op.get_bind().dialect.name
    with op.batch_alter_table("meetings") as batch:
        if dialect == "postgresql":
            batch.add_column(
                sa.Column(
                    "owner_key_id",
                    sa.Integer(),
                    sa.ForeignKey("api_keys.id", ondelete="SET NULL"),
                    nullable=True,
                )
            )
        else:
            batch.add_column(sa.Column("owner_key_id", sa.Integer(), nullable=True))
        batch.create_index("ix_meetings_owner_key_id", ["owner_key_id"])


def downgrade() -> None:
    with op.batch_alter_table("meetings") as batch:
        batch.drop_index("ix_meetings_owner_key_id")
        batch.drop_column("owner_key_id")
    op.drop_index("ix_idempotency_records_created_at", table_name="idempotency_records")
    op.drop_index("ix_idempotency_records_idempotency_key", table_name="idempotency_records")
    op.drop_index("ix_idempotency_records_api_key_id", table_name="idempotency_records")
    op.drop_table("idempotency_records")
    op.drop_index("ix_api_keys_prefix", table_name="api_keys")
    op.drop_index("ix_api_keys_key_hash", table_name="api_keys")
    op.drop_table("api_keys")
