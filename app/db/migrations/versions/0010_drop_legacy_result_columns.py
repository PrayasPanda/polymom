"""drop legacy per-stage JSON columns from meetings (data now lives in run tables)

Downgrade re-creates the columns and refills them from each meeting's latest
successful run (or latest run), so no data is lost in either direction.

Revision ID: 0010
Revises: 0009
Create Date: 2026-09-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from app.db.base import UTCDateTime

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

STAGE_COLUMNS = {
    "preprocess": "audio_quality",
    "diarize": "diarization",
    "identify_languages": "language_summary",
    "transcribe": "transcript",
    "align": "speaker_transcript",
    "analytics": "analytics",
    "summarize": "summary",
}
LEGACY_COLUMNS = [*STAGE_COLUMNS.values(), "summary_error", "speaker_names"]


def upgrade() -> None:
    with op.batch_alter_table("meetings") as batch:
        for column in LEGACY_COLUMNS:
            batch.drop_column(column)


def downgrade() -> None:
    with op.batch_alter_table("meetings") as batch:
        for column in STAGE_COLUMNS.values():
            batch.add_column(sa.Column(column, sa.JSON(), nullable=True))
        batch.add_column(sa.Column("summary_error", sa.Text(), nullable=True))
        batch.add_column(sa.Column("speaker_names", sa.JSON(), nullable=True))

    bind = op.get_bind()
    meetings = sa.table(
        "meetings",
        sa.column("id", sa.Uuid()),
        sa.column("summary_error", sa.Text()),
        sa.column("speaker_names", sa.JSON()),
        *(sa.column(c, sa.JSON()) for c in STAGE_COLUMNS.values()),
    )
    runs = sa.table(
        "processing_runs",
        sa.column("id", sa.Uuid()),
        sa.column("meeting_id", sa.Uuid()),
        sa.column("status", sa.String()),
        sa.column("started_at", UTCDateTime()),
    )
    stages = sa.table(
        "stage_results",
        sa.column("run_id", sa.Uuid()),
        sa.column("stage_name", sa.String()),
        sa.column("status", sa.String()),
        sa.column("output", sa.JSON()),
        sa.column("error", sa.Text()),
    )
    summaries = sa.table(
        "summaries",
        sa.column("run_id", sa.Uuid()),
        sa.column("content", sa.JSON()),
        sa.column("created_at", UTCDateTime()),
    )
    speakers = sa.table(
        "speakers",
        sa.column("meeting_id", sa.Uuid()),
        sa.column("label", sa.String()),
        sa.column("display_name", sa.String()),
    )

    for (meeting_id,) in bind.execute(sa.select(meetings.c.id)).all():
        run_ids = [
            r.id
            for r in bind.execute(
                sa.select(runs.c.id, runs.c.status)
                .where(runs.c.meeting_id == meeting_id)
                .order_by(
                    runs.c.status.in_(["completed", "completed_with_errors"]).desc(),
                    runs.c.started_at.desc(),
                )
                .limit(1)
            )
        ]
        values: dict[str, object] = {}
        if run_ids:
            for s in bind.execute(sa.select(stages).where(stages.c.run_id == run_ids[0])):
                target = STAGE_COLUMNS.get(s.stage_name)
                if target is not None and s.status == "completed" and s.output is not None:
                    values[target] = s.output
                if s.stage_name == "summarize" and s.status == "failed":
                    values["summary_error"] = s.error
            latest_summary = bind.execute(
                sa.select(summaries.c.content)
                .where(summaries.c.run_id == run_ids[0])
                .order_by(summaries.c.created_at.desc())
                .limit(1)
            ).scalar()
            if latest_summary is not None:
                values["summary"] = latest_summary
        names = {
            r.label: r.display_name
            for r in bind.execute(sa.select(speakers).where(speakers.c.meeting_id == meeting_id))
            if r.display_name
        }
        values["speaker_names"] = names or None
        bind.execute(meetings.update().where(meetings.c.id == meeting_id).values(**values))
