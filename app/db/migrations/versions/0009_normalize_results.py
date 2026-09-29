"""normalize stage outputs into runs, stage results, speakers, utterances, summaries

Creates the run-scoped tables and the search index, adds sha256 and filter columns,
renames stored_path/processed_path to artifact keys, and backfills one processing
run per already-processed meeting from the legacy JSON columns (which 0010 drops).

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-29
"""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import PurePath
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

from app.db.base import UTCDateTime

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON: sa.types.TypeEngine[Any] = sa.JSON().with_variant(JSONB(), "postgresql")

# Legacy column -> pipeline stage name.
LEGACY_STAGES = {
    "audio_quality": "preprocess",
    "diarization": "diarize",
    "language_summary": "identify_languages",
    "transcript": "transcribe",
    "speaker_transcript": "align",
    "analytics": "analytics",
    "summary": "summarize",
}


def _fk(table: str) -> sa.ForeignKey:
    return sa.ForeignKey(f"{table}.id", ondelete="CASCADE")


def create_search_index(dialect: str) -> None:
    if dialect == "sqlite":
        op.execute(
            "CREATE VIRTUAL TABLE search_index USING fts5("
            "text, kind UNINDEXED, meeting_id UNINDEXED, run_id UNINDEXED, ref UNINDEXED, "
            "start UNINDEXED, speaker UNINDEXED, tokenize='trigram')"
        )
        return
    op.execute(
        "CREATE TABLE search_index ("
        "id BIGSERIAL PRIMARY KEY, text TEXT NOT NULL, kind VARCHAR(16) NOT NULL, "
        "meeting_id UUID NOT NULL REFERENCES meetings(id) ON DELETE CASCADE, "
        "run_id UUID, ref VARCHAR(64), start DOUBLE PRECISION, speaker VARCHAR(64), "
        "tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED)"
    )
    op.execute("CREATE INDEX ix_search_index_tsv ON search_index USING GIN (tsv)")
    op.execute("CREATE INDEX ix_search_index_meeting ON search_index (meeting_id)")


def legacy_key(path: str | None, folder: str) -> str | None:
    """Absolute legacy file path -> key relative to STORAGE_DIR (the local store root)."""
    if not path:
        return None
    return f"{folder}/{PurePath(path.replace(chr(92), '/')).name}"


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    op.create_table(
        "processing_runs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("meeting_id", sa.Uuid(), _fk("meetings"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("started_at", UTCDateTime(), nullable=False),
        sa.Column("finished_at", UTCDateTime(), nullable=True),
        sa.Column("config_snapshot", JSON, nullable=False),
        sa.Column("model_versions", JSON, nullable=False),
        sa.Column("timings_ms", JSON, nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_processing_runs_meeting_started", "processing_runs", ["meeting_id", "started_at"]
    )
    op.create_table(
        "stage_results",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Uuid(), _fk("processing_runs"), nullable=False),
        sa.Column("stage_name", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("output", JSON, nullable=True),
        sa.Column("output_ref", sa.String(1024), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", UTCDateTime(), nullable=False),
        sa.UniqueConstraint("run_id", "stage_name"),
    )
    op.create_index("ix_stage_results_run_id", "stage_results", ["run_id"])
    op.create_table(
        "speakers",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("meeting_id", sa.Uuid(), _fk("meetings"), nullable=False),
        sa.Column("label", sa.String(64), nullable=False),
        sa.Column("display_name", sa.String(100), nullable=True),
        sa.Column("stats", JSON, nullable=True),
        sa.Column("updated_at", UTCDateTime(), nullable=False),
        sa.UniqueConstraint("meeting_id", "label"),
    )
    op.create_index("ix_speakers_meeting_id", "speakers", ["meeting_id"])
    op.create_table(
        "utterances",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("meeting_id", sa.Uuid(), _fk("meetings"), nullable=False),
        sa.Column("run_id", sa.Uuid(), _fk("processing_runs"), nullable=False),
        sa.Column("utterance_index", sa.Integer(), nullable=False),
        sa.Column("speaker", sa.String(64), nullable=False),
        sa.Column("start", sa.Float(), nullable=False),
        sa.Column("end", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("primary_language", sa.String(16), nullable=True),
        sa.Column("is_code_mixed", sa.Boolean(), nullable=False),
        sa.Column("has_overlap", sa.Boolean(), nullable=False),
        sa.Column("alignment_precision", sa.String(16), nullable=False),
        sa.UniqueConstraint("run_id", "utterance_index"),
    )
    op.create_index("ix_utterances_meeting_start", "utterances", ["meeting_id", "start"])
    op.create_index("ix_utterances_run_id", "utterances", ["run_id"])
    op.create_table(
        "summaries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Uuid(), _fk("processing_runs"), nullable=False),
        sa.Column("meeting_id", sa.Uuid(), _fk("meetings"), nullable=False),
        sa.Column("output_language", sa.String(8), nullable=False),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("model", sa.String(200), nullable=False),
        sa.Column("prompt_version", sa.String(200), nullable=False),
        sa.Column("content", JSON, nullable=False),
        sa.Column("verification_report", JSON, nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
    )
    op.create_index("ix_summaries_run_created", "summaries", ["run_id", "created_at"])
    op.create_index("ix_summaries_meeting_id", "summaries", ["meeting_id"])
    op.create_table(
        "artifact_cleanup_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("key", sa.String(1024), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("created_at", UTCDateTime(), nullable=False),
    )
    create_search_index(dialect)

    with op.batch_alter_table("meetings") as batch:
        batch.alter_column("stored_path", new_column_name="upload_key")
        batch.alter_column("processed_path", new_column_name="processed_key")
        batch.add_column(sa.Column("sha256", sa.String(64), nullable=True))
        batch.add_column(sa.Column("detected_languages", sa.String(64), nullable=True))
        batch.add_column(sa.Column("num_speakers", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("raw_audio_purged_at", UTCDateTime(), nullable=True))
        batch.create_index("ix_meetings_sha256", ["sha256"])

    backfill(bind)


def backfill(bind: sa.Connection) -> None:
    meetings = sa.table(
        "meetings",
        sa.column("id", sa.Uuid()),
        sa.column("status", sa.String()),
        sa.column("error", sa.Text()),
        sa.column("created_at", UTCDateTime()),
        sa.column("updated_at", UTCDateTime()),
        sa.column("upload_key", sa.String()),
        sa.column("processed_key", sa.String()),
        sa.column("detected_languages", sa.String()),
        sa.column("num_speakers", sa.Integer()),
        sa.column("summary_error", sa.Text()),
        sa.column("speaker_names", sa.JSON()),
        *(sa.column(c, sa.JSON()) for c in LEGACY_STAGES),
    )
    runs = sa.table(
        "processing_runs",
        *(
            sa.column(c, t)  # type: ignore[arg-type]
            for c, t in [
                ("id", sa.Uuid()),
                ("meeting_id", sa.Uuid()),
                ("status", sa.String()),
                ("started_at", UTCDateTime()),
                ("finished_at", UTCDateTime()),
                ("config_snapshot", JSON),
                ("model_versions", JSON),
                ("timings_ms", JSON),
                ("error", sa.Text()),
            ]
        ),
    )
    stages = sa.table(
        "stage_results",
        *(
            sa.column(c, t)  # type: ignore[arg-type]
            for c, t in [
                ("run_id", sa.Uuid()),
                ("stage_name", sa.String()),
                ("status", sa.String()),
                ("output", JSON),
                ("error", sa.Text()),
                ("created_at", UTCDateTime()),
            ]
        ),
    )
    speakers = sa.table(
        "speakers",
        *(
            sa.column(c, t)  # type: ignore[arg-type]
            for c, t in [
                ("meeting_id", sa.Uuid()),
                ("label", sa.String()),
                ("display_name", sa.String()),
                ("stats", JSON),
                ("updated_at", UTCDateTime()),
            ]
        ),
    )
    utterances = sa.table(
        "utterances",
        *(
            sa.column(c, t)  # type: ignore[arg-type]
            for c, t in [
                ("meeting_id", sa.Uuid()),
                ("run_id", sa.Uuid()),
                ("utterance_index", sa.Integer()),
                ("speaker", sa.String()),
                ("start", sa.Float()),
                ("end", sa.Float()),
                ("text", sa.Text()),
                ("primary_language", sa.String()),
                ("is_code_mixed", sa.Boolean()),
                ("has_overlap", sa.Boolean()),
                ("alignment_precision", sa.String()),
            ]
        ),
    )
    summaries = sa.table(
        "summaries",
        *(
            sa.column(c, t)  # type: ignore[arg-type]
            for c, t in [
                ("run_id", sa.Uuid()),
                ("meeting_id", sa.Uuid()),
                ("output_language", sa.String()),
                ("provider", sa.String()),
                ("model", sa.String()),
                ("prompt_version", sa.String()),
                ("content", JSON),
                ("verification_report", JSON),
                ("created_at", UTCDateTime()),
            ]
        ),
    )

    now = datetime.now(UTC)
    for row in bind.execute(sa.select(meetings)).mappings().all():
        mid = row["id"]
        values: dict[str, Any] = {
            "upload_key": legacy_key(row["upload_key"], "uploads") or "",
            "processed_key": legacy_key(row["processed_key"], "processed"),
        }
        outputs = {c: row[c] for c in LEGACY_STAGES if row[c] is not None}
        names: dict[str, str] = row["speaker_names"] or {}
        if outputs or row["summary_error"]:
            run_id = uuid.uuid4()
            status = row["status"] if row["status"] != "processing" else "failed"
            diar = outputs.get("diarization") or {}
            summary = outputs.get("summary") or {}
            model_versions = {
                k: v
                for k, v in {
                    "diarize": diar.get("model_name"),
                    "summarize": (summary.get("model_info") or {}).get("model"),
                }.items()
                if v
            }
            bind.execute(
                runs.insert().values(
                    id=run_id,
                    meeting_id=mid,
                    status=status,
                    started_at=row["created_at"] or now,
                    finished_at=row["updated_at"] or now,
                    config_snapshot={"migrated_from": "legacy meeting columns"},
                    model_versions=model_versions,
                    timings_ms={},
                    error=row["error"],
                )
            )
            for column, output in outputs.items():
                bind.execute(
                    stages.insert().values(
                        run_id=run_id,
                        stage_name=LEGACY_STAGES[column],
                        status="completed",
                        output=output,
                        created_at=now,
                    )
                )
            if row["summary_error"] and "summary" not in outputs:
                bind.execute(
                    stages.insert().values(
                        run_id=run_id,
                        stage_name="summarize",
                        status="failed",
                        error=row["summary_error"],
                        created_at=now,
                    )
                )
            transcript = outputs.get("speaker_transcript") or {}
            langs: set[str] = set()
            for u in transcript.get("utterances", []):
                langs.update(u.get("languages_present") or [])
                bind.execute(
                    utterances.insert().values(
                        meeting_id=mid,
                        run_id=run_id,
                        utterance_index=u["id"],
                        speaker=u["speaker"],
                        start=u["start"],
                        end=u["end"],
                        text=u["text"],
                        primary_language=u.get("primary_language"),
                        is_code_mixed=bool(u.get("is_code_mixed")),
                        has_overlap=bool(u.get("has_overlap")),
                        alignment_precision=u.get("alignment_precision") or "word",
                    )
                )
                insert_search_row(
                    bind,
                    "utterance",
                    mid,
                    run_id,
                    str(u["id"]),
                    u["start"],
                    u["speaker"],
                    u["text"],
                )
            stats = {s["speaker"]: s for s in (outputs.get("analytics") or {}).get("speakers", [])}
            labels = (
                set(transcript.get("speakers", []))
                | {t["speaker_label"] for t in diar.get("turns", [])}
                | set(stats)
                | set(names)
            )
            for label in sorted(labels):
                bind.execute(
                    speakers.insert().values(
                        meeting_id=mid,
                        label=label,
                        display_name=names.get(label),
                        stats=stats.get(label),
                        updated_at=now,
                    )
                )
            if summary:
                info = summary.get("model_info") or {}
                bind.execute(
                    summaries.insert().values(
                        run_id=run_id,
                        meeting_id=mid,
                        output_language=summary.get("output_language", "en"),
                        provider=info.get("provider", "unknown"),
                        model=info.get("model", "unknown"),
                        prompt_version=info.get("prompt_version", "unknown"),
                        content=summary,
                        verification_report=summary.get("verification_report") or {},
                        created_at=now,
                    )
                )
                insert_search_row(
                    bind,
                    "summary",
                    mid,
                    run_id,
                    "summary",
                    None,
                    None,
                    f"{summary.get('title', '')}\n{summary.get('executive_summary', '')}",
                )
            values["detected_languages"] = f",{','.join(sorted(langs))}," if langs else None
            values["num_speakers"] = len(labels - {"Unknown"}) or None
        bind.execute(meetings.update().where(meetings.c.id == mid).values(**values))


def insert_search_row(
    bind: sa.Connection,
    kind: str,
    meeting_id: uuid.UUID,
    run_id: uuid.UUID,
    ref: str,
    start: float | None,
    speaker: str | None,
    text: str,
) -> None:
    params = {
        "text": text,
        "kind": kind,
        "meeting_id": str(meeting_id) if bind.dialect.name == "sqlite" else meeting_id,
        "run_id": str(run_id) if bind.dialect.name == "sqlite" else run_id,
        "ref": ref,
        "start": start,
        "speaker": speaker,
    }
    bind.execute(
        sa.text(
            "INSERT INTO search_index (text, kind, meeting_id, run_id, ref, start, speaker) "
            "VALUES (:text, :kind, :meeting_id, :run_id, :ref, :start, :speaker)"
        ),
        params,
    )


def downgrade() -> None:
    """Drops the run tables. Paths stay as keys relative to STORAGE_DIR (not absolute);
    run 0010's downgrade first so the legacy JSON columns are refilled."""
    op.execute("DROP TABLE search_index")
    for table in ("artifact_cleanup_log", "summaries", "utterances", "speakers", "stage_results"):
        op.drop_table(table)
    op.drop_table("processing_runs")
    with op.batch_alter_table("meetings") as batch:
        batch.drop_index("ix_meetings_sha256")
        for column in ("raw_audio_purged_at", "num_speakers", "detected_languages", "sha256"):
            batch.drop_column(column)
        batch.alter_column("processed_key", new_column_name="processed_path")
        batch.alter_column("upload_key", new_column_name="stored_path")
