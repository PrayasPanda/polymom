"""0009/0010: legacy JSON columns -> run tables and back, without data loss."""

import json
import uuid
from pathlib import Path

import sqlalchemy as sa

from app.db.migrate import downgrade_to, upgrade_to, upgrade_to_head
from app.db.session import create_engine

MEETING_ID = uuid.UUID("3f8b6f0e-2c1d-4d6a-9d3e-6c2b1a0f9e7d")
TRANSCRIPT = {
    "utterances": [
        {
            "id": 0,
            "speaker": "Person 1",
            "start": 0.0,
            "end": 3.0,
            "duration": 3.0,
            "text": "हम सहमत हैं",
            "words": [],
            "primary_language": "hi",
            "languages_present": ["hi"],
            "is_code_mixed": False,
            "avg_confidence": None,
            "has_overlap": False,
            "overlapping_speakers": [],
            "alignment_precision": "word",
        },
        {
            "id": 1,
            "speaker": "Person 2",
            "start": 3.0,
            "end": 6.0,
            "duration": 3.0,
            "text": "ଆମେ ରାଜି",
            "words": [],
            "primary_language": "or",
            "languages_present": ["or"],
            "is_code_mixed": False,
            "avg_confidence": None,
            "has_overlap": True,
            "overlapping_speakers": ["Person 1"],
            "alignment_precision": "segment",
        },
    ],
    "speakers": ["Person 1", "Person 2"],
    "total_duration": 6.0,
    "warnings": [],
    "alignment_stats": {
        "total_words": 5,
        "percent_assigned": 100,
        "percent_unknown": 0,
        "percent_segment_level": 40,
    },
}
DIARIZATION = {
    "turns": [
        {
            "speaker_label": "Person 1",
            "raw_label": "S0",
            "start": 0,
            "end": 3,
            "duration": 3,
            "is_overlap": False,
        },
        {
            "speaker_label": "Person 2",
            "raw_label": "S1",
            "start": 3,
            "end": 6,
            "duration": 3,
            "is_overlap": False,
        },
    ],
    "num_speakers": 2,
    "overlap_regions": [],
    "model_name": "pyannote/speaker-diarization-3.1",
    "processing_time_ms": 5,
}
SUMMARY = {
    "title": "Budget",
    "executive_summary": "Agreed.",
    "output_language": "en",
    "model_info": {"provider": "mock", "model": "mock-llm", "prompt_version": "system@1"},
    "verification_report": {"checked": 1},
}
ANALYTICS = {"speakers": [{"speaker": "Person 1", "speaking_time_seconds": 3.0}]}


async def test_upgrade_backfills_and_downgrade_restores(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'm.db').as_posix()}")
    await upgrade_to(engine, "0008")
    async with engine.begin() as conn:
        await conn.execute(
            sa.text(
                "INSERT INTO meetings (id, title, original_filename, stored_path, processed_path, "
                "mime_type, size_bytes, audio_metadata, languages_hint, status, created_at, "
                "updated_at, diarization, speaker_transcript, summary, analytics, audio_quality, "
                "speaker_names) VALUES (:id, 't', 'a.wav', :stored, :processed, 'audio/x-wav', 1, "
                "'{}', '[]', 'completed', '2026-09-28 10:00:00', '2026-09-28 10:05:00', :diar, "
                ":transcript, :summary, :analytics, :quality, :names)"
            ),
            {
                "id": MEETING_ID.hex,
                "stored": "C:\\data\\storage\\uploads\\abc.wav",
                "processed": "/app/storage/processed/abc.wav",
                "diar": json.dumps(DIARIZATION),
                "transcript": json.dumps(TRANSCRIPT, ensure_ascii=False),
                "summary": json.dumps(SUMMARY),
                "analytics": json.dumps(ANALYTICS),
                "quality": json.dumps({"duration_seconds": 6.0}),
                "names": json.dumps({"Person 1": "Ravi"}),
            },
        )

    await upgrade_to_head(engine)

    async with engine.connect() as conn:
        meeting = (
            await conn.execute(
                sa.text(
                    "SELECT upload_key, processed_key, detected_languages, num_speakers "
                    "FROM meetings"
                )
            )
        ).one()
        assert tuple(meeting) == ("uploads/abc.wav", "processed/abc.wav", ",hi,or,", 2)
        run = (
            await conn.execute(sa.text("SELECT id, status, model_versions FROM processing_runs"))
        ).one()
        assert run.status == "completed"
        assert json.loads(run.model_versions)["diarize"] == "pyannote/speaker-diarization-3.1"
        stages = {
            r.stage_name
            for r in await conn.execute(sa.text("SELECT stage_name FROM stage_results"))
        }
        assert stages == {"preprocess", "diarize", "align", "analytics", "summarize"}
        utts = (
            await conn.execute(
                sa.text("SELECT speaker, text, has_overlap FROM utterances ORDER BY start")
            )
        ).all()
        assert [tuple(u) for u in utts] == [("Person 1", "हम सहमत हैं", 0), ("Person 2", "ଆମେ ରାଜି", 1)]
        speakers = (
            await conn.execute(
                sa.text("SELECT label, display_name, stats FROM speakers ORDER BY label")
            )
        ).all()
        assert [(s.label, s.display_name) for s in speakers] == [
            ("Person 1", "Ravi"),
            ("Person 2", None),
        ]
        assert json.loads(speakers[0].stats)["speaking_time_seconds"] == 3.0
        summary = (await conn.execute(sa.text("SELECT model, content FROM summaries"))).one()
        assert summary.model == "mock-llm"
        hits = (
            await conn.execute(
                sa.text("SELECT ref FROM search_index WHERE search_index MATCH :q"), {"q": "सहमत"}
            )
        ).all()
        assert [h.ref for h in hits] == ["0"]
        columns = {
            r.name
            for r in await conn.execute(sa.text("SELECT name FROM pragma_table_info('meetings')"))
        }
        assert "diarization" not in columns
        assert "speaker_names" not in columns

    await downgrade_to(engine, "0008")

    async with engine.connect() as conn:
        row = (
            await conn.execute(
                sa.text(
                    "SELECT stored_path, diarization, speaker_transcript, summary, speaker_names "
                    "FROM meetings"
                )
            )
        ).one()
        assert row.stored_path == "uploads/abc.wav"
        assert json.loads(row.diarization) == DIARIZATION
        assert json.loads(row.speaker_transcript) == TRANSCRIPT
        assert json.loads(row.summary) == SUMMARY
        assert json.loads(row.speaker_names) == {"Person 1": "Ravi"}
        tables = {
            r.name
            for r in await conn.execute(
                sa.text("SELECT name FROM sqlite_master WHERE type='table'")
            )
        }
        assert "processing_runs" not in tables
        assert "search_index" not in tables

    await upgrade_to_head(engine)  # and forward again on the downgraded data
    async with engine.connect() as conn:
        assert (await conn.execute(sa.text("SELECT count(*) FROM utterances"))).scalar() == 2
    await engine.dispose()


async def test_fresh_database_upgrades(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite+aiosqlite:///{(tmp_path / 'f.db').as_posix()}")
    await upgrade_to_head(engine)
    async with engine.connect() as conn:
        assert (await conn.execute(sa.text("SELECT count(*) FROM meetings"))).scalar() == 0
    await engine.dispose()
