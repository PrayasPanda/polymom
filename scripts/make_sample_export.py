"""Render docs/samples/code-mixed-meeting.{result.json,md,pdf,docx} from the code-mixed fixture.

    uv run python scripts/make_sample_export.py                          # LLM_PROVIDER from .env
    LLM_PROVIDER=mock uv run python scripts/make_sample_export.py        # offline, deterministic

The transcript is the hand-written Hinglish + Odia fixture
(tests/fixtures/meetings/mixed_standup.json); analytics and exports are computed by the
real code, and the minutes come from whichever LLM is configured (recorded in
``summary.model_info``).
"""

import asyncio
import json
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.schemas.diarization import SpeakerTurn
from app.schemas.meeting import AudioMetadata, MeetingRead, MeetingStatus
from app.schemas.result import MeetingResult, SpeakerInfo
from app.services.analytics.meeting_stats import build_analytics
from app.services.export import render_export
from app.services.llm import build_llm_client
from app.services.summarization.summarizer import Summarizer
from scripts.eval_summary import load_transcript

FIXTURE = Path("tests/fixtures/meetings/mixed_standup.json")
OUT = Path("docs/samples")
STEM = "code-mixed-meeting"
NAMES = {"Person 1": "Ravi", "Person 2": "सुनीता", "Person 3": "ପ୍ରିୟା"}


async def build(fixture: dict[str, Any]) -> MeetingResult:
    transcript = load_transcript(fixture)
    settings = Settings()
    llm = build_llm_client(settings)
    summary = await Summarizer(llm, settings).summarize(
        transcript, meeting_date=date.fromisoformat(fixture["date"])
    )
    turns = [
        SpeakerTurn(
            speaker_label=u.speaker,
            raw_label=u.speaker,
            start=u.start,
            end=u.end,
            duration=u.duration,
            is_overlap=False,
        )
        for u in transcript.utterances
    ]
    analytics = build_analytics(
        turns,
        transcript,
        interruption_min_overlap=0.5,
        bucket_seconds=60,
        gini_balanced_max=0.2,
        gini_dominated_min=0.4,
    )
    stamp = datetime(2026, 9, 28, 10, 0, tzinfo=UTC)
    return MeetingResult(
        meeting=MeetingRead(
            meeting_id=uuid.UUID(int=0),
            title=fixture["title"],
            original_filename="standup.wav",
            mime_type="audio/x-wav",
            size_bytes=0,
            duration_seconds=transcript.total_duration,
            audio_metadata=AudioMetadata(),
            languages_hint=[],
            expected_speakers=None,
            detected_languages=["en", "hi", "or"],
            status=MeetingStatus.COMPLETED,
            error=None,
            created_at=stamp,
            updated_at=stamp,
        ),
        processing=None,
        speakers=[
            SpeakerInfo(label=label, display_name=name, stats=stats)
            for label, name in NAMES.items()
            for stats in [next((s for s in analytics.speakers if s.speaker == label), None)]
        ],
        transcript=transcript,
        analytics=analytics,
        summary=summary,
        verification_report=summary.verification_report,
        included=["transcript", "analytics", "summary"],
    )


def main() -> None:
    result = asyncio.run(build(json.loads(FIXTURE.read_text(encoding="utf-8"))))
    OUT.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "docx", "md"):
        (OUT / f"{STEM}.{fmt}").write_bytes(render_export(fmt, result))  # type: ignore[arg-type]
    (OUT / f"{STEM}.result.json").write_text(
        result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT}/{STEM}.result.json, .md, .pdf and .docx")


if __name__ == "__main__":
    main()
