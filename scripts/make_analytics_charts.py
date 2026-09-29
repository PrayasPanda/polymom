"""Render the README analytics charts from a synthetic 10-minute, 3-speaker meeting.

uv run python scripts/make_analytics_charts.py   # writes docs/images/*.png (needs `viz`)
"""

import random
from pathlib import Path

from app.schemas.diarization import SpeakerTurn
from app.schemas.transcript import AlignmentStats, SpeakerTranscript
from app.services.analytics.charts import speaking_time_chart, timeline_chart
from app.services.analytics.meeting_stats import build_analytics

OUT = Path("docs/images")


def synthetic_turns(seed: int = 7, duration: float = 600) -> list[SpeakerTurn]:
    rng = random.Random(seed)  # noqa: S311 - not crypto
    weights = {"Person 1": 5, "Person 2": 3, "Person 3": 1}
    turns, t, last = [], 0.0, ""
    while t < duration:
        speaker = rng.choices(
            [s for s in weights if s != last], [w for s, w in weights.items() if s != last]
        )[0]
        length = rng.uniform(3, 25) * weights[speaker] / 3
        start = max(t - rng.choice([0, 0, 0, 0.8]), 0)
        end = min(start + length, duration)
        turns.append(
            SpeakerTurn(
                speaker_label=speaker,
                raw_label=speaker,
                start=start,
                end=end,
                duration=end - start,
                is_overlap=start < t,
            )
        )
        t, last = end + rng.uniform(0, 2), speaker
    return turns


def main() -> None:
    turns = synthetic_turns()
    transcript = SpeakerTranscript(
        utterances=[],
        speakers=[],
        total_duration=600,
        warnings=[],
        alignment_stats=AlignmentStats(
            total_words=0, percent_assigned=0, percent_unknown=0, percent_segment_level=0
        ),
    )
    analytics = build_analytics(
        turns,
        transcript,
        interruption_min_overlap=0.5,
        bucket_seconds=60,
        gini_balanced_max=0.2,
        gini_dominated_min=0.4,
    )
    names = {"Person 1": "Ravi", "Person 2": "Sunita", "Person 3": "Arjun"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "speaking-time.png").write_bytes(speaking_time_chart(analytics, names))
    (OUT / "timeline.png").write_bytes(timeline_chart(analytics, names, turns))
    print(analytics.meeting_stats.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
