"""CSV export of the per-speaker analytics table."""

import csv
import io

from app.schemas.analytics import ConversationAnalytics

CSV_COLUMNS = [
    "speaker",
    "display_name",
    "speaking_time_seconds",
    "speaking_time_percent_of_speech",
    "speaking_time_percent_of_meeting",
    "num_segments",
    "segment_share_percent",
    "num_turns",
    "avg_utterance_seconds",
    "median_utterance_seconds",
    "longest_utterance_seconds",
    "longest_utterance_start",
    "longest_utterance_end",
    "word_count",
    "words_per_minute",
    "first_spoke_at",
    "last_spoke_at",
    "interruptions_made",
    "interruptions_received",
    "overlap_seconds",
    "questions_asked",
    "languages",
]


def speakers_to_csv(analytics: ConversationAnalytics) -> str:
    """One row per speaker; ``display_name`` falls back to the label."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for s in analytics.speakers:
        longest = s.longest_utterance
        writer.writerow(
            {
                **s.model_dump(),
                "display_name": s.speaker_name or s.speaker,
                "longest_utterance_seconds": longest.duration if longest else "",
                "longest_utterance_start": longest.start if longest else "",
                "longest_utterance_end": longest.end if longest else "",
                "languages": ";".join(
                    f"{lang.language}:{lang.percentage}" for lang in s.language_breakdown
                ),
            }
        )
    return buf.getvalue()
