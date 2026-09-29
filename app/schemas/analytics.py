"""Speaker-wise conversation statistics and meeting-level analytics schemas.

Speakers are identified by their diarization label ("Person N"); display names
from ``PATCH /speakers`` are added as ``speaker_name`` when the analytics are read.
"""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from app.schemas.language import LanguageShare


def _round2(value: float) -> float:
    return round(value, 2)


Rounded = Annotated[float, AfterValidator(_round2)]
"""Durations (seconds) and percentages, always rounded to 2 decimals."""

ParticipationBalance = Literal["balanced", "moderately dominated", "dominated", "not applicable"]


class TimeSpan(BaseModel):
    start: Rounded
    end: Rounded
    duration: Rounded


class Monologue(TimeSpan):
    speaker: str


class SpeakerStats(BaseModel):
    speaker: str = Field(description='Diarization label, e.g. "Person 1".')
    speaker_name: str | None = None
    speaking_time_seconds: Rounded
    speaking_time_percent_of_speech: Rounded
    speaking_time_percent_of_meeting: Rounded
    num_segments: int
    segment_share_percent: Rounded
    num_turns: int
    avg_utterance_seconds: Rounded
    median_utterance_seconds: Rounded
    longest_utterance: TimeSpan | None
    word_count: int
    words_per_minute: Rounded
    first_spoke_at: Rounded | None
    last_spoke_at: Rounded | None
    interruptions_made: int
    interruptions_received: int
    overlap_seconds: Rounded
    language_breakdown: list[LanguageShare]
    questions_asked: int


class MeetingStats(BaseModel):
    meeting_duration_seconds: Rounded
    total_speech_seconds: Rounded
    total_silence_seconds: Rounded
    silence_percent: Rounded
    num_speakers: int
    total_utterances: int
    total_words: int
    total_turn_switches: int
    overlap_seconds: Rounded
    overlap_percent: Rounded
    overlap_double_counted_seconds: Rounded
    gini_coefficient: Rounded
    participation_balance: ParticipationBalance
    dominant_speaker: str | None
    least_active_speaker: str | None
    language_distribution: list[LanguageShare]
    num_language_switches: int
    longest_monologue: Monologue | None


class TimelineBucket(BaseModel):
    start: Rounded
    end: Rounded
    speakers: dict[str, Rounded] = Field(description="Speaking seconds per speaker label.")


METRIC_DEFINITIONS: dict[str, str] = {
    "speaking_time_seconds": "Union of the speaker's diarization turns; overlap counts in full "
    "for every overlapping speaker.",
    "speaking_time_percent_of_speech": "Speaking time / total speech time; sums to over 100 "
    "when speech overlaps.",
    "speaking_time_percent_of_meeting": "Speaking time / meeting duration.",
    "num_segments": "Utterances attributed to the speaker in the aligned transcript.",
    "segment_share_percent": "Share of all attributed utterances; sums to 100.",
    "num_turns": "Times the floor switched to the speaker (consecutive turns merged).",
    "avg_utterance_seconds": "Mean utterance duration.",
    "median_utterance_seconds": "Median utterance duration.",
    "longest_utterance": "Longest single utterance with its timestamps.",
    "word_count": "Words attributed to the speaker.",
    "words_per_minute": "Word count / speaking time in minutes.",
    "first_spoke_at": "Start of the speaker's first turn (seconds).",
    "last_spoke_at": "End of the speaker's last turn (seconds).",
    "interruptions_made": "Turns started while another speaker was talking, overlapping them "
    "by more than INTERRUPTION_MIN_OVERLAP_SECONDS.",
    "interruptions_received": "Times another speaker interrupted this speaker.",
    "overlap_seconds": "Time this speaker talked while someone else was talking.",
    "language_breakdown": "Seconds and percent per spoken language.",
    "questions_asked": 'Utterances ending in "?" or containing a Hindi/Odia question word.',
    "meeting_duration_seconds": "Length of the processed recording.",
    "total_speech_seconds": "Time at least one person is speaking (union of all turns).",
    "total_silence_seconds": "Meeting duration minus total speech time.",
    "total_turn_switches": "Times the floor passed from one speaker to another.",
    "overlap_seconds_meeting": "Time two or more people speak at once.",
    "overlap_percent": "Overlap seconds / total speech time.",
    "overlap_double_counted_seconds": "Sum of speaking times minus total speech time.",
    "gini_coefficient": "Inequality of speaking time: 0 = equal, towards 1 = one speaker.",
    "participation_balance": "Gini <= GINI_BALANCED_MAX balanced; >= GINI_DOMINATED_MIN "
    "dominated; else moderately dominated.",
    "dominant_speaker": "Speaker with the most speaking time.",
    "least_active_speaker": "Speaker with the least speaking time.",
    "num_language_switches": "Changes of spoken language between consecutive regions.",
    "longest_monologue": "Longest run of speech by one speaker without a floor switch.",
    "timeline": "Speaking seconds per speaker in fixed BUCKET_SECONDS windows.",
}


class ConversationAnalytics(BaseModel):
    meeting_stats: MeetingStats
    speakers: list[SpeakerStats] = Field(description="Most speaking time first.")
    timeline: list[TimelineBucket]
    metric_definitions: dict[str, str] = Field(default_factory=lambda: dict(METRIC_DEFINITIONS))


class ConversationAnalyticsResponse(ConversationAnalytics):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "meeting_id": "3f8b6f0e-2c1d-4d6a-9d3e-6c2b1a0f9e7d",
                    "speaker_names": {"Person 1": "Ravi"},
                    "meeting_stats": {
                        "meeting_duration_seconds": 1800.0,
                        "total_speech_seconds": 1620.4,
                        "participation_balance": "moderately dominated",
                        "dominant_speaker": "Person 1",
                    },
                    "speakers": [
                        {
                            "speaker": "Person 1",
                            "speaker_name": "Ravi",
                            "speaking_time_seconds": 812.3,
                            "words_per_minute": 131.5,
                        }
                    ],
                }
            ]
        }
    )

    meeting_id: UUID
    speaker_names: dict[str, str] = Field(default_factory=dict)
