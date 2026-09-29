"""Speaker-attributed transcript schemas."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.asr import Word

UNKNOWN_SPEAKER = "Unknown"
AlignmentPrecision = Literal["word", "segment"]


class AlignedWord(Word):
    speaker: str
    overlapping_speakers: list[str] = Field(
        default_factory=list, description="Other speakers talking during this word."
    )
    alignment_precision: AlignmentPrecision = "word"


class Utterance(BaseModel):
    id: int
    speaker: str = Field(description='Diarization label, e.g. "Person 1"; never renumbered.')
    speaker_name: str | None = Field(
        default=None, description="Display name set via PATCH /speakers (output only)."
    )
    start: float
    end: float
    duration: float
    text: str
    words: list[AlignedWord]
    primary_language: str | None
    languages_present: list[str]
    is_code_mixed: bool
    avg_confidence: float | None
    has_overlap: bool
    overlapping_speakers: list[str]
    alignment_precision: AlignmentPrecision = Field(
        description="segment = word times estimated from segment/turn durations."
    )


class AlignmentStats(BaseModel):
    total_words: int
    percent_assigned: float
    percent_unknown: float
    percent_segment_level: float


class SpeakerTranscript(BaseModel):
    utterances: list[Utterance] = Field(description="Sorted by start, ties broken by speaker.")
    speakers: list[str]
    total_duration: float
    warnings: list[str]
    alignment_stats: AlignmentStats


class SpeakerTranscriptResponse(SpeakerTranscript):
    meeting_id: UUID
    speaker_names: dict[str, str] = Field(default_factory=dict)


class SpeakerRenameRequest(BaseModel):
    names: dict[str, str | None] = Field(
        description='{"Person 1": "Ravi"}; null removes a display name.',
        examples=[{"Person 1": "Ravi", "Person 2": "Sunita"}],
    )


class SpeakerNamesResponse(BaseModel):
    meeting_id: UUID
    speaker_names: dict[str, str]
