"""Spoken language identification and code-switching schemas."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class LanguageScore(BaseModel):
    language: str
    score: float


class LanguagePrediction(BaseModel):
    """Audio-level LID for one window, restricted to the supported/hinted languages."""

    language: str = Field(description="Best supported language (a guess if uncertain).")
    confidence: float = Field(description="Renormalized probability of `language`.")
    top_k: list[LanguageScore]
    uncertain: bool = Field(description="True when confidence < LID_MIN_CONFIDENCE.")


class LanguageRegion(BaseModel):
    """A stretch of audio assigned to one language (and speaker) after smoothing."""

    start: float
    end: float
    language: str
    speaker: str | None = None
    confidence: float


class LanguageShare(BaseModel):
    language: str
    duration_seconds: float
    percentage: float


class SpeakerLanguages(BaseModel):
    speaker: str
    dominant_language: str
    languages: list[LanguageShare]


class SwitchPoint(BaseModel):
    timestamp: float
    from_language: str
    to_language: str
    speaker: str | None = Field(description="Who is speaking right after the switch.")


class LanguageSummary(BaseModel):
    languages: list[LanguageShare] = Field(description="Longest first.")
    speakers: list[SpeakerLanguages]
    num_switches: int
    switch_points: list[SwitchPoint]
    code_mixed_segments: int = Field(
        default=0, description="Transcript segments mixing languages at the word level."
    )
    regions: list[LanguageRegion]
    lid_model: str
    candidate_languages: list[str] = Field(description="Languages LID could choose from.")


class LanguageSummaryResponse(LanguageSummary):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "meeting_id": "3f8b6f0e-2c1d-4d6a-9d3e-6c2b1a0f9e7d",
                    "languages": [
                        {"language": "hi", "duration_seconds": 312.4, "percentage": 62.5},
                        {"language": "or", "duration_seconds": 120.0, "percentage": 24.0},
                        {"language": "en", "duration_seconds": 67.5, "percentage": 13.5},
                    ],
                    "speakers": [
                        {
                            "speaker": "Person 1",
                            "dominant_language": "hi",
                            "languages": [
                                {"language": "hi", "duration_seconds": 250.0, "percentage": 80.0}
                            ],
                        }
                    ],
                    "num_switches": 7,
                    "switch_points": [
                        {
                            "timestamp": 95.2,
                            "from_language": "hi",
                            "to_language": "or",
                            "speaker": "Person 2",
                        }
                    ],
                    "code_mixed_segments": 12,
                    "regions": [
                        {
                            "start": 0.0,
                            "end": 95.2,
                            "language": "hi",
                            "speaker": "Person 1",
                            "confidence": 0.93,
                        }
                    ],
                    "lid_model": "facebook/mms-lid-126",
                    "candidate_languages": ["en", "hi", "or"],
                }
            ]
        }
    )

    meeting_id: UUID
