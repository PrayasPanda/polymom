"""Speech recognition schemas.

All text is Unicode NFC in its native script (Latin, Devanagari, Odia);
nothing is ever transliterated.
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Word(BaseModel):
    text: str
    start: float
    end: float
    confidence: float | None = Field(
        default=None, description="0-1 word probability; null when the model gives none."
    )
    language: str | None = Field(
        default=None,
        description="en, hi, or; hi-Latn / or-Latn for romanized Hindi/Odia; null if neutral.",
    )
    script: str | None = Field(default=None, description="ISO 15924: Latn, Deva, Orya.")


class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    language: str | None = Field(description="ISO 639-1 code, e.g. en, hi, or.")
    words: list[Word]
    avg_confidence: float | None
    backend: str = Field(description="ASR backend that produced this segment.")
    low_confidence: bool = False
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    primary_language: str | None = Field(
        default=None, description="Language of most words in the segment."
    )
    languages_present: list[str] = Field(default_factory=list)
    is_code_mixed: bool = False
    code_mix_ratio: float = Field(
        default=0.0, description="Share of language-tagged words not in the primary language."
    )
    lid_confidence: float | None = Field(
        default=None, description="Audio LID confidence of the region this segment came from."
    )
    fallback_used: bool = Field(
        default=False, description="The routed backend failed; Whisper transcribed instead."
    )


class LanguageDuration(BaseModel):
    language: str
    duration_seconds: float


class ASRResult(BaseModel):
    segments: list[TranscriptSegment] = Field(description="Sorted by start time.")
    detected_languages: list[LanguageDuration] = Field(
        description="Speech time per language, longest first."
    )
    model_names: list[str]
    processing_time_ms: int
    requested_language: str | None = Field(
        default=None, description="Language forced from the upload hint, if any."
    )


class TranscriptResponse(ASRResult):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "meeting_id": "3f8b6f0e-2c1d-4d6a-9d3e-6c2b1a0f9e7d",
                    "segments": [
                        {
                            "id": 0,
                            "start": 0.52,
                            "end": 3.9,
                            "text": "आज की मीटिंग का एजेंडा बजट है।",
                            "language": "hi",
                            "words": [
                                {"text": "आज", "start": 0.52, "end": 0.8, "confidence": 0.97}
                            ],
                            "avg_confidence": 0.91,
                            "backend": "whisper",
                            "low_confidence": False,
                            "no_speech_prob": 0.01,
                            "compression_ratio": 1.1,
                        }
                    ],
                    "detected_languages": [{"language": "hi", "duration_seconds": 3.38}],
                    "model_names": ["faster-whisper/large-v3"],
                    "processing_time_ms": 5120,
                    "requested_language": "hi",
                }
            ]
        }
    )

    meeting_id: UUID
