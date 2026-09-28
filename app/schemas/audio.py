"""Audio preprocessing and quality schemas."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class AudioWarningCode(StrEnum):
    LOW_VOLUME = "low_volume"
    CLIPPING = "clipping"
    TOO_SHORT = "too_short"
    MOSTLY_SILENT = "mostly_silent"


class AudioWarning(BaseModel):
    """A quality concern that does not stop processing."""

    code: AudioWarningCode
    message: str


class AudioQuality(BaseModel):
    """Preprocessing outcome persisted on the meeting and returned by the API.

    Level metrics (``loudness_lufs``, ``rms_db``, ``peak_db``) describe the
    original recording, before normalization, so they reflect capture quality.
    ``None`` means the level was not measurable (e.g. digital silence).
    """

    sample_rate: int
    channels: int
    duration_seconds: float = Field(description="Duration of the processed audio.")
    loudness_lufs: float | None = Field(description="Integrated loudness (EBU R128).")
    rms_db: float | None
    peak_db: float | None
    silence_ratio: float = Field(ge=0, le=1)
    leading_silence_seconds: float
    trailing_silence_seconds: float
    trimmed_seconds: float = Field(description="Silence removed when TRIM_SILENCE is on.")
    warnings: list[AudioWarning]
    processing_time_ms: int


class PreprocessResult(AudioQuality):
    processed_path: Path
