"""Application settings loaded from environment variables and ``.env``."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app import __version__


class Settings(BaseSettings):
    """Typed runtime configuration. Every field maps to an upper-case env var."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "polymom"
    app_version: str = __version__
    app_env: Literal["development", "staging", "production", "test"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    hf_token: SecretStr | None = None
    llm_provider: str = "openai"
    llm_api_key: SecretStr | None = None

    storage_dir: Path = Path("./storage")
    database_url: str | None = None
    auto_migrate: bool = True

    max_upload_mb: int = Field(default=200, gt=0)
    upload_chunk_bytes: int = Field(default=1024 * 1024, gt=0)
    allowed_extensions: Annotated[frozenset[str], NoDecode] = frozenset(
        {"wav", "mp3", "m4a", "flac", "ogg", "aac", "mp4", "mkv", "mov", "webm"}
    )

    ffprobe_path: str = "ffprobe"
    ffprobe_timeout_seconds: float = Field(default=30.0, gt=0)

    # Preprocessing
    ffmpeg_path: str = "ffmpeg"
    ffmpeg_timeout_seconds: float = Field(default=1800.0, gt=0)
    target_sample_rate: int = Field(default=16000, ge=8000, le=48000)
    target_loudness_lufs: float = Field(default=-23.0, le=0)
    enable_highpass: bool = True
    highpass_cutoff_hz: int = Field(default=80, gt=0)
    enable_denoise: bool = False
    trim_silence: bool = False
    silence_threshold_db: float = Field(default=-50.0, lt=0)
    silence_min_duration_seconds: float = Field(default=0.5, gt=0)

    # Chunking for long recordings
    chunk_length_seconds: float = Field(default=1800.0, gt=0)
    chunk_overlap_seconds: float = Field(default=5.0, ge=0)

    # Diarization
    diarization_backend: Literal["pyannote", "mock"] = "pyannote"
    diarization_model: str = "pyannote/speaker-diarization-3.1"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    merge_gap_seconds: float = Field(default=0.5, ge=0)
    min_turn_seconds: float = Field(default=0.3, ge=0)
    diarization_chunk_threshold_seconds: float = Field(default=3600.0, gt=0)
    speaker_similarity_threshold: float = Field(default=0.6, ge=-1, le=1)

    # ASR
    asr_backend: Literal["real", "mock"] = "real"
    whisper_model_size: str = "large-v3"
    whisper_compute_type: Literal["auto", "int8", "int8_float16", "float16", "float32"] = "auto"
    odia_model_id: str = "ai4bharat/indic-conformer-600m-multilingual"
    odia_decoding: Literal["ctc", "rnnt"] = "rnnt"
    asr_language_backends: Annotated[dict[str, str], NoDecode] = Field(
        default_factory=lambda: {"en": "whisper", "hi": "whisper", "or": "indic"}
    )
    asr_beam_size: int = Field(default=5, ge=1, le=20)
    asr_vad_filter: bool = True
    asr_low_confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    asr_compression_ratio_threshold: float = Field(default=2.4, gt=0)
    asr_no_speech_threshold: float = Field(default=0.6, ge=0, le=1)

    # Spoken language identification and code-switching
    supported_languages: Annotated[frozenset[str], NoDecode] = frozenset({"en", "hi", "or"})
    language_routing_enabled: bool = True
    lid_backend: Literal["mms", "speechbrain", "whisper", "mock"] = "mms"
    lid_model_id: str = "facebook/mms-lid-126"
    speechbrain_lid_model_id: str = "speechbrain/lang-id-voxlingua107-ecapa"
    lid_min_confidence: float = Field(default=0.5, ge=0, le=1)
    lid_min_window_seconds: float = Field(default=1.5, ge=0)
    lid_max_window_seconds: float = Field(default=15.0, gt=0)

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def _split_extensions(cls, value: object) -> object:
        """Accept a comma-separated string such as ``"wav,.MP3"``."""
        if isinstance(value, str):
            return frozenset(e.strip().lstrip(".").lower() for e in value.split(",") if e.strip())
        return value

    @field_validator("supported_languages", mode="before")
    @classmethod
    def _split_languages(cls, value: object) -> object:
        """Accept ``"en,hi,or"``."""
        if isinstance(value, str):
            return frozenset(v.strip().lower() for v in value.split(",") if v.strip())
        return value

    @field_validator("asr_language_backends", mode="before")
    @classmethod
    def _parse_language_backends(cls, value: object) -> object:
        """Accept ``"en:whisper,hi:whisper,or:indic"``."""
        if isinstance(value, str):
            mapping: dict[str, str] = {}
            for pair in filter(None, (p.strip() for p in value.split(","))):
                lang, sep, backend = pair.partition(":")
                if not sep or not lang.strip() or not backend.strip():
                    raise ValueError(f"expected 'lang:backend', got {pair!r}")
                mapping[lang.strip().lower()] = backend.strip().lower()
            return mapping
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def processed_dir(self) -> Path:
        return self.storage_dir / "processed"

    @property
    def resolved_database_url(self) -> str:
        """``DATABASE_URL`` if set, else a SQLite file inside ``STORAGE_DIR``."""
        if self.database_url:
            return self.database_url
        return f"sqlite+aiosqlite:///{(self.storage_dir / 'polymom.db').resolve().as_posix()}"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
