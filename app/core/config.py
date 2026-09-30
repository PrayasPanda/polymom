"""Application settings loaded from environment variables and ``.env``."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
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

    # LLM summarization
    llm_provider: Literal["openai", "azure", "anthropic", "ollama", "mock"] = "openai"
    llm_model: str | None = Field(
        default=None, description="Empty = provider default (see DEFAULT_LLM_MODELS)."
    )
    llm_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_base_url: str = "https://api.anthropic.com"
    azure_openai_endpoint: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = Field(default=0.2, ge=0, le=2)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_output_tokens: int = Field(default=4096, gt=0)
    llm_input_cost_per_mtok: float = Field(default=0.0, ge=0, description="USD per 1M tokens.")
    llm_output_cost_per_mtok: float = Field(default=0.0, ge=0)
    summary_output_language: Literal["en", "hi", "or"] = "en"
    summary_single_pass_tokens: int = Field(default=12000, gt=0)
    summary_chunk_tokens: int = Field(default=6000, gt=0)
    summary_chunk_overlap_utterances: int = Field(default=2, ge=0)
    evidence_match_threshold: float = Field(default=80.0, ge=0, le=100)
    langfuse_enabled: bool = False
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    storage_dir: Path = Path("./storage")
    database_url: str | None = None
    auto_migrate: bool = True

    # Artifact storage (uploads, processed audio, charts, exports)
    artifact_store: Literal["local", "s3"] = "local"
    s3_bucket: str = "polymom"
    s3_endpoint_url: str | None = Field(default=None, description="Set for MinIO.")
    s3_access_key: SecretStr | None = None
    s3_secret_key: SecretStr | None = None
    s3_region: str | None = None
    stage_output_inline_max_bytes: int = Field(
        default=512 * 1024, gt=0, description="Larger stage outputs go to the artifact store."
    )

    # Retention (scripts/cleanup.py)
    retention_days: int = Field(default=90, ge=0, description="0 disables raw-audio purging.")
    keep_raw_audio: bool = Field(
        default=False, description="true keeps uploads and processed audio forever."
    )

    # Job queue and execution
    redis_url: str | None = Field(
        default=None, description="Required for PIPELINE_EXECUTION=queue."
    )
    pipeline_execution: Literal["queue", "inline"] = Field(
        default="queue",
        description="inline runs the pipeline inside the API request (tests, debug).",
    )
    queue_concurrency_cpu: int = Field(default=4, ge=1)
    queue_concurrency_gpu: int = Field(default=1, ge=1, description="Jobs per GPU worker.")
    queue_concurrency_llm: int = Field(default=4, ge=1)
    stage_timeouts: Annotated[dict[str, float], NoDecode] = Field(
        default_factory=lambda: {
            "preprocess": 1800.0,
            "diarize": 7200.0,
            "identify_languages": 3600.0,
            "transcribe": 10800.0,
            "align": 600.0,
            "analytics": 600.0,
            "summarize": 1800.0,
        }
    )
    max_retries: int = Field(default=3, ge=0, le=10, description="Transient failures only.")
    retry_backoff_seconds: float = Field(default=10.0, ge=0)
    stuck_job_seconds: int = Field(default=600, ge=30)
    heartbeat_seconds: int = Field(default=15, ge=1)
    shutdown_grace_seconds: int = Field(default=120, ge=0)
    worker_metrics_port: int = Field(default=9101, ge=0, description="0 disables.")

    # Limits
    max_audio_duration_minutes: float = Field(default=240.0, gt=0)
    max_json_body_kb: int = Field(default=1024, gt=0)

    # Webhooks
    webhook_secret: SecretStr | None = None
    webhook_timeout_seconds: float = Field(default=10.0, gt=0)
    webhook_max_attempts: int = Field(default=5, ge=1)
    webhook_allow_private_hosts: bool = Field(
        default=False, description="Allow callbacks to private IPs (local testing only)."
    )

    # Security
    api_key_required: bool = True
    rate_limit_default: str = "120/minute"
    rate_limit_upload: str = "10/minute"
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # Observability
    otel_enabled: bool = False
    otel_service_name: str = "polymom"
    metrics_enabled: bool = True

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

    # Speaker / transcript alignment
    align_max_gap_seconds: float = Field(default=1.0, ge=0)
    align_merge_gap_seconds: float = Field(default=1.0, ge=0)
    utterance_max_seconds: float = Field(default=30.0, gt=0)
    utterance_min_words: int = Field(default=2, ge=1)

    # Speaker analytics
    interruption_min_overlap_seconds: float = Field(default=0.5, ge=0)
    bucket_seconds: float = Field(default=60.0, gt=0)
    gini_balanced_max: float = Field(default=0.2, ge=0, le=1)
    gini_dominated_min: float = Field(default=0.4, ge=0, le=1)

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

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept ``"https://a.example,https://b.example"``."""
        if isinstance(value, str):
            return [o.strip().rstrip("/") for o in value.split(",") if o.strip()]
        return value

    @field_validator("stage_timeouts", mode="before")
    @classmethod
    def _parse_timeouts(cls, value: object) -> object:
        """Accept ``"diarize:3600,transcribe:7200"``; unspecified stages keep defaults."""
        if isinstance(value, str):
            parsed: dict[str, float] = {}
            for pair in filter(None, (p.strip() for p in value.split(","))):
                stage, sep, seconds = pair.partition(":")
                if not sep:
                    raise ValueError(f"expected 'stage:seconds', got {pair!r}")
                parsed[stage.strip()] = float(seconds)
            return parsed
        return value

    @model_validator(mode="after")
    def _fail_fast_in_production(self) -> "Settings":
        """Refuse to start in production with missing secrets or unsafe settings."""
        if self.app_env != "production":
            return self
        problems = []
        if self.pipeline_execution == "queue" and not self.redis_url:
            problems.append("REDIS_URL is required with PIPELINE_EXECUTION=queue")
        if not self.api_key_required:
            problems.append("API_KEY_REQUIRED must be true in production")
        if self.llm_provider in ("openai", "azure", "anthropic") and not self.llm_api_key:
            problems.append(f"LLM_API_KEY is required for LLM_PROVIDER={self.llm_provider}")
        if self.diarization_backend == "pyannote" and not self.hf_token:
            problems.append("HF_TOKEN is required for DIARIZATION_BACKEND=pyannote")
        if self.artifact_store == "s3" and not (self.s3_access_key and self.s3_secret_key):
            problems.append("S3_ACCESS_KEY and S3_SECRET_KEY are required for ARTIFACT_STORE=s3")
        if self.webhook_allow_private_hosts:
            problems.append("WEBHOOK_ALLOW_PRIVATE_HOSTS must be false in production")
        if "*" in self.cors_origins:
            problems.append("CORS_ORIGINS must not contain '*' in production")
        if problems:
            raise ValueError("Invalid production configuration: " + "; ".join(problems))
        return self

    def stage_timeout(self, stage: str) -> float:
        return self.stage_timeouts.get(stage, 3600.0)

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
