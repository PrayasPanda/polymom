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
    max_upload_mb: int = Field(default=200, gt=0)
    allowed_extensions: Annotated[frozenset[str], NoDecode] = frozenset(
        {"wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4"}
    )

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def _split_extensions(cls, value: object) -> object:
        """Accept a comma-separated string such as ``"wav,.MP3"``."""
        if isinstance(value, str):
            return frozenset(e.strip().lstrip(".").lower() for e in value.split(",") if e.strip())
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
