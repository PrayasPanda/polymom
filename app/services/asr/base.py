"""ASR backend interface."""

from abc import ABC, abstractmethod
from pathlib import Path

from app.core.config import Settings
from app.schemas.asr import ASRResult


class ASRBackend(ABC):
    """A speech recognition model.

    Backends return raw segments with absolute timestamps (``offset`` added).
    Normalization, hallucination filtering and confidence flags are applied
    centrally by :class:`~app.services.asr.service.TranscriptionService`.
    """

    #: Short routing name used in ``ASR_LANGUAGE_BACKENDS`` (e.g. ``whisper``).
    name: str

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def supported_languages(self) -> frozenset[str]:
        """ISO 639-1 codes this backend can transcribe when forced."""

    @property
    def supports_auto_detect(self) -> bool:
        """Whether ``language=None`` (automatic language detection) is allowed."""
        return False

    @abstractmethod
    async def transcribe(
        self, audio_path: Path, language: str | None, offset: float = 0.0
    ) -> ASRResult:
        """Transcribe a mono 16 kHz WAV. Must not block the event loop."""
