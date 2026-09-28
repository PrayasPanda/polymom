"""Fallback LID with faster-whisper's built-in language detection (en/hi only).

Reuses the ASR model already loaded for transcription, so it adds no download.
Whisper does not know Odia, so it can only choose between English and Hindi.
"""

from pathlib import Path

from starlette.concurrency import run_in_threadpool

from app.core.exceptions import LanguageIdError
from app.services.asr import whisper_backend
from app.services.audio.slicing import read_samples
from app.services.language.base import LanguageIdentifier


class WhisperLanguageIdentifier(LanguageIdentifier):
    name = "whisper"

    @property
    def model_name(self) -> str:
        return f"faster-whisper/{self.settings.whisper_model_size}:lid"

    @property
    def native_languages(self) -> frozenset[str]:
        return frozenset({"en", "hi"})

    def _scores(self, audio_path: Path, start: float, end: float) -> dict[str, float]:
        model = whisper_backend.load_model(self.settings)
        samples, _ = read_samples(audio_path, start, end)
        try:
            with whisper_backend._inference_lock:
                # Language detection happens eagerly; the segment generator is never consumed.
                _, info = model.transcribe(samples, language=None, vad_filter=False)
        except Exception as exc:
            raise LanguageIdError(
                f"Whisper language detection failed: {exc}",
                details={"reason": "inference_failed"},
            ) from exc
        return {str(lang): float(prob) for lang, prob in (info.all_language_probs or [])}

    async def scores(self, audio_path: Path, start: float, end: float) -> dict[str, float]:
        return await run_in_threadpool(self._scores, audio_path, start, end)
