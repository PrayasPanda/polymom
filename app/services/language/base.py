"""Spoken language identification interface and score handling."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path

from app.core.config import Settings
from app.schemas.language import LanguagePrediction, LanguageScore

# Below this, models see too little audio to say anything useful.
MIN_MODEL_SECONDS = 0.4
TOP_K = 3


def restrict_scores(scores: Mapping[str, float], allowed: Iterable[str]) -> dict[str, float]:
    """Keep only ``allowed`` languages and renormalize their scores to sum to 1.

    A model trained on 100+ languages spreads probability over languages we do
    not handle (e.g. Bengali for Odia speech). Restricting to the candidates and
    renormalizing turns "which of *these* is it" into a proper distribution.
    Returns a uniform distribution if none of the allowed languages scored.
    """
    allowed = sorted(set(allowed))
    kept = {lang: max(0.0, float(scores.get(lang, 0.0))) for lang in allowed}
    total = sum(kept.values())
    if total <= 0:
        return dict.fromkeys(allowed, 1.0 / len(allowed)) if allowed else {}
    return {lang: score / total for lang, score in kept.items()}


def predict(
    scores: Mapping[str, float], allowed: Iterable[str], min_confidence: float
) -> LanguagePrediction:
    """Pick the best allowed language; mark ``uncertain`` below ``min_confidence``."""
    restricted = restrict_scores(scores, allowed)
    if not restricted:
        raise ValueError("no candidate languages")
    ranked = sorted(restricted.items(), key=lambda kv: (-kv[1], kv[0]))
    language, confidence = ranked[0]
    return LanguagePrediction(
        language=language,
        confidence=round(confidence, 4),
        top_k=[LanguageScore(language=lang, score=round(s, 4)) for lang, s in ranked[:TOP_K]],
        uncertain=confidence < min_confidence,
    )


class LanguageIdentifier(ABC):
    """Audio language ID over a time window of a mono 16 kHz WAV."""

    name: str

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def native_languages(self) -> frozenset[str]:
        """Supported languages this model can actually recognize."""

    @abstractmethod
    async def scores(self, audio_path: Path, start: float, end: float) -> dict[str, float]:
        """Raw per-language probabilities (ISO 639-1 codes where possible)."""

    async def identify(
        self,
        audio_path: Path,
        start: float,
        end: float,
        candidates: Iterable[str] | None = None,
    ) -> LanguagePrediction:
        """Most likely language among ``candidates`` (default: supported languages).

        Candidates the model cannot recognize are dropped (e.g. Odia for Whisper);
        windows too short for the model come back ``uncertain`` without a model call.
        """
        allowed = set(candidates or self.settings.supported_languages)
        allowed &= set(self.settings.supported_languages)
        usable = allowed & self.native_languages or allowed
        if end - start < MIN_MODEL_SECONDS:
            return predict({}, usable, 1.0).model_copy(update={"uncertain": True})
        raw = await self.scores(audio_path, start, end)
        return predict(raw, usable, self.settings.lid_min_confidence)
