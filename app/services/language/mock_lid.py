"""Deterministic fake LID for tests and machines without ML extras.

The language changes every ``BLOCK_SECONDS`` of meeting time, cycling through
en -> hi -> or (the window midpoint decides). The chosen language scores 0.8;
the remaining 0.2 is split evenly across the other supported languages.
"""

from pathlib import Path

from app.services.language.base import LanguageIdentifier

BLOCK_SECONDS = 6.0
ROTATION = ("en", "hi", "or")
TOP_SCORE = 0.8


class MockLanguageIdentifier(LanguageIdentifier):
    name = "mock"

    @property
    def model_name(self) -> str:
        return "mock-lid"

    @property
    def native_languages(self) -> frozenset[str]:
        return frozenset(ROTATION)

    async def scores(self, audio_path: Path, start: float, end: float) -> dict[str, float]:
        chosen = ROTATION[int(((start + end) / 2) // BLOCK_SECONDS) % len(ROTATION)]
        others = [lang for lang in ROTATION if lang != chosen]
        return {chosen: TOP_SCORE, **dict.fromkeys(others, (1 - TOP_SCORE) / len(others))}
