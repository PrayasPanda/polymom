"""Deterministic fake ASR with scripted English, Hindi and Odia lines.

One segment every ``SEGMENT_SECONDS`` of non-silent audio. A forced language
uses that script; with no language the lines rotate en -> hi -> or, mimicking a
code-mixed meeting. Indic lines are stored in decomposed (NFD) form on purpose,
so the pipeline's NFC normalization is exercised end to end.
"""

import array
import math
import unicodedata
import wave
from pathlib import Path

from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.schemas.asr import ASRResult, TranscriptSegment
from app.services.asr.base import ASRBackend
from app.services.asr.postprocess import approximate_words

SEGMENT_SECONDS = 3.0
SILENCE_PEAK = 100
CONFIDENCE = 0.92

SCRIPTS: dict[str, list[str]] = {
    "en": [
        "Good morning everyone, let us begin.",
        "The first item is the quarterly budget.",
        "We agreed to ship the release on Friday.",
    ],
    "hi": [
        "नमस्ते सभी को, चलिए शुरू करते हैं।",
        "पहला मुद्दा तिमाही बजट का है।",
        "हम शुक्रवार को रिलीज़ करने पर सहमत हैं।",
    ],
    "or": [
        "ସମସ୍ତଙ୍କୁ ନମସ୍କାର, ଆଜିର ଯୋଜନା ଆରମ୍ଭ କରିବା।",
        "ପ୍ରଥମ ବିଷୟ ହେଉଛି ତ୍ରୈମାସିକ ବଜେଟ୍।",
        "ଶୁକ୍ରବାର ଦିନ ରିଲିଜ୍ କରିବାକୁ ଆମେ ରାଜି।",
    ],
}
ROTATION = ("en", "hi", "or")


def _read(path: Path) -> tuple[float, int]:
    with wave.open(str(path), "rb") as wav:
        rate, frames = wav.getframerate(), wav.getnframes()
        samples = array.array("h", wav.readframes(frames))
    return (frames / rate if rate else 0.0), max((abs(s) for s in samples), default=0)


class MockASRBackend(ASRBackend):
    name = "mock"

    def __init__(self, settings: Settings, route: str | None = None) -> None:
        super().__init__(settings)
        if route:
            self.name = f"mock-{route}"

    @property
    def model_name(self) -> str:
        return "mock-asr"

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset(SCRIPTS)

    @property
    def supports_auto_detect(self) -> bool:
        return True

    async def transcribe(
        self, audio_path: Path, language: str | None, offset: float = 0.0
    ) -> ASRResult:
        duration, peak = await run_in_threadpool(_read, audio_path)
        segments: list[TranscriptSegment] = []
        if peak >= SILENCE_PEAK:
            for i in range(math.ceil(duration / SEGMENT_SECONDS)):
                lang = language or ROTATION[i % len(ROTATION)]
                text = SCRIPTS[lang][i % len(SCRIPTS[lang])]
                if lang != "en":
                    text = unicodedata.normalize("NFD", text)
                start = offset + i * SEGMENT_SECONDS
                end = offset + min((i + 1) * SEGMENT_SECONDS, duration)
                words = [
                    w.model_copy(update={"confidence": CONFIDENCE})
                    for w in approximate_words(text, start, end)
                ]
                segments.append(
                    TranscriptSegment(
                        id=i,
                        start=round(start, 3),
                        end=round(end, 3),
                        text=text,
                        language=lang,
                        words=words,
                        avg_confidence=CONFIDENCE,
                        backend=self.name,
                    )
                )
        return ASRResult(
            segments=segments,
            detected_languages=[],
            model_names=[self.model_name],
            processing_time_ms=0,
            requested_language=language,
        )
