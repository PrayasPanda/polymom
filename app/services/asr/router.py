"""Language -> ASR backend routing.

Configured by ``ASR_LANGUAGE_BACKENDS`` (``en:whisper,hi:whisper,or:indic``).
:meth:`ASRRouter.select` picks a backend for one language;
:meth:`ASRRouter.transcribe_regions` transcribes a meeting region by region,
each with the backend (and forced language) for that region's language.
"""

import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from starlette.concurrency import run_in_threadpool

from app.core.exceptions import UnsupportedLanguageError
from app.core.logging import get_logger
from app.schemas.asr import ASRResult, TranscriptSegment
from app.schemas.language import LanguageRegion
from app.services.asr.base import ASRBackend
from app.services.audio.slicing import write_wav_slice

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RegionBatch:
    """Consecutive regions of one language, transcribed with a single model call."""

    start: float
    end: float
    language: str
    confidence: float
    regions: tuple[LanguageRegion, ...]


def batch_regions(
    regions: Sequence[LanguageRegion], max_batch_seconds: float = float("inf")
) -> list[RegionBatch]:
    """Group chronologically consecutive regions that share a language.

    Batches only ever break *between* regions (turn boundaries), never inside
    one, and a new batch starts once the span would exceed ``max_batch_seconds``.
    Confidence is the duration-weighted mean of the grouped regions.
    """
    batches: list[list[LanguageRegion]] = []
    for region in sorted(regions, key=lambda r: (r.start, r.end)):
        current = batches[-1] if batches else None
        if (
            current
            and current[-1].language == region.language
            and region.end - current[0].start <= max_batch_seconds
        ):
            current.append(region)
        else:
            batches.append([region])
    result = []
    for group in batches:
        total = sum(r.end - r.start for r in group)
        conf = sum(r.confidence * (r.end - r.start) for r in group) / total if total else 0.0
        result.append(
            RegionBatch(
                start=group[0].start,
                end=group[-1].end,
                language=group[0].language,
                confidence=round(conf, 4),
                regions=tuple(group),
            )
        )
    return result


class ASRRouter:
    def __init__(
        self,
        backends: Mapping[str, ASRBackend],
        language_backends: Mapping[str, str],
        auto_backend: str = "whisper",
    ) -> None:
        unknown = {name for name in language_backends.values() if name not in backends}
        if auto_backend not in backends:
            unknown.add(auto_backend)
        if unknown:
            raise ValueError(
                f"ASR_LANGUAGE_BACKENDS references unknown backend(s) {sorted(unknown)}; "
                f"available: {sorted(backends)}"
            )
        self.backends = dict(backends)
        self.language_backends = dict(language_backends)
        self.auto_backend = auto_backend

    @property
    def languages(self) -> frozenset[str]:
        return frozenset(self.language_backends)

    def select(self, language: str | None) -> ASRBackend:
        """Backend for ``language``; ``None`` means automatic language detection."""
        if language is None:
            backend = self.backends[self.auto_backend]
            if not backend.supports_auto_detect:
                raise UnsupportedLanguageError(
                    f"Backend '{backend.name}' cannot auto-detect the language; "
                    "pass a language hint on upload.",
                    details={"backend": backend.name},
                )
            return backend

        language = language.lower()
        name = self.language_backends.get(language)
        if name is None:
            raise UnsupportedLanguageError(
                f"No ASR backend is configured for language '{language}'. "
                "Add it to ASR_LANGUAGE_BACKENDS (e.g. 'or:indic').",
                details={"language": language, "configured": sorted(self.language_backends)},
            )
        backend = self.backends[name]
        if language not in backend.supported_languages:
            raise UnsupportedLanguageError(
                f"ASR backend '{name}' does not support language '{language}'. "
                f"It supports: {', '.join(sorted(backend.supported_languages))}.",
                details={"language": language, "backend": name},
            )
        return backend

    async def transcribe_regions(
        self,
        audio_path: Path,
        regions: Sequence[LanguageRegion],
        *,
        max_batch_seconds: float = float("inf"),
    ) -> ASRResult:
        """Transcribe each language region with its own backend and stitch the results.

        Consecutive same-language regions are batched; each batch is cut into a
        temporary WAV (removed afterwards), transcribed with the language forced,
        and its timestamps shifted back to meeting time. If a non-default backend
        fails (e.g. the Odia model), the batch is retried with the auto-detect
        backend and its segments are flagged ``fallback_used``.
        """
        started = time.perf_counter()
        batches = batch_regions(regions, max_batch_seconds)
        segments: list[TranscriptSegment] = []
        models: set[str] = set()
        with tempfile.TemporaryDirectory(dir=audio_path.parent, prefix="asr_regions_") as tmp:
            for i, batch in enumerate(batches):
                piece = Path(tmp) / f"batch_{i:04d}_{batch.language}.wav"
                offset = await run_in_threadpool(
                    write_wav_slice, audio_path, piece, batch.start, batch.end
                )
                result, fallback = await self._transcribe_batch(piece, batch, offset)
                models.update(result.model_names)
                segments.extend(
                    s.model_copy(
                        update={"lid_confidence": batch.confidence, "fallback_used": fallback}
                    )
                    for s in result.segments
                )
        logger.info(
            "routed_transcription",
            batches=len(batches),
            regions=len(regions),
            per_language={
                lang: sum(1 for b in batches if b.language == lang)
                for lang in sorted({b.language for b in batches})
            },
            fallbacks=sum(1 for s in segments if s.fallback_used),
        )
        return ASRResult(
            segments=sorted(segments, key=lambda s: (s.start, s.end)),
            detected_languages=[],
            model_names=sorted(models),
            processing_time_ms=int((time.perf_counter() - started) * 1000),
        )

    async def _transcribe_batch(
        self, piece: Path, batch: RegionBatch, offset: float
    ) -> tuple[ASRResult, bool]:
        backend = self.select(batch.language)
        try:
            return await backend.transcribe(piece, batch.language, offset=offset), False
        except Exception as exc:
            fallback = self.backends[self.auto_backend]
            if backend is fallback:
                raise
            logger.warning(
                "asr_backend_fallback",
                backend=backend.name,
                fallback=fallback.name,
                language=batch.language,
                start=batch.start,
                end=batch.end,
                error=str(exc),
            )
            # No forced language: Whisper cannot be forced to a language it lacks.
            return await fallback.transcribe(piece, None, offset=offset), True
