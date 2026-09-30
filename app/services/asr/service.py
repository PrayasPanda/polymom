"""Transcription orchestration: language strategy, chunking and post-processing."""

import shutil
import time
import uuid
import wave
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.asr import ASRResult, TranscriptSegment
from app.schemas.language import LanguageRegion
from app.services.asr.base import ASRBackend
from app.services.asr.indic_backend import IndicConformerBackend
from app.services.asr.mock_backend import MockASRBackend
from app.services.asr.postprocess import (
    ChunkTranscript,
    FilterConfig,
    filter_hallucinations,
    flag_low_confidence,
    language_durations,
    merge_chunks,
    normalize_segment,
    renumber,
)
from app.services.asr.router import ASRRouter
from app.services.asr.whisper_backend import WhisperBackend
from app.services.audio.chunker import split_wav
from app.services.language.text_tagger import tag_segment

if TYPE_CHECKING:
    from app.pipelines.checkpoints import ChunkHooks

logger = get_logger(__name__)


def build_router(settings: Settings) -> ASRRouter:
    """Router for ``ASR_BACKEND``: real models, or the mock behind every route."""
    backends: dict[str, ASRBackend]
    if settings.asr_backend == "mock":
        backends = {
            "whisper": MockASRBackend(settings, route="whisper"),
            "indic": MockASRBackend(settings, route="indic"),
        }
        backends["mock"] = backends["whisper"]
    else:
        backends = {"whisper": WhisperBackend(settings), "indic": IndicConformerBackend(settings)}
    return ASRRouter(backends, settings.asr_language_backends, auto_backend="whisper")


def choose_language(hints: Sequence[str]) -> str | None:
    """Exactly one hint forces that language; none or several use auto-detection.

    Several hints mean a code-mixed meeting: forcing one language would mangle
    the others, so Whisper decodes freely (per-region routing
    happens in the transcription stage when language ID is enabled).
    """
    return hints[0] if len(hints) == 1 else None


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


class TranscriptionService:
    def __init__(self, router: ASRRouter, settings: Settings) -> None:
        self.router = router
        self._settings = settings

    async def transcribe(
        self,
        meeting_id: uuid.UUID,
        audio_path: Path,
        language_hints: Sequence[str] = (),
        hooks: "ChunkHooks | None" = None,
    ) -> ASRResult:
        started = time.perf_counter()
        language = choose_language(language_hints)
        backend = self.router.select(language)

        duration = await run_in_threadpool(_wav_duration, audio_path)
        if duration > self._settings.chunk_length_seconds:
            segments = await self._transcribe_chunked(
                meeting_id, audio_path, backend, language, hooks
            )
        else:
            segments = (await backend.transcribe(audio_path, language)).segments

        return self._finalize(
            meeting_id, segments, [backend.model_name], language, started, backend.name
        )

    async def transcribe_routed(
        self,
        meeting_id: uuid.UUID,
        audio_path: Path,
        regions: Sequence[LanguageRegion],
        language_hints: Sequence[str] = (),
    ) -> ASRResult:
        """Per-region routing from language ID (see :meth:`ASRRouter.transcribe_regions`)."""
        started = time.perf_counter()
        routed = await self.router.transcribe_regions(
            audio_path, regions, max_batch_seconds=self._settings.chunk_length_seconds
        )
        return self._finalize(
            meeting_id,
            routed.segments,
            routed.model_names,
            choose_language(language_hints),
            started,
            "routed",
        )

    def _finalize(
        self,
        meeting_id: uuid.UUID,
        segments: Sequence[TranscriptSegment],
        model_names: Sequence[str],
        language: str | None,
        started: float,
        backend_name: str,
    ) -> ASRResult:
        """Normalize, drop hallucinations, flag low confidence, tag code-mixing, renumber."""
        settings = self._settings
        kept, dropped = filter_hallucinations(
            [normalize_segment(s) for s in segments],
            FilterConfig(
                low_confidence_threshold=settings.asr_low_confidence_threshold,
                compression_ratio_threshold=settings.asr_compression_ratio_threshold,
                no_speech_threshold=settings.asr_no_speech_threshold,
            ),
        )
        flagged = flag_low_confidence(kept, settings.asr_low_confidence_threshold)
        final = renumber([tag_segment(s) for s in flagged])
        result = ASRResult(
            segments=final,
            detected_languages=language_durations(final),
            model_names=sorted(set(model_names)),
            processing_time_ms=int((time.perf_counter() - started) * 1000),
            requested_language=language,
        )
        logger.info(
            "transcription_completed",
            meeting_id=str(meeting_id),
            backend=backend_name,
            language=language,
            segments=len(final),
            dropped=dict(Counter(reason for _, reason in dropped)),
            low_confidence=sum(s.low_confidence for s in final),
            code_mixed=sum(s.is_code_mixed for s in final),
            fallbacks=sum(s.fallback_used for s in final),
            languages=[d.language for d in result.detected_languages],
            processing_time_ms=result.processing_time_ms,
        )
        return result

    async def _transcribe_chunked(
        self,
        meeting_id: uuid.UUID,
        audio_path: Path,
        backend: ASRBackend,
        language: str | None,
        hooks: "ChunkHooks | None" = None,
    ) -> list[TranscriptSegment]:
        settings = self._settings
        chunk_dir = audio_path.parent / f"{meeting_id}_asr_chunks"
        try:
            chunks = await run_in_threadpool(
                split_wav,
                audio_path,
                chunk_dir,
                chunk_length_seconds=settings.chunk_length_seconds,
                overlap_seconds=settings.chunk_overlap_seconds,
            )
            logger.info("transcription_chunked", meeting_id=str(meeting_id), chunks=len(chunks))
            transcripts = []
            for index, chunk in enumerate(chunks):
                cached = await hooks.load(index) if hooks else None
                if cached is not None:
                    segments = [TranscriptSegment.model_validate(s) for s in cached]
                else:
                    segments = (
                        await backend.transcribe(chunk.path, language, offset=chunk.start_seconds)
                    ).segments
                    if hooks:
                        await hooks.save(index, [s.model_dump(mode="json") for s in segments])
                transcripts.append(
                    ChunkTranscript(
                        start=chunk.start_seconds, end=chunk.end_seconds, segments=segments
                    )
                )
                if hooks:
                    await hooks.after_chunk(index, len(chunks))
        finally:
            await run_in_threadpool(shutil.rmtree, chunk_dir, True)
        return merge_chunks(transcripts)
