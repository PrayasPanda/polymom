"""Diarization orchestration: whole-file for normal meetings, chunked for long ones."""

import shutil
import time
import uuid
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.diarization import DiarizationResult
from app.services.audio.chunker import split_wav
from app.services.diarization.base import DiarizationBackend
from app.services.diarization.mock_backend import MockDiarizationBackend
from app.services.diarization.postprocess import (
    ChunkDiarization,
    RawDiarization,
    RawSegment,
    build_result,
    relink_chunks,
)
from app.services.diarization.pyannote_backend import PyannoteDiarizationBackend

if TYPE_CHECKING:
    from app.pipelines.checkpoints import ChunkHooks

logger = get_logger(__name__)


def build_backend(settings: Settings) -> DiarizationBackend:
    """Backend selected by ``DIARIZATION_BACKEND``."""
    if settings.diarization_backend == "mock":
        return MockDiarizationBackend(settings)
    return PyannoteDiarizationBackend(settings)


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


class DiarizationService:
    def __init__(self, backend: DiarizationBackend, settings: Settings) -> None:
        self.backend = backend
        self._settings = settings

    async def diarize(
        self,
        meeting_id: uuid.UUID,
        audio_path: Path,
        num_speakers: int | None = None,
        hooks: "ChunkHooks | None" = None,
    ) -> DiarizationResult:
        duration = await run_in_threadpool(_wav_duration, audio_path)
        if duration <= self._settings.diarization_chunk_threshold_seconds:
            result = await self.backend.diarize(audio_path, num_speakers=num_speakers)
        else:
            result = await self._diarize_chunked(meeting_id, audio_path, num_speakers, hooks)
        logger.info(
            "diarization_completed",
            meeting_id=str(meeting_id),
            num_speakers=result.num_speakers,
            turns=len(result.turns),
            overlaps=len(result.overlap_regions),
            model=result.model_name,
            processing_time_ms=result.processing_time_ms,
        )
        return result

    async def _diarize_chunked(
        self,
        meeting_id: uuid.UUID,
        audio_path: Path,
        num_speakers: int | None,
        hooks: "ChunkHooks | None" = None,
    ) -> DiarizationResult:
        """Diarize overlapping chunks, then re-link speakers by embedding similarity.

        A chunk may not contain every participant, so ``num_speakers`` is passed
        as an upper bound (``max_speakers``) rather than an exact count.
        """
        settings = self._settings
        started = time.perf_counter()
        chunk_dir = audio_path.parent / f"{meeting_id}_diarization_chunks"
        try:
            chunks = await run_in_threadpool(
                split_wav,
                audio_path,
                chunk_dir,
                chunk_length_seconds=settings.chunk_length_seconds,
                overlap_seconds=settings.chunk_overlap_seconds,
            )
            logger.info("diarization_chunked", meeting_id=str(meeting_id), chunks=len(chunks))
            results = []
            for index, chunk in enumerate(chunks):
                cached = await hooks.load(index) if hooks else None
                if cached is not None:
                    raw = raw_from_json(cached)
                else:
                    raw = await self.backend.diarize_raw(chunk.path, max_speakers=num_speakers)
                    if hooks:
                        await hooks.save(index, raw_to_json(raw))
                results.append(
                    ChunkDiarization(
                        offset=chunk.start_seconds, duration=chunk.duration_seconds, raw=raw
                    )
                )
                if hooks:
                    await hooks.after_chunk(index, len(chunks))
        finally:
            await run_in_threadpool(shutil.rmtree, chunk_dir, True)

        segments = relink_chunks(results, settings.speaker_similarity_threshold)
        return build_result(
            segments,
            merge_gap=settings.merge_gap_seconds,
            min_turn=settings.min_turn_seconds,
            model_name=self.backend.model_name,
            processing_time_ms=int((time.perf_counter() - started) * 1000),
        )


def raw_to_json(raw: RawDiarization) -> dict[str, Any]:
    return {
        "segments": [[seg.start, seg.end, seg.label] for seg in raw.segments],
        "embeddings": raw.embeddings,
    }


def raw_from_json(data: dict[str, Any]) -> RawDiarization:
    return RawDiarization(
        segments=[RawSegment(float(s), float(e), str(label)) for s, e, label in data["segments"]],
        embeddings={k: [float(x) for x in v] for k, v in data.get("embeddings", {}).items()},
    )
