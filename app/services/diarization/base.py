"""Diarization backend interface."""

import time
from abc import ABC, abstractmethod
from pathlib import Path

from app.core.config import Settings
from app.schemas.diarization import DiarizationResult
from app.services.diarization.postprocess import RawDiarization, build_result


class DiarizationBackend(ABC):
    """A speaker diarization model.

    Implementations provide :meth:`diarize_raw` (model output plus per-speaker
    embeddings, used to re-link speakers across chunks); :meth:`diarize` adds the
    shared post-processing.
    """

    model_name: str

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    async def diarize_raw(
        self,
        audio_path: Path,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> RawDiarization:
        """Run the model on a mono 16 kHz WAV. Must not block the event loop."""

    async def diarize(
        self,
        audio_path: Path,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> DiarizationResult:
        """Diarize a single file and return cleaned, consistently labelled turns."""
        started = time.perf_counter()
        raw = await self.diarize_raw(audio_path, num_speakers, min_speakers, max_speakers)
        return build_result(
            raw.segments,
            merge_gap=self.settings.merge_gap_seconds,
            min_turn=self.settings.min_turn_seconds,
            model_name=self.model_name,
            processing_time_ms=int((time.perf_counter() - started) * 1000),
        )
