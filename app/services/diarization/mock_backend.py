"""Deterministic fake diarization for tests and machines without a GPU/HF token.

Speakers take turns in fixed-length slots: at time ``t`` the true speaker is
``floor(t / TURN_SECONDS) % n``. Like a real model, raw labels are assigned per
file by order of appearance, while embeddings identify the true speaker, so
cross-chunk re-linking behaves as it would with pyannote. Files whose peak level
is below ``SILENCE_PEAK`` produce no speech.
"""

import array
import math
import sys
import wave
from pathlib import Path

from starlette.concurrency import run_in_threadpool

from app.services.diarization.base import DiarizationBackend
from app.services.diarization.postprocess import RawDiarization, RawSegment

TURN_SECONDS = 2.0
DEFAULT_SPEAKERS = 2
EMBEDDING_DIM = 8
SILENCE_PEAK = 100  # of 32767, about -50 dBFS


def _read(path: Path) -> tuple[float, int]:
    """Duration and peak absolute sample of a 16-bit WAV."""
    with wave.open(str(path), "rb") as wav:
        rate, frames = wav.getframerate(), wav.getnframes()
        samples = array.array("h", wav.readframes(frames))
    if sys.byteorder == "big":  # pragma: no cover - WAV data is little-endian
        samples.byteswap()
    peak = max((abs(s) for s in samples), default=0)
    return (frames / rate if rate else 0.0), peak


def speaker_embedding(speaker: int) -> list[float]:
    """A distinct unit vector per true speaker."""
    return [1.0 if i == speaker % EMBEDDING_DIM else 0.0 for i in range(EMBEDDING_DIM)]


class MockDiarizationBackend(DiarizationBackend):
    model_name = "mock"

    async def diarize_raw(
        self,
        audio_path: Path,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> RawDiarization:
        duration, peak = await run_in_threadpool(_read, audio_path)
        if peak < SILENCE_PEAK or duration <= 0:
            return RawDiarization(segments=[])

        n = num_speakers or max_speakers or min_speakers or DEFAULT_SPEAKERS
        segments: list[RawSegment] = []
        local: dict[int, str] = {}
        for slot in range(math.ceil(duration / TURN_SECONDS)):
            speaker = slot % n
            label = local.setdefault(speaker, f"SPEAKER_{len(local):02d}")
            start = slot * TURN_SECONDS
            segments.append(RawSegment(start, min(start + TURN_SECONDS, duration), label))
        embeddings = {label: speaker_embedding(spk) for spk, label in local.items()}
        return RawDiarization(segments=segments, embeddings=embeddings)
