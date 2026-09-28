"""Split long processed WAVs into overlapping chunks for model inference.

Chunks are sample-accurate slices written with the stdlib ``wave`` module, so no
re-encoding happens. Each chunk records its absolute offset into the source,
letting later stages map chunk-relative timestamps back to meeting time.
"""

import math
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AudioChunk:
    index: int
    path: Path
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def plan_chunks(
    duration_seconds: float, chunk_length_seconds: float, overlap_seconds: float
) -> list[tuple[float, float]]:
    """Return ``(start, end)`` windows covering ``[0, duration]``.

    Consecutive windows overlap by ``overlap_seconds``; the last one ends exactly
    at ``duration``. Audio no longer than one chunk yields a single window.
    """
    if chunk_length_seconds <= 0:
        raise ValueError("chunk_length_seconds must be positive")
    if not 0 <= overlap_seconds < chunk_length_seconds:
        raise ValueError("overlap_seconds must be >= 0 and < chunk_length_seconds")
    if duration_seconds <= 0:
        return []

    step = chunk_length_seconds - overlap_seconds
    windows: list[tuple[float, float]] = []
    start = 0.0
    while True:
        end = min(start + chunk_length_seconds, duration_seconds)
        windows.append((start, end))
        if end >= duration_seconds:
            return windows
        start += step


def split_wav(
    source: Path,
    output_dir: Path,
    *,
    chunk_length_seconds: float,
    overlap_seconds: float,
) -> list[AudioChunk]:
    """Write ``{stem}_chunk_{i:04d}.wav`` files for each planned window.

    Blocking (file I/O); call via a thread pool from async code.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[AudioChunk] = []
    with wave.open(str(source), "rb") as src:
        params = src.getparams()
        rate = params.framerate
        duration = params.nframes / rate
        for index, (start, end) in enumerate(
            plan_chunks(duration, chunk_length_seconds, overlap_seconds)
        ):
            first = round(start * rate)
            last = min(params.nframes, math.ceil(end * rate))
            src.setpos(first)
            frames = src.readframes(last - first)
            path = output_dir / f"{source.stem}_chunk_{index:04d}.wav"
            with wave.open(str(path), "wb") as dst:
                dst.setparams(params)
                dst.writeframes(frames)
            chunks.append(
                AudioChunk(
                    index=index, path=path, start_seconds=first / rate, end_seconds=last / rate
                )
            )
    return chunks
