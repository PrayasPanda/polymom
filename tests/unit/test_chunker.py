import wave
from pathlib import Path

import pytest

from app.core.config import Settings
from app.services.audio.chunker import plan_chunks, split_wav
from tests.conftest import requires_ffmpeg


def _frames(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        return wav.readframes(wav.getnframes())


@pytest.mark.parametrize(
    ("duration", "length", "overlap", "expected"),
    [
        (180, 60, 5, [(0, 60), (55, 115), (110, 170), (165, 180)]),
        (120, 60, 0, [(0, 60), (60, 120)]),
        (60, 60, 5, [(0, 60)]),
        (10, 1800, 5, [(0, 10)]),
        (0, 60, 5, []),
    ],
)
def test_plan_chunks(
    duration: float, length: float, overlap: float, expected: list[tuple[float, float]]
) -> None:
    assert plan_chunks(duration, length, overlap) == expected


@pytest.mark.parametrize(("length", "overlap"), [(0, 0), (-5, 0), (60, 60), (60, -1)])
def test_plan_chunks_rejects_invalid_settings(length: float, overlap: float) -> None:
    with pytest.raises(ValueError, match="must be"):
        plan_chunks(100, length, overlap)


@requires_ffmpeg
def test_split_wav_offsets_and_overlap(long_3min_path: Path, tmp_path: Path) -> None:
    chunks = split_wav(long_3min_path, tmp_path, chunk_length_seconds=60, overlap_seconds=5)

    assert [(c.index, c.start_seconds, c.end_seconds) for c in chunks] == [
        (0, 0, 60),
        (1, 55, 115),
        (2, 110, 170),
        (3, 165, 180),
    ]
    source = _frames(long_3min_path)
    bytes_per_second = 16000 * 2
    for chunk in chunks:
        data = _frames(chunk.path)
        start = int(chunk.start_seconds * bytes_per_second)
        assert data == source[start : start + len(data)]  # sample-accurate slice
        assert len(data) == int(chunk.duration_seconds * bytes_per_second)
    # Consecutive chunks share exactly `overlap` seconds of audio.
    overlap_bytes = 5 * bytes_per_second
    assert _frames(chunks[0].path)[-overlap_bytes:] == _frames(chunks[1].path)[:overlap_bytes]
    assert chunks[0].path.name == "long_chunk_0000.wav"


@requires_ffmpeg
def test_split_wav_with_default_config_keeps_short_audio_whole(
    long_3min_path: Path, tmp_path: Path
) -> None:
    settings = Settings(_env_file=None)

    chunks = split_wav(
        long_3min_path,
        tmp_path,
        chunk_length_seconds=settings.chunk_length_seconds,
        overlap_seconds=settings.chunk_overlap_seconds,
    )

    assert len(chunks) == 1
    assert (chunks[0].start_seconds, chunks[0].end_seconds) == (0, 180)
