"""Cut time ranges out of PCM WAV files (stdlib only, sample accurate)."""

import importlib
import wave
from pathlib import Path
from typing import Any


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def write_wav_slice(source: Path, dest: Path, start: float, end: float) -> float:
    """Write ``source[start:end]`` to ``dest`` with the same format.

    Returns the actual start time used (clamped to the file), which callers add
    back to timestamps produced for the slice.
    """
    with wave.open(str(source), "rb") as src:
        params = src.getparams()
        rate = params.framerate
        first = max(0, min(params.nframes, round(start * rate)))
        last = max(first, min(params.nframes, round(end * rate)))
        src.setpos(first)
        frames = src.readframes(last - first)
    with wave.open(str(dest), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(frames)
    return first / rate


def read_samples(path: Path, start: float, end: float) -> tuple[Any, int]:
    """Mono float32 numpy samples in ``[-1, 1]`` for a time range, and the sample rate.

    Requires numpy (``ml`` extra); used by model backends only.
    """
    numpy = importlib.import_module("numpy")
    with wave.open(str(path), "rb") as wav:
        rate, channels = wav.getframerate(), wav.getnchannels()
        first = max(0, round(start * rate))
        wav.setpos(min(first, wav.getnframes()))
        count = max(0, round(end * rate) - first)
        data = wav.readframes(count)
    samples = numpy.frombuffer(data, dtype="<i2").astype("float32") / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate
