"""Silence and level analysis of a recording in a single ffmpeg pass.

Filters: ``silencedetect`` (silent spans), ``astats`` (RMS/peak) and
``ebur128`` (integrated loudness). Results are parsed from ffmpeg's log output.
"""

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import Settings
from app.schemas.audio import AudioWarning, AudioWarningCode
from app.services.audio.ffmpeg import run_ffmpeg

LOW_VOLUME_RMS_DB = -40.0
CLIPPING_PEAK_DB = -0.1
MIN_DURATION_SECONDS = 2.0
MAX_SILENCE_RATIO = 0.8
# Silence closer than this to either end counts as leading/trailing silence.
EDGE_TOLERANCE_SECONDS = 0.05

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
_TIME_RE = re.compile(r"time=(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)")
_LUFS_RE = re.compile(r"Integrated loudness:\s*I:\s*(-?[\d.]+|-?inf|nan)\s*LUFS", re.S)
_PEAK_RE = re.compile(r"Peak level dB:\s*(-?[\d.]+|-?inf|nan)")
_RMS_RE = re.compile(r"RMS level dB:\s*(-?[\d.]+|-?inf|nan)")


@dataclass(frozen=True, slots=True)
class AudioAnalysis:
    duration_seconds: float
    loudness_lufs: float | None
    rms_db: float | None
    peak_db: float | None
    silences: list[tuple[float, float]] = field(default_factory=list)

    @property
    def silence_seconds(self) -> float:
        return sum(end - start for start, end in self.silences)

    @property
    def silence_ratio(self) -> float:
        if self.duration_seconds <= 0:
            return 1.0
        return min(1.0, self.silence_seconds / self.duration_seconds)

    @property
    def leading_silence_seconds(self) -> float:
        if self.silences and self.silences[0][0] <= EDGE_TOLERANCE_SECONDS:
            return self.silences[0][1]
        return 0.0

    @property
    def trailing_silence_seconds(self) -> float:
        if self.silences and self.silences[-1][1] >= self.duration_seconds - EDGE_TOLERANCE_SECONDS:
            # A span covering the whole file is already counted as leading silence.
            if self.silences[-1][0] <= EDGE_TOLERANCE_SECONDS:
                return 0.0
            return self.duration_seconds - self.silences[-1][0]
        return 0.0


def quality_warnings(analysis: AudioAnalysis) -> list[AudioWarning]:
    """Non-fatal warnings about recording quality."""
    warnings: list[AudioWarning] = []
    if analysis.rms_db is None or analysis.rms_db < LOW_VOLUME_RMS_DB:
        level = "unmeasurable" if analysis.rms_db is None else f"{analysis.rms_db:.1f} dBFS"
        warnings.append(
            AudioWarning(
                code=AudioWarningCode.LOW_VOLUME,
                message=f"Very low volume (RMS {level}); transcription may be unreliable.",
            )
        )
    if analysis.peak_db is not None and analysis.peak_db >= CLIPPING_PEAK_DB:
        warnings.append(
            AudioWarning(
                code=AudioWarningCode.CLIPPING,
                message=f"Audio is clipping (peak {analysis.peak_db:.1f} dBFS).",
            )
        )
    if analysis.duration_seconds < MIN_DURATION_SECONDS:
        warnings.append(
            AudioWarning(
                code=AudioWarningCode.TOO_SHORT,
                message=f"Audio is only {analysis.duration_seconds:.1f}s long.",
            )
        )
    if analysis.silence_ratio > MAX_SILENCE_RATIO:
        warnings.append(
            AudioWarning(
                code=AudioWarningCode.MOSTLY_SILENT,
                message=f"{analysis.silence_ratio:.0%} of the audio is silence.",
            )
        )
    return warnings


def _level(match: re.Match[str] | None) -> float | None:
    if match is None:
        return None
    value = float(match.group(1))
    return round(value, 2) if math.isfinite(value) else None


def _hms(match: re.Match[str]) -> float:
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_analysis(stderr: str) -> AudioAnalysis:
    """Parse ffmpeg's log from :func:`analyze` into an :class:`AudioAnalysis`."""
    duration = 0.0
    if match := _DURATION_RE.search(stderr):
        duration = _hms(match)
    elif times := list(_TIME_RE.finditer(stderr)):
        duration = _hms(times[-1])

    starts = [float(m.group(1)) for m in _SILENCE_START_RE.finditer(stderr)]
    ends = [float(m.group(1)) for m in _SILENCE_END_RE.finditer(stderr)]
    silences: list[tuple[float, float]] = []
    for i, start in enumerate(starts):
        # Older ffmpeg omits the final silence_end when silence runs to EOF.
        end = ends[i] if i < len(ends) else duration
        silences.append((max(0.0, start), min(end, duration) if duration else end))

    # astats prints per-channel blocks before "Overall"; take the overall values.
    overall = stderr.rsplit("Overall", 1)[-1]
    return AudioAnalysis(
        duration_seconds=round(duration, 3),
        loudness_lufs=_level(_LUFS_RE.search(stderr)),
        rms_db=_level(_RMS_RE.search(overall)),
        peak_db=_level(_PEAK_RE.search(overall)),
        silences=silences,
    )


async def analyze(path: Path, settings: Settings) -> AudioAnalysis:
    """Measure silence, RMS/peak level and loudness of the first audio stream."""
    filters = ",".join(
        [
            f"silencedetect=noise={settings.silence_threshold_db}dB"
            f":d={settings.silence_min_duration_seconds}",
            "astats=measure_perchannel=none:measure_overall=RMS_level+Peak_level",
            "ebur128=framelog=quiet",
        ]
    )
    stderr = await run_ffmpeg(
        settings.ffmpeg_path,
        [
            *("-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0"),
            *("-af", filters, "-f", "null", "-"),
        ],
        timeout_seconds=settings.ffmpeg_timeout_seconds,
    )
    return parse_analysis(stderr)
