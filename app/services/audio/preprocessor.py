"""Audio preprocessing: extract, (trim), filter, normalize loudness, resample.

Output is mono 16-bit PCM WAV at ``TARGET_SAMPLE_RATE`` (16 kHz by default),
the input format Whisper and pyannote expect. The original upload is never modified.
"""

import time
import uuid
import wave
from pathlib import Path

from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.exceptions import AudioProcessingError
from app.core.logging import get_logger
from app.schemas.audio import PreprocessResult
from app.services.audio.analysis import AudioAnalysis, analyze, quality_warnings
from app.services.audio.ffmpeg import run_ffmpeg

logger = get_logger(__name__)


def build_filter_chain(settings: Settings, trim: tuple[float, float] | None = None) -> str:
    """ffmpeg ``-af`` chain. ``trim`` is ``(start, end)`` in input seconds."""
    filters: list[str] = []
    if trim is not None:
        start, end = trim
        filters += [f"atrim=start={start:.3f}:end={end:.3f}", "asetpts=PTS-STARTPTS"]
    if settings.enable_highpass:
        filters.append(f"highpass=f={settings.highpass_cutoff_hz}")
    if settings.enable_denoise:
        filters.append("afftdn=nf=-25")
    # Single-pass EBU R128 normalization; loudnorm upsamples internally, so resample after.
    filters.append(f"loudnorm=I={settings.target_loudness_lufs}:TP=-2:LRA=11")
    filters.append(f"aresample={settings.target_sample_rate}")
    return ",".join(filters)


def trim_window(analysis: AudioAnalysis) -> tuple[float, float] | None:
    """Span to keep after removing leading/trailing silence, if any is removable."""
    start = analysis.leading_silence_seconds
    end = analysis.duration_seconds - analysis.trailing_silence_seconds
    if (start <= 0 and end >= analysis.duration_seconds) or end - start <= 0:
        return None
    return start, end


def _read_wav_format(path: Path) -> tuple[int, int, float]:
    with wave.open(str(path), "rb") as wav:
        rate, channels, frames = wav.getframerate(), wav.getnchannels(), wav.getnframes()
    return rate, channels, frames / rate if rate else 0.0


class AudioPreprocessor:
    """Turns a validated upload into a normalized, model-ready WAV."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def output_path(self, meeting_id: uuid.UUID) -> Path:
        return self._settings.processed_dir / f"{meeting_id}.wav"

    async def process(self, meeting_id: uuid.UUID, input_path: Path) -> PreprocessResult:
        """Analyze ``input_path`` and write ``processed/{meeting_id}.wav``.

        Raises:
            AudioProcessingError / FFmpegTimeoutError: partial output is removed.
        """
        settings = self._settings
        started = time.perf_counter()
        final = self.output_path(meeting_id)
        partial = final.with_suffix(".wav.part")
        await run_in_threadpool(final.parent.mkdir, parents=True, exist_ok=True)

        analysis = await analyze(input_path, settings)
        trim = trim_window(analysis) if settings.trim_silence else None

        args = ["-hide_banner", "-nostats", "-y", "-i", str(input_path)]
        args += ["-map", "0:a:0", "-vn", "-sn", "-dn"]
        args += ["-af", build_filter_chain(settings, trim)]
        args += ["-ac", "1", "-ar", str(settings.target_sample_rate), "-c:a", "pcm_s16le"]
        args += ["-f", "wav", str(partial)]
        try:
            await run_ffmpeg(
                settings.ffmpeg_path, args, timeout_seconds=settings.ffmpeg_timeout_seconds
            )
            rate, channels, duration = await run_in_threadpool(_read_wav_format, partial)
            if duration <= 0:
                raise AudioProcessingError(
                    "Preprocessing produced no audio.", details={"reason": "empty_output"}
                )
            await run_in_threadpool(partial.replace, final)
        except (wave.Error, EOFError) as exc:
            partial.unlink(missing_ok=True)
            raise AudioProcessingError(
                "Preprocessing produced an invalid WAV file.", details={"reason": "invalid_output"}
            ) from exc
        except BaseException:
            partial.unlink(missing_ok=True)
            final.unlink(missing_ok=True)
            raise

        trimmed = round(analysis.duration_seconds - (trim[1] - trim[0]), 3) if trim else 0.0
        result = PreprocessResult(
            processed_path=final,
            sample_rate=rate,
            channels=channels,
            duration_seconds=round(duration, 3),
            loudness_lufs=analysis.loudness_lufs,
            rms_db=analysis.rms_db,
            peak_db=analysis.peak_db,
            silence_ratio=round(analysis.silence_ratio, 4),
            leading_silence_seconds=round(analysis.leading_silence_seconds, 3),
            trailing_silence_seconds=round(analysis.trailing_silence_seconds, 3),
            trimmed_seconds=trimmed,
            warnings=quality_warnings(analysis),
            processing_time_ms=int((time.perf_counter() - started) * 1000),
        )
        logger.info(
            "audio_preprocessed",
            meeting_id=str(meeting_id),
            duration_seconds=result.duration_seconds,
            loudness_lufs=result.loudness_lufs,
            silence_ratio=result.silence_ratio,
            warnings=[w.code.value for w in result.warnings],
            processing_time_ms=result.processing_time_ms,
        )
        return result
