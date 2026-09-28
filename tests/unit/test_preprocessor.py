import hashlib
import sys
import uuid
import wave
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.exceptions import AudioProcessingError, FFmpegTimeoutError
from app.schemas.audio import AudioWarningCode, PreprocessResult
from app.services.audio import preprocessor as preprocessor_module
from app.services.audio.ffmpeg import run_ffmpeg
from app.services.audio.preprocessor import AudioPreprocessor, build_filter_chain
from tests.conftest import requires_ffmpeg


def _codes(result: PreprocessResult) -> set[AudioWarningCode]:
    return {w.code for w in result.warnings}


def _wav_format(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as wav:
        return wav.getnchannels(), wav.getframerate(), wav.getsampwidth()


async def _process(settings: Settings, path: Path) -> PreprocessResult:
    return await AudioPreprocessor(settings).process(uuid.uuid4(), path)


@requires_ffmpeg
async def test_output_is_mono_16k_pcm_and_original_untouched(
    settings: Settings, stereo_44k_path: Path
) -> None:
    before = hashlib.sha256(stereo_44k_path.read_bytes()).hexdigest()
    meeting_id = uuid.uuid4()

    result = await AudioPreprocessor(settings).process(meeting_id, stereo_44k_path)

    assert result.processed_path == settings.processed_dir / f"{meeting_id}.wav"
    assert _wav_format(result.processed_path) == (1, 16000, 2)  # mono, 16 kHz, 16-bit
    assert (result.sample_rate, result.channels) == (16000, 1)
    assert result.duration_seconds == pytest.approx(3.0, abs=0.05)
    assert result.loudness_lufs is not None
    assert result.warnings == []
    assert result.processing_time_ms >= 0
    assert hashlib.sha256(stereo_44k_path.read_bytes()).hexdigest() == before
    assert [p.name for p in settings.processed_dir.iterdir()] == [f"{meeting_id}.wav"]


@requires_ffmpeg
async def test_extracts_audio_from_video(settings: Settings, video_with_audio_path: Path) -> None:
    result = await _process(settings, video_with_audio_path)

    assert _wav_format(result.processed_path) == (1, 16000, 2)
    assert result.duration_seconds == pytest.approx(3.0, abs=0.1)


@requires_ffmpeg
async def test_custom_sample_rate_and_denoise(tmp_path: Path, stereo_44k_path: Path) -> None:
    settings = Settings(
        _env_file=None, storage_dir=tmp_path, target_sample_rate=8000, enable_denoise=True
    )

    result = await _process(settings, stereo_44k_path)

    assert _wav_format(result.processed_path) == (1, 8000, 2)


@requires_ffmpeg
async def test_near_silent_file_warns_low_volume_and_mostly_silent(
    settings: Settings, near_silent_path: Path
) -> None:
    result = await _process(settings, near_silent_path)

    assert _codes(result) == {AudioWarningCode.LOW_VOLUME, AudioWarningCode.MOSTLY_SILENT}
    assert result.silence_ratio > 0.8
    assert result.processed_path.exists()  # warnings never fail processing


@requires_ffmpeg
async def test_clipped_file_warns_clipping(settings: Settings, clipped_path: Path) -> None:
    result = await _process(settings, clipped_path)

    assert _codes(result) == {AudioWarningCode.CLIPPING}
    assert result.peak_db is not None
    assert result.peak_db >= -0.1


@requires_ffmpeg
async def test_short_file_warns_too_short(settings: Settings, short_path: Path) -> None:
    result = await _process(settings, short_path)

    assert _codes(result) == {AudioWarningCode.TOO_SHORT}


@requires_ffmpeg
async def test_silence_is_reported_but_not_trimmed_by_default(
    settings: Settings, padded_path: Path
) -> None:
    result = await _process(settings, padded_path)

    assert result.leading_silence_seconds == pytest.approx(1.0, abs=0.05)
    assert result.trailing_silence_seconds == pytest.approx(1.0, abs=0.05)
    assert result.silence_ratio == pytest.approx(0.5, abs=0.05)
    assert result.trimmed_seconds == 0
    assert result.duration_seconds == pytest.approx(4.0, abs=0.05)


@requires_ffmpeg
async def test_silence_trimmed_when_enabled(tmp_path: Path, padded_path: Path) -> None:
    settings = Settings(_env_file=None, storage_dir=tmp_path, trim_silence=True)

    result = await _process(settings, padded_path)

    assert result.duration_seconds == pytest.approx(2.0, abs=0.1)
    assert result.trimmed_seconds == pytest.approx(2.0, abs=0.1)


@requires_ffmpeg
async def test_partial_output_removed_on_failure(
    settings: Settings, stereo_44k_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _write_then_time_out(binary: str, args: list[str], *, timeout_seconds: float) -> str:
        Path(args[-1]).write_bytes(b"partial")
        raise FFmpegTimeoutError("Audio processing timed out after 1 seconds.")

    monkeypatch.setattr(preprocessor_module, "run_ffmpeg", _write_then_time_out)

    with pytest.raises(FFmpegTimeoutError):
        await _process(settings, stereo_44k_path)
    assert list(settings.processed_dir.iterdir()) == []


@requires_ffmpeg
async def test_invalid_output_is_processing_error(
    settings: Settings, stereo_44k_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _write_garbage(binary: str, args: list[str], *, timeout_seconds: float) -> str:
        Path(args[-1]).write_bytes(b"not a wav file")
        return ""

    monkeypatch.setattr(preprocessor_module, "run_ffmpeg", _write_garbage)

    with pytest.raises(AudioProcessingError) as exc:
        await _process(settings, stereo_44k_path)
    assert exc.value.details == {"reason": "invalid_output"}
    assert list(settings.processed_dir.iterdir()) == []


def test_filter_chain_respects_flags() -> None:
    base = Settings(_env_file=None, enable_highpass=False, enable_denoise=False)
    full = Settings(
        _env_file=None, enable_highpass=True, highpass_cutoff_hz=100, enable_denoise=True
    )

    assert build_filter_chain(base) == "loudnorm=I=-23.0:TP=-2:LRA=11,aresample=16000"
    chain = build_filter_chain(full, trim=(1.0, 3.5)).split(",")
    assert chain[:4] == [
        "atrim=start=1.000:end=3.500",
        "asetpts=PTS-STARTPTS",
        "highpass=f=100",
        "afftdn=nf=-25",
    ]
    assert chain[-2].startswith("loudnorm=")


async def test_run_ffmpeg_timeout_kills_process() -> None:
    with pytest.raises(FFmpegTimeoutError) as exc:
        await run_ffmpeg(sys.executable, ["-c", "import time; time.sleep(30)"], timeout_seconds=0.5)
    assert exc.value.status_code == 504
    assert exc.value.details == {"timeout_seconds": 0.5}


async def test_run_ffmpeg_nonzero_exit() -> None:
    with pytest.raises(AudioProcessingError) as exc:
        await run_ffmpeg(sys.executable, ["-c", "raise SystemExit(3)"], timeout_seconds=10)
    assert exc.value.details == {"reason": "ffmpeg_error", "returncode": 3}


async def test_run_ffmpeg_missing_binary() -> None:
    with pytest.raises(AudioProcessingError) as exc:
        await run_ffmpeg("definitely-not-ffmpeg-xyz", ["-version"], timeout_seconds=5)
    assert exc.value.details == {"reason": "ffmpeg_not_found"}
