import subprocess
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.exceptions import (
    CorruptedMediaError,
    EmptyFileError,
    FileTooLargeError,
    MediaProbeUnavailableError,
    UnsupportedFileTypeError,
)
from app.services.audio.validator import MediaValidator, sanitize_filename
from tests.conftest import make_wav, requires_ffmpeg


class BytesSource:
    """Minimal async reader standing in for ``UploadFile``."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0
        self.max_read = 0

    async def read(self, size: int = -1) -> bytes:
        end = len(self._data) if size < 0 else self._pos + size
        chunk = self._data[self._pos : end]
        self._pos += len(chunk)
        self.max_read = max(self.max_read, len(chunk))
        return chunk


@pytest.fixture
def validator(settings: Settings) -> MediaValidator:
    return MediaValidator(settings)


async def _validate(validator: MediaValidator, data: bytes, name: str, dest: Path) -> object:
    return await validator.validate(BytesSource(data), name, dest, "m1")


@requires_ffmpeg
async def test_valid_wav_is_stored_with_metadata(
    validator: MediaValidator, wav_bytes: bytes, tmp_path: Path
) -> None:
    media = await validator.validate(BytesSource(wav_bytes), "../../My Talk.WAV", tmp_path, "m1")

    assert media.path == tmp_path / "m1.wav"
    assert media.path.read_bytes() == wav_bytes
    assert media.original_filename == "My Talk.WAV"
    assert media.mime_type == "audio/x-wav"
    assert media.size_bytes == len(wav_bytes)
    assert media.metadata.codec == "pcm_s16le"
    assert media.metadata.sample_rate == 16000
    assert media.metadata.channels == 1
    assert media.metadata.duration_seconds == pytest.approx(0.5, abs=0.01)
    assert list(tmp_path.iterdir()) == [media.path]


@requires_ffmpeg
async def test_valid_mp3(validator: MediaValidator, mp3_bytes: bytes, tmp_path: Path) -> None:
    media = await validator.validate(BytesSource(mp3_bytes), "tone.mp3", tmp_path, "m1")

    assert media.mime_type == "audio/mpeg"
    assert media.metadata.codec == "mp3"


async def test_disallowed_extension_rejected_before_reading(
    validator: MediaValidator, tmp_path: Path
) -> None:
    with pytest.raises(UnsupportedFileTypeError) as exc:
        await _validate(validator, b"MZ\x90\x00", "virus.exe", tmp_path)

    assert exc.value.details is not None
    assert exc.value.details["extension"] == "exe"
    assert not any(tmp_path.iterdir())


async def test_missing_extension_rejected(validator: MediaValidator, tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFileTypeError, match="no extension"):
        await _validate(validator, b"data", "recording", tmp_path)


async def test_fake_extension_rejected(validator: MediaValidator, tmp_path: Path) -> None:
    with pytest.raises(UnsupportedFileTypeError, match="does not match"):
        await _validate(validator, b"just some text, not audio" * 20, "notes.wav", tmp_path)
    assert not any(tmp_path.iterdir())


@requires_ffmpeg
async def test_real_type_mismatch_rejected(
    validator: MediaValidator, mp3_bytes: bytes, tmp_path: Path
) -> None:
    with pytest.raises(UnsupportedFileTypeError) as exc:
        await _validate(validator, mp3_bytes, "tone.wav", tmp_path)

    assert exc.value.details == {"extension": "wav", "detected_mime_type": "audio/mpeg"}


async def test_oversized_file_rejected_while_streaming(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None, storage_dir=tmp_path, max_upload_mb=1, upload_chunk_bytes=64 * 1024
    )
    source = BytesSource(make_wav(seconds=40))  # ~1.3 MB

    with pytest.raises(FileTooLargeError):
        await MediaValidator(settings).validate(source, "big.wav", tmp_path / "up", "m1")

    assert source.max_read <= 64 * 1024
    assert not any((tmp_path / "up").iterdir())


async def test_empty_file_rejected(validator: MediaValidator, tmp_path: Path) -> None:
    with pytest.raises(EmptyFileError):
        await _validate(validator, b"", "empty.wav", tmp_path)
    assert not any(tmp_path.iterdir())


@requires_ffmpeg
async def test_corrupted_file_rejected(
    validator: MediaValidator, corrupted_wav_bytes: bytes, tmp_path: Path
) -> None:
    with pytest.raises(CorruptedMediaError):
        await _validate(validator, corrupted_wav_bytes, "broken.wav", tmp_path)
    assert not any(tmp_path.iterdir())


@requires_ffmpeg
async def test_zero_duration_audio_rejected(
    validator: MediaValidator, silent_empty_wav_bytes: bytes, tmp_path: Path
) -> None:
    with pytest.raises(CorruptedMediaError):
        await _validate(validator, silent_empty_wav_bytes, "blank.wav", tmp_path)


@requires_ffmpeg
async def test_video_without_audio_rejected(
    validator: MediaValidator, video_without_audio_bytes: bytes, tmp_path: Path
) -> None:
    with pytest.raises(CorruptedMediaError) as exc:
        await _validate(validator, video_without_audio_bytes, "screen.mp4", tmp_path)

    assert exc.value.details == {"reason": "no_audio_stream"}
    assert not any(tmp_path.iterdir())


async def test_missing_ffprobe_is_server_error(wav_bytes: bytes, tmp_path: Path) -> None:
    settings = Settings(_env_file=None, ffprobe_path="definitely-not-ffprobe-xyz")

    with pytest.raises(MediaProbeUnavailableError):
        await MediaValidator(settings).validate(BytesSource(wav_bytes), "a.wav", tmp_path, "m1")
    assert not any(tmp_path.iterdir())


async def test_probe_timeout_is_corrupted(
    validator: MediaValidator, wav_bytes: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=1)

    monkeypatch.setattr(subprocess, "run", _timeout)

    with pytest.raises(CorruptedMediaError) as exc:
        await _validate(validator, wav_bytes, "a.wav", tmp_path)
    assert exc.value.details == {"reason": "probe_timeout"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("meeting.wav", "meeting.wav"),
        ("../../etc/passwd.wav", "passwd.wav"),
        ("C:\\Users\\me\\call.mp3", "call.mp3"),
        ("a<b>|c?.wav", "a_b_c_.wav"),
        ("..hidden.wav", "hidden.wav"),
        ("बैठक ଆଲୋଚନା.m4a", "बैठक ଆଲୋଚନା.m4a"),
        ("", "upload"),
        (None, "upload"),
        ("...", "upload"),
    ],
)
def test_sanitize_filename(raw: str | None, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_caps_length_and_keeps_extension() -> None:
    result = sanitize_filename("x" * 400 + ".flac")

    assert len(result) == 255
    assert result.endswith(".flac")
