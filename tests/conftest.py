"""Shared pytest fixtures. Media fixtures are generated at test time, never committed."""

import io
import math
import shutil
import struct
import subprocess
import wave
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.main import create_app

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")


def make_wav(seconds: float = 0.5, rate: int = 16000) -> bytes:
    """A mono 16-bit sine-wave WAV."""
    frames = int(seconds * rate)
    samples = (int(8000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(frames))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", s) for s in samples))
    return buf.getvalue()


def _ffmpeg(out: Path, *args: str) -> bytes:
    subprocess.run(  # noqa: S603
        ["ffmpeg", "-v", "error", "-y", *args, str(out)],  # noqa: S607
        check=True,
        timeout=60,
    )
    return out.read_bytes()


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="session")
def wav_bytes() -> bytes:
    return make_wav()


@pytest.fixture(scope="session")
def silent_empty_wav_bytes() -> bytes:
    """Valid WAV header with zero audio frames."""
    return make_wav(seconds=0)


@pytest.fixture(scope="session")
def corrupted_wav_bytes() -> bytes:
    """RIFF/WAVE magic followed by garbage: passes magic check, fails ffprobe."""
    return b"RIFF" + struct.pack("<I", 4096) + b"WAVE" + b"\xde\xad\xbe\xef" * 1024


@pytest.fixture(scope="session")
def mp3_bytes(media_dir: Path) -> bytes:
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    return _ffmpeg(media_dir / "tone.mp3", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5")


@pytest.fixture(scope="session")
def video_without_audio_bytes(media_dir: Path) -> bytes:
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    return _ffmpeg(
        media_dir / "silent.mp4",
        *("-f", "lavfi", "-i", "color=c=black:s=32x32:d=0.5"),
        *("-c:v", "mpeg4", "-an"),
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        storage_dir=tmp_path / "storage",
        diarization_backend="mock",
        llm_provider="mock",
        asr_backend="mock",
        lid_backend="mock",
    )


@pytest.fixture
def client_factory() -> Callable[[Settings], AsyncIterator[AsyncClient]]:
    async def _make(settings: Settings) -> AsyncIterator[AsyncClient]:
        app = create_app(settings)
        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac,
        ):
            yield ac

    return _make


@pytest.fixture
async def client(
    settings: Settings, client_factory: Callable[[Settings], AsyncIterator[AsyncClient]]
) -> AsyncIterator[AsyncClient]:
    async for ac in client_factory(settings):
        yield ac


# --- Preprocessing fixtures (generated with ffmpeg lavfi sources) ---


def _lavfi(media_dir: Path, name: str, *args: str) -> Path:
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    out = media_dir / name
    if not out.exists():
        _ffmpeg(out, *args)
    return out


@pytest.fixture(scope="session")
def stereo_44k_path(media_dir: Path) -> Path:
    """3 s stereo 44.1 kHz tone."""
    return _lavfi(
        media_dir,
        "stereo44k.wav",
        "-f",
        "lavfi",
        "-i",
        "sine=f=440:d=3:sample_rate=44100",
        "-ac",
        "2",
    )


@pytest.fixture(scope="session")
def video_with_audio_path(media_dir: Path) -> Path:
    return _lavfi(
        media_dir,
        "talk.mp4",
        *("-f", "lavfi", "-i", "color=c=black:s=32x32:d=3"),
        *("-f", "lavfi", "-i", "sine=f=300:d=3"),
        *("-c:v", "mpeg4", "-c:a", "aac", "-shortest"),
    )


@pytest.fixture(scope="session")
def near_silent_path(media_dir: Path) -> Path:
    """Tone at about -70 dBFS: below the silence threshold and very quiet."""
    return _lavfi(media_dir, "quiet.wav", "-f", "lavfi", "-i", "sine=f=440:d=4,volume=0.0005")


@pytest.fixture(scope="session")
def clipped_path(media_dir: Path) -> Path:
    """Tone amplified far past full scale, so samples clip."""
    return _lavfi(media_dir, "clipped.wav", "-f", "lavfi", "-i", "sine=f=440:d=3,volume=20")


@pytest.fixture(scope="session")
def short_path(media_dir: Path) -> Path:
    return _lavfi(media_dir, "short.wav", "-f", "lavfi", "-i", "sine=f=440:d=1")


@pytest.fixture(scope="session")
def padded_path(media_dir: Path) -> Path:
    """1 s silence, 2 s tone, 1 s silence."""
    expr = "if(between(t,1,3),0.5*sin(2*PI*440*t),0)"
    return _lavfi(media_dir, "padded.wav", "-f", "lavfi", "-i", f"aevalsrc='{expr}':d=4:s=16000")


@pytest.fixture(scope="session")
def long_3min_path(media_dir: Path) -> Path:
    """3-minute 16 kHz mono tone for chunking tests."""
    return _lavfi(media_dir, "long.wav", "-f", "lavfi", "-i", "sine=f=440:d=180:sample_rate=16000")
