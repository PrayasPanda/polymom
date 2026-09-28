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
    return Settings(_env_file=None, app_env="test", storage_dir=tmp_path / "storage")


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
