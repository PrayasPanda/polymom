"""Async ffmpeg invocation with a hard timeout."""

import asyncio
import contextlib
from collections.abc import Sequence

from app.core.exceptions import AudioProcessingError, FFmpegTimeoutError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Keep only the tail of stderr in errors/logs; ffmpeg can be very chatty.
STDERR_TAIL_CHARS = 2000


async def run_ffmpeg(binary: str, args: Sequence[str], *, timeout_seconds: float) -> str:
    """Run ``binary args...`` and return its decoded stderr (where ffmpeg reports).

    Raises:
        FFmpegTimeoutError: the process exceeded ``timeout_seconds`` and was killed.
        AudioProcessingError: the binary is missing or exited non-zero.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise AudioProcessingError(
            "Audio processing is unavailable (ffmpeg not found).",
            details={"reason": "ffmpeg_not_found"},
        ) from exc

    try:
        _, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except TimeoutError as exc:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        raise FFmpegTimeoutError(
            f"Audio processing timed out after {timeout_seconds:g} seconds.",
            details={"timeout_seconds": timeout_seconds},
        ) from exc
    except BaseException:
        # Cancellation: never leave an orphaned ffmpeg behind.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        raise

    stderr = stderr_bytes.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        logger.warning(
            "ffmpeg_failed", returncode=proc.returncode, stderr=stderr[-STDERR_TAIL_CHARS:]
        )
        raise AudioProcessingError(
            "ffmpeg could not process the audio.",
            details={"reason": "ffmpeg_error", "returncode": proc.returncode},
        )
    return stderr
