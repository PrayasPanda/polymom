"""Artifact storage port: uploads, processed audio, large stage outputs, charts, exports.

Keys are ``/``-separated and relative, laid out as ``meetings/{meeting_id}/...``:

- ``meetings/{id}/upload/original.{ext}``            the uploaded recording
- ``meetings/{id}/{run_id}/processed.wav``           normalized audio of a run
- ``meetings/{id}/{run_id}/stages/{stage}.json``     stage outputs too large for the DB
- ``meetings/{id}/{run_id}/charts/{name}.png``       cached charts
- ``meetings/{id}/{run_id}/exports/minutes.{fmt}``   cached exports
"""

import re
import tempfile
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

from starlette.concurrency import run_in_threadpool

_KEY = re.compile(r"^[A-Za-z0-9._\-/]+$")
CHUNK_BYTES = 1024 * 1024


def validate_key(key: str) -> str:
    """Reject absolute keys, ``..`` and odd characters so keys can't escape the store."""
    parts = PurePosixPath(key).parts
    if not key or not _KEY.match(key) or key.startswith("/") or ".." in parts:
        raise ValueError(f"invalid artifact key: {key!r}")
    return key


def meeting_prefix(meeting_id: uuid.UUID) -> str:
    return f"meetings/{meeting_id}/"


def run_key(meeting_id: uuid.UUID, run_id: uuid.UUID, name: str) -> str:
    return f"meetings/{meeting_id}/{run_id}/{name}"


class ArtifactStore(ABC):
    backend: str

    @abstractmethod
    async def put(self, key: str, data: bytes, content_type: str | None = None) -> None: ...

    @abstractmethod
    async def put_file(self, key: str, path: Path, content_type: str | None = None) -> None:
        """Upload a local file (streamed, not read into memory)."""

    @abstractmethod
    async def get(self, key: str) -> bytes:
        """Raise :class:`FileNotFoundError` when missing."""

    @abstractmethod
    def stream(self, key: str) -> AsyncIterator[bytes]:
        """Yield the artifact in chunks."""

    @abstractmethod
    async def exists(self, key: str) -> bool: ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete one key; missing keys are ignored."""

    @abstractmethod
    async def delete_prefix(self, prefix: str) -> list[str]:
        """Delete every key under ``prefix``; returns the deleted keys."""

    @abstractmethod
    async def presigned_url(self, key: str, expires_seconds: int = 3600) -> str | None:
        """A time-limited direct URL, or ``None`` when the backend can't make one."""

    @asynccontextmanager
    async def local_path(self, key: str) -> AsyncIterator[Path]:
        """A local file with the artifact's content (ffmpeg and models need paths).

        The default downloads to a temporary file removed afterwards;
        the local store yields its own file directly.
        """
        suffix = PurePosixPath(key).suffix
        with tempfile.TemporaryDirectory(prefix="polymom-") as tmp:
            path = Path(tmp) / f"artifact{suffix}"
            with path.open("wb") as fh:
                async for chunk in self.stream(key):
                    await run_in_threadpool(fh.write, chunk)
            yield path
