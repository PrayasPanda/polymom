"""Artifacts as files under a root directory (``STORAGE_DIR`` by default)."""

import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.concurrency import run_in_threadpool

from app.repositories.artifacts.base import CHUNK_BYTES, ArtifactStore, validate_key


class LocalArtifactStore(ArtifactStore):
    backend = "local"

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def path(self, key: str) -> Path:
        path = (self.root / validate_key(key)).resolve()
        if not path.is_relative_to(self.root):  # pragma: no cover - validate_key guards this
            raise ValueError(f"invalid artifact key: {key!r}")
        return path

    async def put(self, key: str, data: bytes, content_type: str | None = None) -> None:
        path = self.path(key)
        await run_in_threadpool(path.parent.mkdir, parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        await run_in_threadpool(tmp.write_bytes, data)
        await run_in_threadpool(tmp.replace, path)

    async def put_file(self, key: str, path: Path, content_type: str | None = None) -> None:
        dest = self.path(key)
        if path.resolve() == dest:  # noqa: ASYNC240 - no I/O for an absolute path
            return
        await run_in_threadpool(dest.parent.mkdir, parents=True, exist_ok=True)
        await run_in_threadpool(shutil.copyfile, path, dest)

    async def get(self, key: str) -> bytes:
        return await run_in_threadpool(self.path(key).read_bytes)

    async def stream(self, key: str) -> AsyncIterator[bytes]:
        path = self.path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        fh = await run_in_threadpool(path.open, "rb")
        try:
            while chunk := await run_in_threadpool(fh.read, CHUNK_BYTES):
                yield chunk
        finally:
            fh.close()

    async def exists(self, key: str) -> bool:
        return await run_in_threadpool(self.path(key).is_file)

    async def delete(self, key: str) -> None:
        await run_in_threadpool(self.path(key).unlink, missing_ok=True)

    async def delete_prefix(self, prefix: str) -> list[str]:
        base = self.path(prefix.rstrip("/"))

        def _delete() -> list[str]:
            if base.is_file():
                base.unlink()
                return [prefix.rstrip("/")]
            if not base.is_dir():
                return []
            keys = [p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file()]
            shutil.rmtree(base)
            return keys

        return await run_in_threadpool(_delete)

    async def presigned_url(self, key: str, expires_seconds: int = 3600) -> str | None:
        return None

    @asynccontextmanager
    async def local_path(self, key: str) -> AsyncIterator[Path]:
        path = self.path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        yield path
