"""Chunk-level checkpoints and cooperative stop points for long stages.

Diarization and ASR split long audio into chunks. After each chunk its raw result
is written to the artifact store under the run and the stage fingerprint, so a
crash, a cancel or a worker shutdown at chunk 7 of 10 resumes at chunk 7 — the
first six are read back instead of recomputed. A changed stage config changes
the fingerprint, so stale checkpoints are never reused.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from app.repositories.artifacts import ArtifactStore, run_key

if TYPE_CHECKING:
    from app.workers.progress import ProgressReporter

StopReason = Literal["cancelled", "shutdown"]
StopCheck = Callable[[], Awaitable[StopReason | None]]


class StopRequested(Exception):  # noqa: N818 - control flow, not an error
    """Raised at a safe point (between chunks or stages) to stop cooperatively."""

    def __init__(self, reason: StopReason) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class ChunkHooks:
    store: ArtifactStore
    meeting_id: uuid.UUID
    run_id: uuid.UUID
    stage: str
    fingerprint: str
    reporter: ProgressReporter | None = None
    should_stop: StopCheck | None = None

    def _key(self, index: int) -> str:
        return run_key(
            self.meeting_id,
            self.run_id,
            f"checkpoints/{self.stage}/{self.fingerprint[:16]}/chunk-{index:04d}.json",
        )

    async def load(self, index: int) -> Any | None:
        """The saved result of chunk ``index``, or ``None`` if it must be computed."""
        key = self._key(index)
        if not await self.store.exists(key):
            return None
        return json.loads(await self.store.get(key))

    async def save(self, index: int, data: Any) -> None:
        await self.store.put(
            self._key(index), json.dumps(data, ensure_ascii=False).encode(), "application/json"
        )

    async def after_chunk(self, index: int, total: int) -> None:
        """Report progress, then stop here if a cancel or shutdown was requested."""
        if self.reporter is not None:
            await self.reporter.chunk(self.stage, index + 1, total)
        if self.should_stop is not None and (reason := await self.should_stop()):
            raise StopRequested(reason)
