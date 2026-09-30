"""Live progress tracking backed by Redis.

The worker writes a JSON snapshot to ``polymom:progress:{meeting_id}`` after every
stage and chunk, and publishes the same snapshot on ``polymom:progress:channel:{id}``.
``GET /status`` reads the key; ``GET /status/stream`` (SSE) subscribes to the channel.
When Redis is not configured (``PIPELINE_EXECUTION=inline``) a no-op reporter is used
and status is derived from the database instead.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, Field

from app.schemas.meeting import MeetingStatus

if TYPE_CHECKING:
    from redis.asyncio import Redis

# Ordered pipeline stages with a rough share of total time, for the overall percent
# and ETA. Diarization and ASR dominate; the weights only need to be proportional.
STAGE_WEIGHTS: dict[str, float] = {
    "preprocess": 0.05,
    "diarize": 0.30,
    "identify_languages": 0.10,
    "transcribe": 0.35,
    "align": 0.05,
    "analytics": 0.05,
    "summarize": 0.10,
}
PROGRESS_TTL_SECONDS = 24 * 3600


def _key(meeting_id: uuid.UUID) -> str:
    return f"polymom:progress:{meeting_id}"


def channel(meeting_id: uuid.UUID) -> str:
    return f"polymom:progress:channel:{meeting_id}"


class StageProgress(BaseModel):
    name: str
    status: str = "pending"  # pending | running | completed | failed | skipped
    chunk: int | None = None
    total_chunks: int | None = None
    duration_ms: int | None = None


class Progress(BaseModel):
    meeting_id: uuid.UUID
    run_id: uuid.UUID | None = None
    status: MeetingStatus
    current_stage: str | None = None
    percent: float = 0.0
    stages: list[StageProgress] = Field(default_factory=list)
    audio_seconds: float | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    error: str | None = None
    updated_at: float = Field(default_factory=time.time)


class ProgressReporter(Protocol):
    async def start(
        self, run_id: uuid.UUID, stages: list[str], audio_seconds: float | None
    ) -> None: ...
    async def stage(self, name: str, status: str) -> None: ...
    async def chunk(self, name: str, done: int, total: int) -> None: ...
    async def finish(self, status: MeetingStatus, error: str | None) -> None: ...


class NullProgressReporter:
    """Used in inline mode; status comes from the database."""

    async def start(
        self, run_id: uuid.UUID, stages: list[str], audio_seconds: float | None
    ) -> None:
        return None

    async def stage(self, name: str, status: str) -> None:
        return None

    async def chunk(self, name: str, done: int, total: int) -> None:
        return None

    async def finish(self, status: MeetingStatus, error: str | None) -> None:
        return None


class RedisProgressReporter:
    """Writes and publishes :class:`Progress` snapshots as the run advances."""

    def __init__(self, redis: Redis, meeting_id: uuid.UUID) -> None:
        self._redis = redis
        self._started = time.perf_counter()
        self._progress = Progress(meeting_id=meeting_id, status=MeetingStatus.PROCESSING)

    async def start(
        self, run_id: uuid.UUID, stages: list[str], audio_seconds: float | None
    ) -> None:
        self._progress.run_id = run_id
        self._progress.audio_seconds = audio_seconds
        self._progress.stages = [StageProgress(name=s) for s in stages]
        await self._publish()

    async def stage(self, name: str, status: str) -> None:
        for sp in self._progress.stages:
            if sp.name == name:
                sp.status = status
                sp.chunk = None
        self._progress.current_stage = name if status == "running" else self._progress.current_stage
        self._recompute()
        await self._publish()

    async def chunk(self, name: str, done: int, total: int) -> None:
        for sp in self._progress.stages:
            if sp.name == name:
                sp.chunk, sp.total_chunks = done, total
        self._recompute()
        await self._publish()

    async def finish(self, status: MeetingStatus, error: str | None) -> None:
        self._progress.status = status
        self._progress.error = error
        self._progress.percent = 100.0 if status != MeetingStatus.FAILED else self._progress.percent
        self._progress.current_stage = None
        self._progress.eta_seconds = 0.0
        await self._publish()

    def _recompute(self) -> None:
        done = 0.0
        for sp in self._progress.stages:
            weight = STAGE_WEIGHTS.get(sp.name, 0.05)
            if sp.status in ("completed", "skipped"):
                done += weight
            elif sp.status == "running" and sp.total_chunks:
                done += weight * (sp.chunk or 0) / sp.total_chunks
        total_weight = sum(STAGE_WEIGHTS.get(sp.name, 0.05) for sp in self._progress.stages) or 1.0
        fraction = min(done / total_weight, 0.99)
        self._progress.percent = round(fraction * 100, 1)
        elapsed = time.perf_counter() - self._started
        self._progress.elapsed_seconds = round(elapsed, 1)
        self._progress.eta_seconds = (
            round(elapsed / fraction - elapsed, 1) if fraction > 0.02 else None
        )

    async def _publish(self) -> None:
        self._progress.updated_at = time.time()
        payload = self._progress.model_dump_json()
        key = _key(self._progress.meeting_id)
        await self._redis.set(key, payload, ex=PROGRESS_TTL_SECONDS)
        await self._redis.publish(channel(self._progress.meeting_id), payload)


async def read_progress(redis: Redis, meeting_id: uuid.UUID) -> Progress | None:
    raw = await redis.get(_key(meeting_id))
    return Progress.model_validate_json(raw) if raw else None
