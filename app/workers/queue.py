"""Queue abstraction with two backends: arq (Redis) and inline (in-process, for tests).

The API only enqueues jobs. Workers ({\"cpu\", \"gpu\", \"llm\"}) run them; each queue
maps to one arq worker process with its own concurrency (``QUEUE_CONCURRENCY_*``).
"""

from __future__ import annotations

import uuid
from typing import Any, Literal, Protocol

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

QueueName = Literal["cpu", "gpu", "llm"]
QUEUE_NAMES: tuple[QueueName, ...] = ("cpu", "gpu", "llm")


class JobQueue(Protocol):
    async def enqueue_meeting(
        self,
        meeting_id: uuid.UUID,
        *,
        from_stage: str | None = None,
        request_id: str | None = None,
        callback_url: str | None = None,
    ) -> str: ...

    async def enqueue_regenerate_summary(
        self,
        meeting_id: uuid.UUID,
        *,
        output_language: str | None,
        model: str | None,
        request_id: str | None = None,
    ) -> str: ...

    async def cancel(self, meeting_id: uuid.UUID) -> bool: ...

    async def queue_depth(self, queue: QueueName) -> int: ...

    async def close(self) -> None: ...


def build_queue(settings: Settings, inline_runner: InlineRunner | None = None) -> JobQueue:
    if settings.pipeline_execution == "inline":
        if inline_runner is None:
            raise RuntimeError(
                "PIPELINE_EXECUTION=inline requires an InlineRunner; the API runs "
                "the pipeline through BackgroundTasks in inline mode instead."
            )
        return InlineQueue(inline_runner)
    return ArqJobQueue(settings)


# --- Inline (for tests and local dev without Redis) ---


class InlineRunner(Protocol):
    async def process(
        self,
        meeting_id: uuid.UUID,
        *,
        from_stage: str | None,
        callback_url: str | None,
    ) -> None: ...

    async def regenerate_summary(
        self,
        meeting_id: uuid.UUID,
        *,
        output_language: str | None,
        model: str | None,
    ) -> None: ...


class InlineQueue:
    """Runs the pipeline in-process. Never used in production."""

    def __init__(self, runner: InlineRunner) -> None:
        self._runner = runner

    async def enqueue_meeting(
        self,
        meeting_id: uuid.UUID,
        *,
        from_stage: str | None = None,
        request_id: str | None = None,
        callback_url: str | None = None,
    ) -> str:
        await self._runner.process(meeting_id, from_stage=from_stage, callback_url=callback_url)
        return f"inline:{meeting_id}"

    async def enqueue_regenerate_summary(
        self,
        meeting_id: uuid.UUID,
        *,
        output_language: str | None,
        model: str | None,
        request_id: str | None = None,
    ) -> str:
        await self._runner.regenerate_summary(
            meeting_id, output_language=output_language, model=model
        )
        return f"inline:regen:{meeting_id}"

    async def cancel(self, meeting_id: uuid.UUID) -> bool:
        return False

    async def queue_depth(self, queue: QueueName) -> int:
        return 0

    async def close(self) -> None:
        return None


# --- arq (Redis) ---

CANCEL_KEY_PREFIX = "polymom:cancel:"


def cancel_key(meeting_id: uuid.UUID) -> str:
    return f"{CANCEL_KEY_PREFIX}{meeting_id}"


class ArqJobQueue:
    """Fan-out: the first stage runs on the ``cpu`` worker, which enqueues follow-ups.

    Jobs share the same ``_job_id`` per meeting, so a second submit while one is
    running returns the same job (arq refuses duplicates by default).
    """

    def __init__(self, settings: Settings) -> None:
        from arq.connections import RedisSettings

        self._settings = settings
        if not settings.redis_url:
            from app.workers.redis_client import RedisNotConfiguredError

            raise RedisNotConfiguredError("REDIS_URL is required for PIPELINE_EXECUTION=queue.")
        self._redis_settings = RedisSettings.from_dsn(settings.redis_url)
        self._pool: Any | None = None

    async def _get_pool(self) -> Any:
        if self._pool is None:
            from arq import create_pool

            self._pool = await create_pool(self._redis_settings)
        return self._pool

    async def enqueue_meeting(
        self,
        meeting_id: uuid.UUID,
        *,
        from_stage: str | None = None,
        request_id: str | None = None,
        callback_url: str | None = None,
    ) -> str:
        pool = await self._get_pool()
        await pool.delete(cancel_key(meeting_id))  # a new run clears an old cancel flag
        job = await pool.enqueue_job(
            "run_pipeline",
            str(meeting_id),
            from_stage,
            request_id,
            callback_url,
            None,
            _queue_name="polymom:cpu",
        )
        return str(job.job_id) if job is not None else ""

    async def enqueue_regenerate_summary(
        self,
        meeting_id: uuid.UUID,
        *,
        output_language: str | None,
        model: str | None,
        request_id: str | None = None,
    ) -> str:
        pool = await self._get_pool()
        job = await pool.enqueue_job(
            "regenerate_summary",
            str(meeting_id),
            output_language,
            model,
            request_id,
            _queue_name="polymom:llm",
        )
        return job.job_id if job is not None else f"regen:{meeting_id}"

    async def cancel(self, meeting_id: uuid.UUID) -> bool:
        pool = await self._get_pool()
        await pool.set(cancel_key(meeting_id), b"1", ex=24 * 3600)
        return True

    async def queue_depth(self, queue: QueueName) -> int:
        pool = await self._get_pool()
        depth = await pool.zcard(f"polymom:{queue}")
        return int(depth)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


async def is_cancelled(redis: Any, meeting_id: uuid.UUID) -> bool:
    return bool(await redis.get(cancel_key(meeting_id)))
