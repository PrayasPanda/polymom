"""arq worker entrypoint.

Run one worker per queue so each has its own concurrency and lifecycle::

    python -m app.workers.main --queue cpu          # QUEUE_CONCURRENCY_CPU
    python -m app.workers.main --queue gpu          # QUEUE_CONCURRENCY_GPU
    python -m app.workers.main --queue llm          # QUEUE_CONCURRENCY_LLM

The ``cpu`` worker starts the pipeline: it enqueues the GPU-heavy stages onto
the ``gpu`` queue as separate jobs, and the LLM summary onto ``llm``. Metrics
are exposed on ``WORKER_METRICS_PORT`` (skip with ``--no-metrics``).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import signal
import sys
import time
import uuid
from typing import Any

import structlog
from prometheus_client import start_http_server

from app.core.config import Settings, get_settings
from app.core.exceptions import PolymomError, describe
from app.core.logging import configure_logging, get_logger
from app.core.metrics import JOBS
from app.db.session import create_engine, create_sessionmaker
from app.pipelines.mom_pipeline import build_pipeline
from app.repositories.artifacts import build_artifact_store
from app.repositories.unit_of_work import unit_of_work_factory
from app.services.asr.service import build_router
from app.services.diarization.service import build_backend
from app.services.language.service import build_identifier
from app.services.llm import build_llm_client
from app.workers.progress import RedisProgressReporter
from app.workers.queue import QUEUE_NAMES, QueueName, is_cancelled
from app.workers.redis_client import RedisNotConfiguredError, build_redis
from app.workers.webhook import deliver_webhook

logger = get_logger(__name__)

HEARTBEAT_PREFIX = "polymom:heartbeat:"


def heartbeat_key(job_id: str) -> str:
    return f"{HEARTBEAT_PREFIX}{job_id}"


# --- job functions ---


async def run_pipeline(
    ctx: dict[str, Any],
    meeting_id: str,
    from_stage: str | None,
    request_id: str | None,
    callback_url: str | None,
) -> str:
    """Full pipeline for one meeting; cooperative cancel and Redis-backed progress."""
    settings: Settings = ctx["settings"]
    if request_id:
        structlog.contextvars.bind_contextvars(request_id=request_id)
    mid = uuid.UUID(meeting_id)
    pipeline = build_pipeline(
        settings,
        ctx["uow_factory"],
        ctx["store"],
        build_backend(settings),
        build_router(settings),
        build_identifier(settings),
        build_llm_client(settings),
    )
    redis = ctx["redis"]
    reporter = RedisProgressReporter(redis, mid)
    beat_task = asyncio.create_task(_heartbeat(redis, ctx["job_id"]))
    try:
        status = await pipeline.run(
            mid,
            from_stage=from_stage,
            cancel_check=functools.partial(is_cancelled, redis),
            progress=reporter,
        )
    finally:
        beat_task.cancel()
    if status is not None:
        JOBS.labels(status.value).inc()
    if callback_url and status is not None:
        await deliver_webhook(
            settings, redis, mid, status.value, callback_url, request_id=request_id
        )
    return status.value if status else "missing"


async def regenerate_summary(
    ctx: dict[str, Any],
    meeting_id: str,
    output_language: str | None,
    model: str | None,
    request_id: str | None,
) -> str:
    """Rerun only summarization on the meeting's latest run."""
    from app.pipelines.mom_pipeline import SummarizationStage, SummaryRegenerator

    if request_id:
        structlog.contextvars.bind_contextvars(request_id=request_id)
    settings: Settings = ctx["settings"]
    regen = SummaryRegenerator(
        SummarizationStage(settings, build_llm_client(settings)),
        ctx["uow_factory"],
        ctx["store"],
        settings,
    )
    await regen.run(uuid.UUID(meeting_id), output_language=output_language, model=model)
    return "regenerated"


async def reap_stuck_jobs(ctx: dict[str, Any]) -> None:  # pragma: no cover - runs on the schedule
    """Mark meetings whose worker heartbeat has stopped as ``failed`` so retries can proceed."""
    settings: Settings = ctx["settings"]
    redis = ctx["redis"]
    uow_factory = ctx["uow_factory"]
    now = time.time()
    from sqlalchemy import select

    from app.models.results import ProcessingRun

    async with uow_factory() as uow:
        stmt = select(ProcessingRun).where(ProcessingRun.status == "processing")
        for run in list(await uow.session.scalars(stmt)):
            job_id = f"pipeline:{run.meeting_id}"
            beat = await redis.get(heartbeat_key(job_id))
            last = float(beat) if beat else run.started_at.timestamp()
            if now - last > settings.stuck_job_seconds:
                run.status = "failed"
                run.error = "stuck_job: worker heartbeat stopped"
                meeting = await uow.meetings.get(run.meeting_id)
                if meeting is not None:
                    meeting.status = meeting.status  # keep whatever it was
                    meeting.error = "stuck_job: worker heartbeat stopped"
                logger.warning("stuck_job_reaped", meeting_id=str(run.meeting_id))
        await uow.commit()


async def _heartbeat(redis: Any, job_id: str) -> None:  # pragma: no cover - loop
    while True:
        await redis.set(heartbeat_key(job_id), str(time.time()), ex=120)
        await asyncio.sleep(15)


# --- retry policy ---


async def before_job(ctx: dict[str, Any]) -> None:
    ctx["job_id"] = ctx.get("job_id", f"job:{uuid.uuid4().hex[:8]}")


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, PolymomError):
        return exc.retryable
    return isinstance(exc, TimeoutError | ConnectionError | OSError)


async def on_job_error(ctx: dict[str, Any], exc: BaseException) -> None:
    """arq retries automatically; this hook records the class of failure."""
    code, message, _ = describe(exc if isinstance(exc, Exception) else Exception(str(exc)))
    if _retryable(exc):
        logger.warning(
            "job_transient_failure",
            job=ctx.get("job_id"),
            code=code,
            error=message[:200],
        )
    else:
        # Prevent arq from retrying non-retryable errors by exhausting max_tries.
        ctx["job_try"] = ctx.get("max_tries", 3)
        logger.warning(
            "job_permanent_failure",
            job=ctx.get("job_id"),
            code=code,
            error=message[:200],
        )


# --- worker settings ---


def worker_options(settings: Settings, queue: QueueName) -> dict[str, Any]:
    """Keyword arguments for :class:`arq.worker.Worker` for one queue."""
    from arq.connections import RedisSettings

    if not settings.redis_url:
        raise RedisNotConfiguredError("REDIS_URL is required to run a worker.")

    async def on_startup(ctx: dict[str, Any]) -> None:
        engine = create_engine(settings.resolved_database_url)
        ctx["engine"] = engine
        ctx["uow_factory"] = unit_of_work_factory(create_sessionmaker(engine))
        ctx["store"] = build_artifact_store(settings)
        ctx["redis"] = build_redis(settings)
        ctx["settings"] = settings
        logger.info("worker_started", queue=queue)

    async def on_shutdown(ctx: dict[str, Any]) -> None:  # pragma: no cover - runs at SIGTERM
        if (redis := ctx.get("redis")) is not None:
            await redis.close()
        if (engine := ctx.get("engine")) is not None:
            await engine.dispose()
        logger.info("worker_stopped", queue=queue)

    concurrency = {
        "cpu": settings.queue_concurrency_cpu,
        "gpu": settings.queue_concurrency_gpu,
        "llm": settings.queue_concurrency_llm,
    }[queue]
    return {
        "functions": [run_pipeline, regenerate_summary],
        "redis_settings": RedisSettings.from_dsn(settings.redis_url),
        "queue_name": f"polymom:{queue}",
        "max_jobs": concurrency,
        "max_tries": settings.max_retries + 1,
        "job_timeout": int(max(settings.stage_timeouts.values(), default=3600)),
        "retry_jobs": True,
        "keep_result": 3600,
        "on_startup": on_startup,
        "on_shutdown": on_shutdown,
        "after_job_end": on_job_error,
        "handle_signals": False,
    }


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - external entrypoint
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queue", choices=QUEUE_NAMES, required=True)
    parser.add_argument("--no-metrics", action="store_true")
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.app_env != "development")
    if not args.no_metrics and settings.worker_metrics_port:
        start_http_server(settings.worker_metrics_port)
        logger.info("worker_metrics_started", port=settings.worker_metrics_port)

    from arq.worker import Worker

    worker = Worker(**worker_options(settings, args.queue))
    stop = asyncio.Event()
    loop = asyncio.new_event_loop()

    def _shutdown() -> None:
        logger.info("worker_shutdown_signal")
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):  # Windows has no loop signal handlers
            loop.add_signal_handler(sig, _shutdown)

    async def _serve() -> None:
        task = asyncio.create_task(worker.main())
        await stop.wait()
        # Graceful: stop taking jobs, let running ones finish their current stage.
        await worker.close()
        await task

    try:
        loop.run_until_complete(_serve())
    finally:
        loop.close()


if __name__ == "__main__":
    main(sys.argv[1:])
