"""arq worker entrypoint.

Run one worker per queue so each has its own concurrency and lifecycle::

    python -m app.workers.main --queue cpu          # QUEUE_CONCURRENCY_CPU
    python -m app.workers.main --queue gpu          # QUEUE_CONCURRENCY_GPU
    python -m app.workers.main --queue llm          # QUEUE_CONCURRENCY_LLM

The API enqueues onto ``cpu``. Each worker runs only the stages of its own queue
(preprocess/align/analytics on cpu; diarize/LID/ASR on gpu; summarize on llm) and
hands the run off to the next queue as a continuation job, so one GPU is never
oversubscribed. The ``cpu`` worker also runs the stuck-job reaper every minute. Metrics
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
from arq import cron
from prometheus_client import start_http_server

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.core.tracing import setup_tracing, span
from app.db.session import create_engine, create_sessionmaker
from app.pipelines.mom_pipeline import MoMPipeline, TransientStageError, build_pipeline
from app.repositories.artifacts import build_artifact_store
from app.repositories.unit_of_work import unit_of_work_factory
from app.schemas.meeting import MeetingStatus
from app.services.asr.service import build_router
from app.services.diarization.service import build_backend
from app.services.language.service import build_identifier
from app.services.llm import build_llm_client
from app.workers.progress import RedisProgressReporter
from app.workers.queue import QUEUE_NAMES, QueueName, is_cancelled
from app.workers.redis_client import RedisNotConfiguredError
from app.workers.webhook import deliver_webhook

logger = get_logger(__name__)

HEARTBEAT_PREFIX = "polymom:heartbeat:"
HEARTBEAT_TTL_SECONDS = 3600


def heartbeat_key(meeting_id: uuid.UUID | str) -> str:
    return f"{HEARTBEAT_PREFIX}{meeting_id}"


def retry_delay(settings: Settings, job_try: int) -> float:
    """Exponential backoff for transient failures: base * 2^(try-1), capped at 10 minutes."""
    return float(min(settings.retry_backoff_seconds * 2 ** max(job_try - 1, 0), 600.0))


async def enqueue_run(
    pool: Any,
    meeting_id: uuid.UUID | str,
    queue: str,
    *,
    run_id: uuid.UUID | str | None = None,
    from_stage: str | None = None,
    request_id: str | None = None,
    callback_url: str | None = None,
) -> None:
    """Enqueue ``run_pipeline`` on ``polymom:{queue}`` (a new run, or a continuation)."""
    await pool.enqueue_job(
        "run_pipeline",
        str(meeting_id),
        from_stage,
        request_id,
        callback_url,
        str(run_id) if run_id else None,
        _queue_name=f"polymom:{queue}",
    )


# --- job functions ---


async def run_pipeline(
    ctx: dict[str, Any],
    meeting_id: str,
    from_stage: str | None,
    request_id: str | None,
    callback_url: str | None,
    run_id: str | None = None,
) -> str:
    """Run this worker's share of a meeting's pipeline.

    Stages belonging to another queue are handed off as a continuation job on that
    queue (``run_id`` set), so a GPU is only ever used by the ``gpu`` workers.
    """
    settings: Settings = ctx["settings"]
    pool = ctx["redis"]
    queue: str = ctx["queue"]
    mid = uuid.UUID(meeting_id)
    rid = uuid.UUID(run_id) if run_id else None
    structlog.contextvars.bind_contextvars(
        meeting_id=meeting_id, queue=queue, **({"request_id": request_id} if request_id else {})
    )
    pipeline: MoMPipeline = ctx["pipeline_factory"]()

    async def handoff(target_run: uuid.UUID, target_queue: str) -> None:
        await enqueue_run(
            pool,
            mid,
            target_queue,
            run_id=target_run,
            request_id=request_id,
            callback_url=callback_url,
        )

    beat = asyncio.create_task(_heartbeat(pool, mid, settings.heartbeat_seconds))
    try:
        with span("job.run_pipeline", meeting_id=meeting_id, queue=queue, run_id=run_id):
            status = await pipeline.run(
                mid,
                run_id=rid,
                from_stage=from_stage if rid is None else None,
                cancel_check=functools.partial(is_cancelled, pool),
                progress=RedisProgressReporter(pool, mid),
                queue=queue,
                handoff=handoff,
                shutdown_check=ctx["shutdown"].is_set,
            )
    except TransientStageError as transient:
        job_try = int(ctx.get("job_try", 1))
        if job_try >= ctx["max_tries"] or transient.run_id is None:
            error = f"{transient.cause.code}: {transient.cause.message}"
            if transient.run_id is not None:
                await pipeline.fail_run(mid, transient.run_id, error)
            status = MeetingStatus.FAILED
            logger.warning("job_retries_exhausted", stage=transient.stage, tries=job_try)
        else:
            delay = retry_delay(settings, job_try)
            logger.warning(
                "job_transient_failure_retrying",
                stage=transient.stage,
                code=transient.cause.code,
                try_=job_try,
                defer_seconds=delay,
            )
            # Retry the same run: completed stages and chunks are skipped on resume.
            await enqueue_run(
                pool,
                mid,
                queue,
                run_id=transient.run_id,
                request_id=request_id,
                callback_url=callback_url,
            )
            return "retrying"
    finally:
        beat.cancel()
    if status is None:
        return "missing"
    if status != MeetingStatus.PROCESSING and callback_url:
        await deliver_webhook(settings, mid, status.value, callback_url, request_id=request_id)
    return str(status.value)


async def regenerate_summary(
    ctx: dict[str, Any],
    meeting_id: str,
    output_language: str | None,
    model: str | None,
    request_id: str | None,
) -> str:
    """Rerun only summarization on the meeting's latest run (``llm`` queue)."""
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


async def reap_stuck_jobs(ctx: dict[str, Any]) -> int:
    """Re-queue runs whose heartbeat stopped for longer than ``STUCK_JOB_SECONDS``.

    A worker that crashed or was killed without a graceful shutdown leaves its run
    ``processing``. The continuation resumes it: completed stages and chunks are
    skipped. Runs on the ``cpu`` worker every minute.
    """
    from sqlalchemy import select

    from app.models.results import ProcessingRun

    settings: Settings = ctx["settings"]
    pool = ctx["redis"]
    now = time.time()
    reaped = 0
    async with ctx["uow_factory"]() as uow:
        running = list(
            await uow.session.scalars(
                select(ProcessingRun).where(ProcessingRun.status == "processing")
            )
        )
        for run in running:
            beat = await pool.get(heartbeat_key(run.meeting_id))
            last = float(beat) if beat else run.started_at.timestamp()
            if now - last <= settings.stuck_job_seconds:
                continue
            meeting = await uow.meetings.get(run.meeting_id)
            if meeting is None:
                continue
            await pool.set(heartbeat_key(run.meeting_id), str(now), ex=HEARTBEAT_TTL_SECONDS)
            await enqueue_run(
                pool, run.meeting_id, "cpu", run_id=run.id, callback_url=meeting.callback_url
            )
            reaped += 1
            logger.warning(
                "stuck_job_requeued",
                meeting_id=str(run.meeting_id),
                run_id=str(run.id),
                silent_seconds=int(now - last),
            )
    return reaped


async def _heartbeat(pool: Any, meeting_id: uuid.UUID, every: float) -> None:
    while True:
        await pool.set(heartbeat_key(meeting_id), str(time.time()), ex=HEARTBEAT_TTL_SECONDS)
        await asyncio.sleep(every)


# --- worker settings ---


def prepare_context(
    ctx: dict[str, Any], settings: Settings, queue: str, shutdown: asyncio.Event
) -> None:
    """Everything a job needs, built once per worker process (also used by tests)."""
    engine = create_engine(settings.resolved_database_url)
    uow_factory = unit_of_work_factory(create_sessionmaker(engine))
    store = build_artifact_store(settings)
    ctx.update(
        engine=engine,
        uow_factory=uow_factory,
        store=store,
        settings=settings,
        queue=queue,
        shutdown=shutdown,
        max_tries=settings.max_retries + 1,
        pipeline_factory=lambda: build_pipeline(
            settings,
            uow_factory,
            store,
            build_backend(settings),
            build_router(settings),
            build_identifier(settings),
            build_llm_client(settings),
        ),
    )


def worker_options(
    settings: Settings, queue: QueueName, shutdown: asyncio.Event | None = None
) -> dict[str, Any]:
    """Keyword arguments for :class:`arq.worker.Worker` for one queue."""
    from arq.connections import RedisSettings

    if not settings.redis_url:
        raise RedisNotConfiguredError("REDIS_URL is required to run a worker.")

    async def on_startup(ctx: dict[str, Any]) -> None:
        # arq puts its own ArqRedis pool in ctx["redis"]; jobs reuse it for progress,
        # heartbeats, cancel flags and hand-off enqueues.
        prepare_context(ctx, settings, queue, shutdown or asyncio.Event())
        logger.info("worker_started", queue=queue)

    async def on_shutdown(ctx: dict[str, Any]) -> None:  # pragma: no cover - runs at SIGTERM
        if (engine := ctx.get("engine")) is not None:
            await engine.dispose()
        logger.info("worker_stopped", queue=queue)

    concurrency = {
        "cpu": settings.queue_concurrency_cpu,
        "gpu": settings.queue_concurrency_gpu,
        "llm": settings.queue_concurrency_llm,
    }[queue]
    return {
        "functions": [run_pipeline, regenerate_summary, reap_stuck_jobs],
        "redis_settings": RedisSettings.from_dsn(settings.redis_url),
        "queue_name": f"polymom:{queue}",
        "max_jobs": concurrency,
        "max_tries": settings.max_retries + 1,
        "job_timeout": int(max(settings.stage_timeouts.values(), default=3600)),
        "retry_jobs": True,
        "keep_result": 3600,
        "on_startup": on_startup,
        "on_shutdown": on_shutdown,
        "handle_signals": False,
        "cron_jobs": [cron(reap_stuck_jobs, second=0, run_at_startup=True)]
        if queue == "cpu"
        else [],
        # Give running jobs time to reach a chunk boundary and checkpoint on SIGTERM.
        "job_completion_wait": settings.shutdown_grace_seconds,
    }


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - external entrypoint
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queue", choices=QUEUE_NAMES, required=True)
    parser.add_argument("--no-metrics", action="store_true")
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.app_env != "development")
    setup_tracing(settings)
    if not args.no_metrics and settings.worker_metrics_port:
        start_http_server(settings.worker_metrics_port)
        logger.info("worker_metrics_started", port=settings.worker_metrics_port)

    from arq.worker import Worker

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stop = asyncio.Event()
    worker = Worker(**worker_options(settings, args.queue, shutdown=stop))

    def _shutdown() -> None:
        # Jobs see ctx["shutdown"] set: they stop at the next chunk or stage boundary,
        # checkpoint, and enqueue a continuation that resumes on another worker.
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
