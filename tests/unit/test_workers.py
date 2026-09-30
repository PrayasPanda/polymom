"""Worker jobs, progress, webhooks and the stuck-job reaper (fakeredis, no Docker)."""

import asyncio
import json
import time
import uuid
from typing import Any

import httpx
import pytest
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis
from pydantic import SecretStr

from app.core.config import Settings
from app.core.exceptions import LLMError, ValidationError
from app.models.meeting import Meeting
from app.pipelines.mom_pipeline import MoMPipeline
from app.repositories.artifacts import LocalArtifactStore
from app.repositories.unit_of_work import UnitOfWorkFactory
from app.schemas.meeting import MeetingStatus
from app.workers import main as jobs
from app.workers import webhook
from app.workers.progress import RedisProgressReporter, read_progress
from app.workers.queue import cancel_key, is_cancelled
from tests.unit.test_pipeline import create_meeting
from tests.unit.test_resume import CountingStage


class FakeArqPool(FakeRedis):
    """fakeredis plus the one arq method jobs use."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(server=FakeServer(), **kwargs)
        self.enqueued: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def enqueue_job(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.enqueued.append((name, args, kwargs))


@pytest.fixture
async def meeting(uow_factory: UnitOfWorkFactory, store: LocalArtifactStore) -> Meeting:
    return await create_meeting(uow_factory, store)


def job_ctx(
    pipeline: MoMPipeline,
    settings: Settings,
    uow_factory: UnitOfWorkFactory,
    queue: str = "cpu",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "redis": FakeArqPool(),
        "settings": settings,
        "queue": queue,
        "shutdown": asyncio.Event(),
        "max_tries": settings.max_retries + 1,
        "pipeline_factory": lambda: pipeline,
        "uow_factory": uow_factory,
        "job_try": 1,
        **extra,
    }


async def test_job_hands_off_to_the_next_queue(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    stages = [CountingStage("pre"), CountingStage("asr", queue="gpu")]
    ctx = job_ctx(MoMPipeline(stages, uow_factory, store, settings), settings, uow_factory)

    result = await jobs.run_pipeline(ctx, str(meeting.id), None, "req-1", None)

    assert result == "processing"
    [(name, args, kwargs)] = ctx["redis"].enqueued
    assert name == "run_pipeline"
    assert kwargs["_queue_name"] == "polymom:gpu"
    assert args[0] == str(meeting.id)
    assert args[2] == "req-1"  # request_id travels with the job
    assert args[4] is not None  # continuation carries the run_id
    assert await ctx["redis"].get(jobs.heartbeat_key(meeting.id))


async def test_job_retries_transient_errors_then_gives_up(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    flaky = CountingStage("flaky", error=LLMError("429"), fail_times=99)
    pipeline = MoMPipeline([flaky], uow_factory, store, settings)
    ctx = job_ctx(pipeline, settings, uow_factory)

    assert await jobs.run_pipeline(ctx, str(meeting.id), None, None, None) == "retrying"
    [(_, args, kwargs)] = ctx["redis"].enqueued
    assert kwargs["_queue_name"] == "polymom:cpu"  # same queue, same run
    run_id = args[4]

    ctx["job_try"] = ctx["max_tries"]
    final = await jobs.run_pipeline(ctx, str(meeting.id), None, None, None, run_id)

    assert final == "failed"
    async with uow_factory() as uow:
        stored = await uow.meetings.get(meeting.id)
    assert stored is not None
    assert stored.status == MeetingStatus.FAILED
    assert (stored.error or "").startswith("llm_error")


async def test_job_does_not_retry_permanent_errors(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    bad = CountingStage("bad", error=ValidationError("corrupt"), fail_times=99)
    ctx = job_ctx(MoMPipeline([bad], uow_factory, store, settings), settings, uow_factory)

    assert await jobs.run_pipeline(ctx, str(meeting.id), None, None, None) == "failed"
    assert ctx["redis"].enqueued == []
    assert bad.calls == 1


async def test_shutdown_checkpoints_and_enqueues_a_continuation(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    stages = [CountingStage("one"), CountingStage("two")]
    ctx = job_ctx(MoMPipeline(stages, uow_factory, store, settings), settings, uow_factory)

    class ShutdownAfterFirstStage:
        calls = 0

        def is_set(self) -> bool:
            self.calls += 1
            return self.calls > 1

    ctx["shutdown"] = ShutdownAfterFirstStage()

    assert await jobs.run_pipeline(ctx, str(meeting.id), None, None, None) == "processing"
    [(_, args, kwargs)] = ctx["redis"].enqueued
    assert kwargs["_queue_name"] == "polymom:cpu"
    assert [s.calls for s in stages] == [1, 0]
    assert args[4] is not None


async def test_reaper_requeues_runs_with_a_stale_heartbeat(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    async with uow_factory() as uow:
        run = await uow.results.create_run(meeting.id, {})
        await uow.commit()
    ctx = job_ctx(
        MoMPipeline([CountingStage("x")], uow_factory, store, settings), settings, uow_factory
    )
    pool: FakeArqPool = ctx["redis"]

    await pool.set(jobs.heartbeat_key(meeting.id), str(time.time()))
    assert await jobs.reap_stuck_jobs(ctx) == 0  # alive

    stale = time.time() - settings.stuck_job_seconds - 5
    await pool.set(jobs.heartbeat_key(meeting.id), str(stale))
    assert await jobs.reap_stuck_jobs(ctx) == 1
    [(_, args, kwargs)] = pool.enqueued
    assert (args[0], args[4], kwargs["_queue_name"]) == (
        str(meeting.id),
        str(run.id),
        "polymom:cpu",
    )
    assert await jobs.reap_stuck_jobs(ctx) == 0  # heartbeat was refreshed on requeue


def test_retry_delay_is_exponential_and_capped(settings: Settings) -> None:
    s = settings.model_copy(update={"retry_backoff_seconds": 10.0})
    assert [jobs.retry_delay(s, n) for n in (1, 2, 3)] == [10.0, 20.0, 40.0]
    assert jobs.retry_delay(s, 20) == 600.0


# --- progress ---


async def test_progress_snapshots_and_pubsub() -> None:
    redis = FakeRedis(server=FakeServer())
    meeting_id = uuid.uuid4()
    pubsub = redis.pubsub()
    await pubsub.subscribe(f"polymom:progress:channel:{meeting_id}")
    reporter = RedisProgressReporter(redis, meeting_id)

    await reporter.start(uuid.uuid4(), ["preprocess", "diarize", "transcribe"], 120.0)
    await reporter.stage("preprocess", "completed")
    await reporter.stage("diarize", "running")
    await reporter.chunk("diarize", 1, 2)

    snapshot = await read_progress(redis, meeting_id)
    assert snapshot is not None
    assert snapshot.current_stage == "diarize"
    diarize = next(s for s in snapshot.stages if s.name == "diarize")
    assert (diarize.chunk, diarize.total_chunks) == (1, 2)
    # preprocess (0.05) + half of diarize (0.15) out of 0.70 total weight
    assert snapshot.percent == pytest.approx(28.6, abs=0.1)
    messages = []
    for _ in range(10):  # the first read consumes the subscribe confirmation
        message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.05)
        if message is not None:
            messages.append(json.loads(message["data"]))
    assert len(messages) == 4
    await reporter.finish(MeetingStatus.COMPLETED, None)
    done = await read_progress(redis, meeting_id)
    assert done is not None
    assert (done.status, done.percent) == (MeetingStatus.COMPLETED, 100.0)


async def test_cancel_flag_roundtrip() -> None:
    redis = FakeRedis(server=FakeServer())
    meeting_id = uuid.uuid4()
    assert not await is_cancelled(redis, meeting_id)
    await redis.set(cancel_key(meeting_id), b"1")
    assert await is_cancelled(redis, meeting_id)


# --- webhooks ---


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com",
        "http://localhost:8000/hook",
        "http://127.0.0.1/hook",
        "http://10.1.2.3/hook",
        "http://192.168.0.10/hook",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/hook",
        "http://0.0.0.0/hook",
        "https:///nohost",
    ],
)
def test_ssrf_blocked_urls(url: str) -> None:
    with pytest.raises(ValidationError):
        webhook.validate_callback_url(url, allow_private=False)


def test_public_url_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webhook, "_resolve", lambda host: ["93.184.216.34"])
    assert webhook.validate_callback_url("https://hooks.example.com/x", allow_private=False)
    monkeypatch.setattr(webhook, "_resolve", lambda host: ["10.0.0.5"])  # DNS rebinding
    with pytest.raises(ValidationError, match="private IP"):
        webhook.validate_callback_url("https://evil.example.com/x", allow_private=False)


def test_signature_roundtrip() -> None:
    body = b'{"status":"completed"}'
    header = webhook.sign("s3cret", body, 1700000000)
    assert header.startswith("sha256=")
    assert webhook.signature_ok("s3cret", body, 1700000000, header)
    assert not webhook.signature_ok("s3cret", body + b" ", 1700000000, header)
    assert not webhook.signature_ok("other", body, 1700000000, header)
    assert not webhook.signature_ok("s3cret", body, 1700000001, header)


def hook_settings(settings: Settings, **kw: Any) -> Settings:
    return settings.model_copy(
        update={"webhook_secret": SecretStr("s3cret"), "webhook_allow_private_hosts": True, **kw}
    )


async def test_webhook_delivery_retries_and_signs(settings: Settings) -> None:
    seen: list[httpx.Request] = []
    replies = iter([503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(next(replies))

    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    meeting_id = uuid.uuid4()
    ok = await webhook.deliver_webhook(
        hook_settings(settings),
        meeting_id,
        "completed",
        "http://127.0.0.1:9/hook",
        request_id="req-9",
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )

    assert ok
    assert len(seen) == 2
    assert sleeps == [2.0]
    last = seen[-1]
    assert json.loads(last.content) == {"meeting_id": str(meeting_id), "status": "completed"}
    assert last.headers["x-request-id"] == "req-9"
    assert webhook.signature_ok(
        "s3cret",
        last.content,
        int(last.headers["x-polymom-timestamp"]),
        last.headers["x-polymom-signature"],
    )


async def test_webhook_stops_on_client_error_and_skips_without_secret(settings: Settings) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(410)

    assert not await webhook.deliver_webhook(
        hook_settings(settings),
        uuid.uuid4(),
        "failed",
        "http://127.0.0.1:9/hook",
        transport=httpx.MockTransport(handler),
    )
    assert len(calls) == 1
    assert not await webhook.deliver_webhook(
        settings, uuid.uuid4(), "failed", "http://127.0.0.1:9/hook"
    )
    blocked = settings.model_copy(update={"webhook_secret": SecretStr("s")})
    assert not await webhook.deliver_webhook(
        blocked, uuid.uuid4(), "failed", "http://127.0.0.1:9/hook"
    )


async def test_webhook_gives_up_after_max_attempts(settings: Settings) -> None:
    async def sleep(_: float) -> None:
        return None

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    assert not await webhook.deliver_webhook(
        hook_settings(settings, webhook_max_attempts=3),
        uuid.uuid4(),
        "completed",
        "http://127.0.0.1:9/hook",
        transport=httpx.MockTransport(handler),
        sleep=sleep,
    )


async def test_handoff_loops_are_stopped(
    meeting: Meeting, uow_factory: UnitOfWorkFactory, store: LocalArtifactStore, settings: Settings
) -> None:
    stages = [CountingStage("pre"), CountingStage("asr", queue="gpu")]
    ctx = job_ctx(MoMPipeline(stages, uow_factory, store, settings), settings, uow_factory)
    await jobs.run_pipeline(ctx, str(meeting.id), None, None, None)
    [(_, args, _)] = ctx["redis"].enqueued
    await ctx["redis"].set(f"polymom:hops:{args[4]}", jobs.MAX_HANDOFFS)

    result = await jobs.run_pipeline(ctx, str(meeting.id), None, None, None, args[4])

    assert result == "failed"
    async with uow_factory() as uow:
        stored = await uow.meetings.get(meeting.id)
    assert stored is not None
    assert (stored.error or "").startswith("handoff_loop")
