"""End-to-end through a real Redis and real arq workers (testcontainers).

Marked ``redis``; skipped when Docker is unavailable. Workers run in burst mode,
one queue at a time, exactly as the three worker processes would in production:
the run hops cpu -> gpu -> cpu -> llm through continuation jobs.
"""

import contextlib
from collections.abc import AsyncIterator, Callable, Iterator

import pytest
from httpx import AsyncClient

from app.core.config import Settings
from app.workers.main import worker_options
from tests.conftest import make_wav, requires_ffmpeg

pytestmark = [pytest.mark.redis, requires_ffmpeg]

URL = "/api/v1/meetings"
ClientFactory = Callable[[Settings], AsyncIterator[AsyncClient]]


@pytest.fixture(scope="module")
def redis_url() -> Iterator[str]:
    try:
        from testcontainers.redis import RedisContainer

        container = RedisContainer("redis:7-alpine")
        container.start()
    except Exception as exc:  # Docker not running, image pull blocked, ...
        pytest.skip(f"Redis container unavailable: {exc}")
    try:
        host, port = container.get_container_host_ip(), container.get_exposed_port(6379)
        yield f"redis://{host}:{port}/0"
    finally:
        container.stop()


@pytest.fixture
def queued(settings: Settings, redis_url: str) -> Settings:
    return settings.model_copy(update={"pipeline_execution": "queue", "redis_url": redis_url})


async def drain(settings: Settings, rounds: int = 8) -> list[str]:
    """Run burst workers queue by queue until nothing is left; returns queues that ran."""
    from arq import create_pool
    from arq.connections import RedisSettings
    from arq.worker import Worker

    ran: list[str] = []
    pool = await create_pool(RedisSettings.from_dsn(settings.redis_url or ""))
    try:
        for _ in range(rounds):
            busy = False
            for queue in ("cpu", "gpu", "llm"):
                if await pool.zcard(f"polymom:{queue}") == 0:
                    continue
                busy = True
                options = worker_options(settings, queue)  # type: ignore[arg-type]
                options["cron_jobs"] = []
                worker = Worker(**options, burst=True, poll_delay=0.05)
                try:
                    await worker.main()
                finally:
                    with contextlib.suppress(AttributeError):  # arq uses SIGUSR1; not on Windows
                        await worker.close()
                ran.append(queue)
            if not busy:
                break
    finally:
        await pool.close()
    return ran


async def test_pipeline_runs_across_resource_queues(
    queued: Settings, client_factory: ClientFactory
) -> None:
    async for client in client_factory(queued):
        upload = await client.post(URL, files={"file": ("m.wav", make_wav(seconds=9))})
        meeting_id = upload.json()["meeting_id"]
        accepted = await client.post(f"{URL}/{meeting_id}/process")
        assert accepted.status_code == 202
        assert (await client.get(f"{URL}/{meeting_id}")).json()["status"] == "processing"

        metrics = (await client.get("/api/v1/metrics")).text
        assert 'polymom_queue_depth{queue="cpu"} 1.0' in metrics

        ran = await drain(queued)

        assert ran[:4] == ["cpu", "gpu", "cpu", "llm"]  # the run hopped between queues
        meeting = (await client.get(f"{URL}/{meeting_id}")).json()
        assert meeting["status"] == "completed", meeting["error"]
        runs = (await client.get(f"{URL}/{meeting_id}/runs")).json()["items"]
        assert len(runs) == 1  # one run, continued by four jobs
        status = (await client.get(f"{URL}/{meeting_id}/status")).json()
        assert (status["status"], status["percent"]) == ("completed", 100.0)
        assert {s["status"] for s in status["stages"]} == {"completed"}
        summary = await client.get(f"{URL}/{meeting_id}/summary")
        assert summary.status_code == 200


async def test_cancel_before_the_worker_starts(
    queued: Settings, client_factory: ClientFactory
) -> None:
    async for client in client_factory(queued):
        upload = await client.post(URL, files={"file": ("m.wav", make_wav(seconds=3))})
        meeting_id = upload.json()["meeting_id"]
        await client.post(f"{URL}/{meeting_id}/process")
        assert (await client.post(f"{URL}/{meeting_id}/cancel")).status_code == 202

        await drain(queued)

        meeting = (await client.get(f"{URL}/{meeting_id}")).json()
        assert meeting["status"] == "cancelled"
        assert meeting["error"].startswith("cancelled:")
        # A cancelled meeting can be processed again without force.
        assert (await client.post(f"{URL}/{meeting_id}/process")).status_code == 202
