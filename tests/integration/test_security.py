"""Authentication, tenant isolation (BOLA), rate limits, idempotency and hardening."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.core.config import Settings
from app.db.session import create_engine, create_sessionmaker
from app.repositories.unit_of_work import unit_of_work_factory
from app.workers.progress import RedisProgressReporter
from app.workers.redis_client import build_redis
from tests.conftest import make_wav, requires_ffmpeg

URL = "/api/v1/meetings"
ClientFactory = Callable[[Settings], AsyncIterator[AsyncClient]]


async def create_key(settings: Settings, label: str) -> str:
    engine = create_engine(settings.resolved_database_url)
    try:
        async with unit_of_work_factory(create_sessionmaker(engine))() as uow:
            _, token = await uow.api_keys.create(label=label, owner=label)
            await uow.commit()
        return token
    finally:
        await engine.dispose()


@pytest.fixture
def secure(settings: Settings) -> Settings:
    return settings.model_copy(update={"api_key_required": True})


async def upload(client: AsyncClient, key: str | None, seconds: float = 1, **kw: str) -> dict:
    headers = {"X-API-Key": key} if key else {}
    response = await client.post(
        URL,
        files={"file": ("m.wav", make_wav(seconds=seconds))},
        headers=headers,
        params={"allow_duplicate": "true"},
        data=kw,
    )
    return {"status": response.status_code, "json": response.json(), "headers": response.headers}


@requires_ffmpeg
async def test_api_keys_and_tenant_isolation(
    secure: Settings, client_factory: ClientFactory
) -> None:
    async for client in client_factory(secure):
        alice, bob = await create_key(secure, "alice"), await create_key(secure, "bob")
        A, B = {"X-API-Key": alice}, {"X-API-Key": bob}  # noqa: N806

        missing = await client.get(URL)
        assert missing.status_code == 401
        assert missing.json()["error"]["code"] == "unauthorized"
        assert (await client.get(URL, headers={"X-API-Key": "pk_nope"})).status_code == 401

        created = await upload(client, alice)
        assert created["status"] == 202
        meeting_id = created["json"]["meeting_id"]

        assert (await client.get(f"{URL}/{meeting_id}", headers=A)).status_code == 200
        # Bob cannot see, list, process, cancel, export or delete Alice's meeting.
        for method, path in [
            ("GET", f"{URL}/{meeting_id}"),
            ("GET", f"{URL}/{meeting_id}/status"),
            ("GET", f"{URL}/{meeting_id}/result"),
            ("GET", f"{URL}/{meeting_id}/export"),
            ("POST", f"{URL}/{meeting_id}/process"),
            ("POST", f"{URL}/{meeting_id}/cancel"),
            ("DELETE", f"{URL}/{meeting_id}"),
        ]:
            response = await client.request(method, path, headers=B)
            assert response.status_code == 404, (method, path)
            assert response.json()["error"]["code"] == "meeting_not_found"
        assert (await client.get(URL, headers=B)).json()["total"] == 0
        assert (await client.get(URL, headers=A)).json()["total"] == 1
        # Bob's identical upload is a new meeting, not a "duplicate" of Alice's.
        bob_upload = await client.post(
            URL, files={"file": ("m.wav", make_wav(seconds=1))}, headers=B
        )
        assert bob_upload.json()["duplicate"] is False


@requires_ffmpeg
async def test_rate_limit_returns_429_with_retry_after(
    settings: Settings, client_factory: ClientFactory
) -> None:
    limited = settings.model_copy(update={"rate_limit_upload": "2/minute"})
    async for client in client_factory(limited):
        statuses = [(await upload(client, None))["status"] for _ in range(3)]
        assert statuses == [202, 202, 429]
        blocked = await upload(client, None)
        assert blocked["json"]["error"]["code"] == "rate_limited"
        assert int(blocked["headers"]["retry-after"]) > 0
        assert (await client.get(URL)).status_code == 200  # the default budget is separate


@requires_ffmpeg
async def test_idempotency_keys(client: AsyncClient) -> None:
    headers = {"Idempotency-Key": "upload-123"}
    files = {"file": ("m.wav", make_wav(seconds=1))}
    first = await client.post(URL, files=files, headers=headers, params={"allow_duplicate": "true"})
    again = await client.post(URL, files=files, headers=headers, params={"allow_duplicate": "true"})

    assert first.status_code == again.status_code == 202
    assert first.json() == again.json()
    assert again.headers["idempotent-replayed"] == "true"
    assert (await client.get(URL)).json()["total"] == 1  # not uploaded twice

    meeting_id = first.json()["meeting_id"]
    reused = await client.post(f"{URL}/{meeting_id}/process", headers=headers)
    assert reused.status_code == 409
    assert reused.json()["error"]["code"] == "idempotency_key_reused"


@requires_ffmpeg
async def test_callback_url_ssrf_and_duration_limit(
    settings: Settings, client_factory: ClientFactory
) -> None:
    short = settings.model_copy(update={"max_audio_duration_minutes": 0.01})  # 0.6 s
    async for client in client_factory(short):
        ssrf = await upload(client, None, callback_url="http://169.254.169.254/latest/meta-data")
        assert ssrf["status"] == 422
        assert "private" in ssrf["json"]["error"]["message"]
        scheme = await upload(client, None, callback_url="file:///etc/passwd")
        assert scheme["status"] == 422

        too_long = await upload(client, None, seconds=2)
        assert too_long["status"] == 422
        assert too_long["json"]["error"]["code"] == "audio_too_long"
        assert too_long["json"]["error"]["remediation"]


@requires_ffmpeg
async def test_malicious_playlist_with_external_reference_is_rejected(
    tmp_path: Path, settings: Settings
) -> None:
    """An HLS playlist pointing at an internal URL must never be fetched by ffprobe."""
    from app.core.exceptions import CorruptedMediaError
    from app.services.audio.validator import MediaValidator

    playlist = tmp_path / "evil.m3u8"
    playlist.write_text(
        "#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10,\nhttp://127.0.0.1:9/secret.ts\n"
        "#EXT-X-ENDLIST\n"
    )
    with pytest.raises(CorruptedMediaError):
        await MediaValidator(settings).probe(playlist)


async def test_security_headers_request_id_and_health(client: AsyncClient) -> None:
    response = await client.get(URL, headers={"X-Request-ID": "abc123"})
    assert response.headers["x-request-id"] == "abc123"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'none'" in response.headers["content-security-policy"]
    generated = await client.get(URL, headers={"X-Request-ID": "not valid!"})
    assert len(generated.headers["x-request-id"]) == 32  # unsafe ids are replaced

    assert (await client.get("/api/v1/health/live")).json() == {"status": "ok"}
    ready = await client.get("/api/v1/health/ready")
    assert ready.status_code == 200
    checks = ready.json()["checks"]
    assert checks["database"]["ok"]
    assert checks["artifact_store"]["ok"]
    assert checks["models"]["ok"]  # mock backends

    metrics = await client.get("/api/v1/metrics")
    assert metrics.status_code == 200
    assert "polymom_http_requests_total" in metrics.text
    assert 'route="/meetings"' in metrics.text  # route template, not the raw path


async def test_unhandled_errors_never_leak_stack_traces(
    settings: Settings, client_factory: ClientFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services.meeting_service import MeetingService

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(MeetingService, "page", boom)
    async for client in client_factory(settings):
        response = await client.get(URL)
    assert response.status_code == 500
    body = response.json()["error"]
    assert body["code"] == "internal_error"
    assert "secret" not in response.text
    assert body["request_id"] == response.headers["x-request-id"]


async def test_cors_allowlist(settings: Settings, client_factory: ClientFactory) -> None:
    cors = settings.model_copy(update={"cors_origins": ["https://app.example.com"]})
    async for client in client_factory(cors):
        allowed = await client.options(
            URL,
            headers={
                "Origin": "https://app.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        denied = await client.options(
            URL,
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
        )
    assert allowed.headers["access-control-allow-origin"] == "https://app.example.com"
    assert "access-control-allow-origin" not in denied.headers


@pytest.mark.parametrize(
    ("overrides", "problem"),
    [
        ({"redis_url": None}, "REDIS_URL"),
        ({"api_key_required": False}, "API_KEY_REQUIRED"),
        ({"llm_provider": "openai", "llm_api_key": None}, "LLM_API_KEY"),
        ({"cors_origins": "*"}, "CORS_ORIGINS"),
    ],
)
def test_production_settings_fail_fast(overrides: dict[str, object], problem: str) -> None:
    base: dict[str, object] = {
        "_env_file": None,
        "app_env": "production",
        "redis_url": "redis://redis:6379/0",
        "llm_provider": "mock",
        "diarization_backend": "mock",
    }
    with pytest.raises(ValueError, match=problem):
        Settings(**{**base, **overrides})  # type: ignore[arg-type]


@requires_ffmpeg
async def test_status_snapshot_and_sse_stream(
    settings: Settings, client_factory: ClientFactory
) -> None:
    queued = settings.model_copy(
        update={"pipeline_execution": "queue", "redis_url": "fakeredis://"}
    )
    async for client in client_factory(queued):
        meeting_id = (await upload(client, None))["json"]["meeting_id"]
        from uuid import UUID

        reporter = RedisProgressReporter(build_redis(queued), UUID(meeting_id))

        async def publish(reporter: RedisProgressReporter) -> None:
            await asyncio.sleep(0.2)
            await reporter.start(UUID(int=1), ["preprocess", "diarize"], 60.0)
            await reporter.stage("preprocess", "completed")
            await reporter.stage("diarize", "running")
            from app.schemas.meeting import MeetingStatus

            await reporter.finish(MeetingStatus.COMPLETED, None)

        task = asyncio.create_task(publish(reporter))
        stream = await client.get(f"{URL}/{meeting_id}/status/stream")
        await task

        assert stream.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line.removeprefix("data: "))
            for line in stream.text.splitlines()
            if line.startswith("data: ") and line != "data: "
        ]
        assert [e["status"] for e in events][-1] == "completed"
        assert any(e["current_stage"] == "diarize" for e in events)
        snapshot = (await client.get(f"{URL}/{meeting_id}/status")).json()
        assert (snapshot["status"], snapshot["percent"]) == ("completed", 100.0)


async def test_sse_requires_the_queue(client: AsyncClient) -> None:
    meeting_id = "3f8b6f0e-2c1d-4d6a-9d3e-6c2b1a0f9e7d"
    response = await client.get(f"{URL}/{meeting_id}/status/stream")
    assert response.status_code == 404  # unknown meeting is checked first


async def test_oversized_json_body_is_rejected(client: AsyncClient) -> None:
    body = b'{"names": {"Person 1": "' + b"x" * 2_000_000 + b'"}}'
    response = await client.patch(
        "/api/v1/meetings/00000000-0000-0000-0000-000000000000/speakers",
        content=body,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
