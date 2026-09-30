"""Live progress: ``GET /status`` (snapshot) and ``GET /status/stream`` (SSE)."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from app.api.auth import CurrentApiKey  # ensures auth is applied
from app.api.deps import MeetingServiceDep, SettingsDep
from app.core.exceptions import current_request_id
from app.services.meeting_service import MeetingService
from app.workers.progress import Progress, channel, read_progress
from app.workers.redis_client import RedisNotConfiguredError, build_redis

router = APIRouter(prefix="/meetings", tags=["status"])

SSE_PING_SECONDS = 15.0
TERMINAL_STATUSES = frozenset({"completed", "completed_with_errors", "cancelled", "failed"})


async def _resolve_meeting(meeting_id: uuid.UUID, service: MeetingService) -> None:
    await service.get(meeting_id)  # 404 / 401 / owner check


@router.get(
    "/{meeting_id}/status",
    summary="Live progress: percent, current stage, ETA (Redis when the queue runs)",
)
async def get_status(
    meeting_id: uuid.UUID,
    service: MeetingServiceDep,
    settings: SettingsDep,
    _key: CurrentApiKey,
) -> dict[str, object]:
    """Snapshot from Redis, falling back to the last run in the DB (inline mode)."""
    meeting = await service.get(meeting_id)
    if settings.pipeline_execution == "queue" and settings.redis_url:
        redis = build_redis(settings)
        try:
            progress = await read_progress(redis, meeting_id)
        finally:
            await redis.close()
        if progress is not None:
            return progress.model_dump(mode="json")
    run = await service.default_run(meeting_id)
    return Progress(
        meeting_id=meeting_id,
        run_id=run.id if run else None,
        status=meeting.status,
        percent=100.0 if run and run.status == "completed" else 0.0,
        current_stage=None,
        elapsed_seconds=0.0,
        error=meeting.error,
    ).model_dump(mode="json")


@router.get(
    "/{meeting_id}/status/stream",
    summary="Server-Sent Events stream of progress updates (pushed via Redis pub/sub)",
)
async def status_stream(
    meeting_id: uuid.UUID,
    request: Request,
    service: MeetingServiceDep,
    settings: SettingsDep,
    _key: CurrentApiKey,
) -> EventSourceResponse:
    await _resolve_meeting(meeting_id, service)
    if settings.pipeline_execution != "queue" or not settings.redis_url:
        raise RedisNotConfiguredError(
            "SSE progress is only available when PIPELINE_EXECUTION=queue."
        )

    async def event_source() -> AsyncIterator[dict[str, Any]]:
        redis = build_redis(settings)
        pubsub = redis.pubsub()
        await pubsub.subscribe(channel(meeting_id))
        try:
            snapshot = await read_progress(redis, meeting_id)
            if snapshot is not None:
                yield {"event": "progress", "data": snapshot.model_dump_json()}
                if snapshot.status in TERMINAL_STATUSES:
                    return  # already finished: nothing more will be published
            last_ping = asyncio.get_event_loop().time()
            while True:
                if await request.is_disconnected():
                    return
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message.get("type") == "message":
                    raw = message["data"]
                    data = raw.decode() if isinstance(raw, bytes) else str(raw)
                    yield {"event": "progress", "data": data}
                    if json.loads(data).get("status") in TERMINAL_STATUSES:
                        return
                now = asyncio.get_event_loop().time()
                if now - last_ping > SSE_PING_SECONDS:
                    yield {"event": "ping", "data": ""}
                    last_ping = now
        finally:
            await pubsub.unsubscribe(channel(meeting_id))
            await pubsub.close()
            await redis.close()

    return EventSourceResponse(event_source(), headers={"X-Request-ID": current_request_id() or ""})
