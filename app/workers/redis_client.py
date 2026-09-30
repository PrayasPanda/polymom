"""Shared Redis client. In tests, ``fakeredis`` is substituted via the settings."""

from typing import Any

from app.core.config import Settings
from app.core.exceptions import PolymomError


class RedisNotConfiguredError(PolymomError):
    status_code = 503
    code = "redis_not_configured"
    remediation = "Set REDIS_URL to run the queue, or use PIPELINE_EXECUTION=inline."


def build_redis(settings: Settings) -> Any:
    """Create an async Redis client. Uses ``fakeredis`` when the URL uses that scheme."""
    if not settings.redis_url:
        raise RedisNotConfiguredError("REDIS_URL is required for the job queue.")
    url = settings.redis_url
    if url.startswith("fakeredis://"):
        import fakeredis.aioredis

        return fakeredis.aioredis.FakeRedis()
    from redis.asyncio import Redis

    return Redis.from_url(url, decode_responses=False)
