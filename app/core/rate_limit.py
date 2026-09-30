"""Per-API-key rate limiting with the ``limits`` library (moving window).

Limits run as FastAPI dependencies *after* authentication, so they count per API key
(per client IP when auth is off). Storage is Redis when ``REDIS_URL`` is set, shared
by every API replica, and in-process memory otherwise. Two budgets:

- ``RATE_LIMIT_DEFAULT`` for every authenticated endpoint;
- ``RATE_LIMIT_UPLOAD`` for the expensive ones: upload, process, summary regeneration.

We use ``limits`` directly, the engine slowapi wraps: slowapi's decorators bind to one
module-level limiter created at import time, which conflicts with this app factory
(each app, and each test, has its own settings and storage).
"""

import math
import time
from collections.abc import Awaitable, Callable
from typing import Literal

from fastapi import Request
from limits import RateLimitItem, parse
from limits.aio.storage import Storage
from limits.aio.strategies import MovingWindowRateLimiter
from limits.storage import storage_from_string

from app.core.config import Settings
from app.core.exceptions import RateLimitedError

Budget = Literal["default", "upload"]


class RateLimiter:
    def __init__(self, settings: Settings) -> None:
        url = settings.redis_url
        use_redis = bool(url) and not str(url).startswith("fakeredis://")
        uri = f"async+{url}" if use_redis else "async+memory://"
        # redis-py (already a dependency for arq) instead of limits' default coredis driver.
        options = {"implementation": "redispy"} if use_redis else {}
        storage = storage_from_string(uri, **options)
        assert isinstance(storage, Storage)  # noqa: S101 - "async+" URIs give async storage
        self.storage = storage
        self.strategy = MovingWindowRateLimiter(self.storage)
        self.items: dict[str, RateLimitItem] = {
            "default": parse(settings.rate_limit_default),
            "upload": parse(settings.rate_limit_upload),
        }

    async def check(self, budget: Budget, key: str) -> None:
        item = self.items[budget]
        if await self.strategy.hit(item, budget, key):
            return
        stats = await self.strategy.get_window_stats(item, budget, key)
        retry_after = max(1, math.ceil(stats.reset_time - time.time()))
        raise RateLimitedError(
            f"Rate limit exceeded ({item}).",
            details={"limit": str(item), "budget": budget, "retry_after": retry_after},
        )


def client_key(request: Request) -> str:
    """Per API key when authenticated, otherwise per client IP."""
    api_key_id = getattr(request.state, "api_key_id", None)
    if api_key_id is not None:
        return f"key:{api_key_id}"
    return f"ip:{request.client.host}" if request.client else "ip:unknown"


def rate_limit(budget: Budget) -> Callable[[Request], Awaitable[None]]:
    async def dependency(request: Request) -> None:
        limiter: RateLimiter = request.app.state.rate_limiter
        await limiter.check(budget, client_key(request))

    return dependency
