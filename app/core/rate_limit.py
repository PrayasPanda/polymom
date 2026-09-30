"""Per-API-key rate limiting via slowapi (Redis when configured, in-memory otherwise)."""

from typing import Any

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

from app.core.config import Settings
from app.core.exceptions import RateLimitedError


def key_func(request: Request) -> str:
    """Limit per API key when authenticated, otherwise per client IP."""
    api_key_id = getattr(request.state, "api_key_id", None)
    if api_key_id is not None:
        return f"key:{api_key_id}"
    return f"ip:{request.client.host}" if request.client else "ip:unknown"


def build_limiter(settings: Settings) -> Limiter:
    storage_uri = settings.redis_url if settings.redis_url else "memory://"
    return Limiter(
        key_func=key_func,
        default_limits=[settings.rate_limit_default],
        storage_uri=storage_uri,
        headers_enabled=True,
    )


def rate_limit_handler(request: Request, exc: Exception) -> Any:
    """Convert slowapi's exception into the project's error envelope with Retry-After."""
    if not isinstance(exc, RateLimitExceeded):  # pragma: no cover
        raise exc
    raise RateLimitedError(
        f"Rate limit exceeded: {exc.detail}",
        details={"retry_after": _seconds_from_detail(str(exc.detail))},
    )


def _seconds_from_detail(detail: str) -> int:
    """Extract the ``/minute`` etc. window as seconds, best-effort (5 as fallback)."""
    for unit, seconds in (("second", 1), ("minute", 60), ("hour", 3600), ("day", 86400)):
        if unit in detail:
            return seconds
    return 5
