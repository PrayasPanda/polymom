"""Cross-cutting HTTP middleware: request id, metrics, security headers.

``request_id`` is generated (or taken from an inbound ``X-Request-ID``) and bound
into structlog contextvars, so it appears in every log line and error body for the
request, and is echoed back in the ``X-Request-ID`` response header. Jobs enqueued
during the request carry it forward.
"""

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.core.exceptions import PolymomError, error_body
from app.core.logging import get_logger
from app.core.metrics import REQUEST_LATENCY, REQUESTS

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


def new_request_id() -> str:
    return uuid.uuid4().hex


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Bind a request id, time the request, and record Prometheus metrics."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        inbound = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = inbound if 0 < len(inbound) <= 64 and inbound.isalnum() else new_request_id()
        structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Unhandled errors become a generic 500 here, while request_id is still
            # bound, so the body and the logged traceback share the same id.
            logger.exception("request_crashed", path=request.url.path)
            response = JSONResponse(
                error_body(
                    "internal_error",
                    "An unexpected error occurred.",
                    remediation=PolymomError.remediation,
                ),
                status_code=500,
            )
        finally:
            structlog.contextvars.unbind_contextvars("request_id", "path")
        elapsed = time.perf_counter() - started
        route = _route_template(request)
        REQUESTS.labels(request.method, route, response.status_code).inc()
        REQUEST_LATENCY.labels(request.method, route).observe(elapsed)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Conservative security headers for a JSON API (no HTML, so CSP stays strict)."""

    def __init__(self, app: ASGIApp, *, hsts: bool) -> None:
        super().__init__(app)
        self._hsts = hsts

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'"
        )
        response.headers.setdefault("Cache-Control", "no-store")
        if self._hsts:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response


def _route_template(request: Request) -> str:
    """The matched route path (``/meetings/{meeting_id}``), to bound metric cardinality."""
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return str(path) if path else request.url.path
