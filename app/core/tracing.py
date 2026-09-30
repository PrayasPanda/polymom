"""Optional OpenTelemetry tracing (``OTEL_ENABLED``, ``uv sync --extra otel``).

When enabled: FastAPI requests get server spans automatically, and :func:`span`
wraps jobs and pipeline stages. Export goes to the OTLP/HTTP endpoint from the
standard ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var (e.g. an OTel collector or Jaeger).
When disabled or not installed, :func:`span` is a no-op, so call sites stay simple.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_tracer: Any = None


def setup_tracing(settings: Settings, app: Any | None = None) -> bool:
    """Configure the global tracer; returns False when disabled or unavailable."""
    global _tracer
    if not settings.otel_enabled:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("otel_unavailable", hint="uv sync --extra otel")
        return False
    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.otel_service_name})
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("polymom")
    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        except ImportError:  # pragma: no cover - part of the same extra
            logger.warning("otel_fastapi_instrumentation_unavailable")
    logger.info("otel_enabled", service=settings.otel_service_name)
    return True


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """A span when tracing is on; otherwise nothing."""
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, str(value))
        yield
