"""Optional Langfuse tracing: one trace per meeting, one generation span per LLM call.

Enabled by ``LANGFUSE_ENABLED`` and the ``tracing`` extra (``langfuse>=2,<3``).
Tracing never breaks summarization: any Langfuse error is logged and ignored.
"""

from typing import Any, Protocol

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.llm.base import LLMCall

logger = get_logger(__name__)


class Tracer(Protocol):
    def record(self, call: LLMCall, *, prompt_version: str, input: str) -> None: ...

    def flush(self) -> None: ...


class NoopTracer:
    def record(self, call: LLMCall, *, prompt_version: str, input: str) -> None:
        return None

    def flush(self) -> None:
        return None


class LangfuseTracer:  # pragma: no cover - needs the optional langfuse package and a server
    def __init__(self, settings: Settings, meeting_id: str) -> None:
        from langfuse import Langfuse

        self._client: Any = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value()
            if settings.langfuse_public_key
            else None,
            secret_key=settings.langfuse_secret_key.get_secret_value()
            if settings.langfuse_secret_key
            else None,
            host=settings.langfuse_host,
        )
        self._trace = self._client.trace(name="meeting-summary", session_id=meeting_id)

    def record(self, call: LLMCall, *, prompt_version: str, input: str) -> None:
        try:
            self._trace.generation(
                name=call.name,
                model=call.model,
                input=input,
                output=call.output,
                usage={"input": call.prompt_tokens, "output": call.completion_tokens},
                metadata={
                    "prompt_version": prompt_version,
                    "attempts": call.attempts,
                    "cost_usd": call.cost_usd,
                },
                tags=[prompt_version],
            )
        except Exception:
            logger.warning("langfuse_record_failed", exc_info=True)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            logger.warning("langfuse_flush_failed", exc_info=True)


def build_tracer(settings: Settings, meeting_id: str) -> Tracer:
    if not settings.langfuse_enabled:
        return NoopTracer()
    try:  # pragma: no cover - optional dependency
        return LangfuseTracer(settings, meeting_id)
    except Exception:
        logger.warning("langfuse_unavailable", exc_info=True)
        return NoopTracer()
