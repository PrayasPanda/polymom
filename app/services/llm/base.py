"""Provider-agnostic structured-output LLM client.

Subclasses implement one raw call (:meth:`LLMClient._complete`) that asks the
provider for JSON matching a schema, using its native structured-output mode
where it has one. The base class adds what every provider needs:

- Pydantic validation plus up to ``LLM_MAX_RETRIES`` repair attempts that send
  the validation error back to the model;
- timeouts, and exponential backoff on HTTP 429 / 5xx;
- token usage, latency and cost per call (:attr:`LLMClient.calls`).
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.exceptions import LLMConfigurationError, LLMError, LLMOutputError
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0


@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class LLMCall:
    """Usage record for one :meth:`LLMClient.generate_structured` call."""

    name: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    attempts: int = 0
    output: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    provider: str = "base"

    def __init__(
        self,
        settings: Settings,
        model: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self.model = model
        self.calls: list[LLMCall] = []
        self._transport = transport
        self._sleep = sleep

    @abstractmethod
    async def _complete(
        self, system: str, messages: list[Message], schema: type[BaseModel], temperature: float
    ) -> Completion:
        """One provider call returning raw JSON text for ``schema``."""

    async def generate_structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        temperature: float | None = None,
        *,
        name: str = "generate",
    ) -> T:
        """Return a validated ``schema`` instance, repairing invalid output up to N times."""
        temp = self.settings.llm_temperature if temperature is None else temperature
        messages = [Message("user", user)]
        call = LLMCall(name=name, model=self.model)
        self.calls.append(call)
        started = time.perf_counter()
        try:
            for attempt in range(self.settings.llm_max_retries + 1):
                call.attempts = attempt + 1
                completion = await self._complete(system, messages, schema, temp)
                call.prompt_tokens += completion.prompt_tokens
                call.completion_tokens += completion.completion_tokens
                try:
                    result = schema.model_validate_json(extract_json(completion.text))
                except (ValidationError, ValueError) as exc:
                    logger.warning("llm_output_invalid", name=name, attempt=attempt + 1)
                    messages += [
                        Message("assistant", completion.text),
                        Message("user", repair_prompt(exc)),
                    ]
                    continue
                call.output = result.model_dump(mode="json")
                return result
        finally:
            call.latency_ms = int((time.perf_counter() - started) * 1000)
            call.cost_usd = self._cost(call.prompt_tokens, call.completion_tokens)
        raise LLMOutputError(
            f"The model did not return valid {schema.__name__} JSON after "
            f"{self.settings.llm_max_retries + 1} attempts.",
            details={"provider": self.provider, "model": self.model},
        )

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        s = self.settings
        return (
            prompt_tokens * s.llm_input_cost_per_mtok
            + completion_tokens * s.llm_output_cost_per_mtok
        ) / 1_000_000

    async def _post(
        self, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """POST JSON with a timeout and exponential backoff on retryable statuses."""
        retries = self.settings.llm_max_retries
        async with httpx.AsyncClient(
            timeout=self.settings.llm_timeout_seconds, transport=self._transport
        ) as client:
            for attempt in range(retries + 1):
                try:
                    response = await client.post(url, json=payload, headers=headers)
                except httpx.TimeoutException as exc:
                    if attempt == retries:
                        raise LLMError(f"{self.provider} request timed out.") from exc
                    await self._sleep(backoff(attempt))
                    continue
                except httpx.HTTPError as exc:
                    raise LLMError(f"{self.provider} request failed: {exc}") from exc
                if response.status_code in RETRY_STATUSES and attempt < retries:
                    delay = retry_after(response) or backoff(attempt)
                    logger.warning("llm_retry", status=response.status_code, delay=delay)
                    await self._sleep(delay)
                    continue
                if response.status_code in (401, 403):
                    raise LLMConfigurationError(
                        f"{self.provider} rejected the credentials (HTTP {response.status_code}).",
                        details={"body": response.text[:500]},
                    )
                if response.status_code >= 400:
                    raise LLMError(
                        f"{self.provider} returned HTTP {response.status_code}.",
                        details={"body": response.text[:500]},
                    )
                data: dict[str, Any] = response.json()
                return data
        raise LLMError(f"{self.provider} request failed.")  # pragma: no cover


def backoff(attempt: int) -> float:
    return float(min(BACKOFF_BASE_SECONDS * 2**attempt, BACKOFF_MAX_SECONDS))


def retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after", "")
    try:
        return min(float(value), BACKOFF_MAX_SECONDS)
    except ValueError:
        return None


def extract_json(text: str) -> str:
    """Strip Markdown code fences and any prose around the outermost JSON object."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in the response")
    return text[start : end + 1]


def repair_prompt(error: Exception) -> str:
    detail = str(error)[:2000]
    return (
        "Your previous reply was not valid JSON for the required schema.\n"
        f"Validation error:\n{detail}\n"
        "Reply again with only the corrected JSON object, no prose."
    )


def schema_json(schema: type[BaseModel]) -> dict[str, Any]:
    return schema.model_json_schema()
