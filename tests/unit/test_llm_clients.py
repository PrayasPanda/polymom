"""HTTP providers against httpx.MockTransport: payloads, parsing, retries, errors."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.core.exceptions import LLMError
from app.schemas.summary import SummaryHeader
from app.services.llm.anthropic_client import AnthropicClient
from app.services.llm.base import LLMClient, backoff
from app.services.llm.ollama_client import OllamaClient
from app.services.llm.openai_client import OpenAIClient

HEADER = {"title": "T", "executive_summary": "S."}
Handler = Callable[[httpx.Request], httpx.Response]


def settings(**kw: Any) -> Settings:
    return Settings(
        _env_file=None,
        llm_api_key=SecretStr("sk-test"),
        llm_max_retries=2,
        **kw,
    )


def make(cls: type[LLMClient], handler: Handler, delays: list[float], **kw: Any) -> LLMClient:
    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    client_kw = {k: kw.pop(k) for k in ("azure",) if k in kw}
    return cls(
        settings(**kw), "model-x", transport=httpx.MockTransport(handler), sleep=sleep, **client_kw
    )


def openai_reply(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )


async def test_openai_payload_and_parsing() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return openai_reply(json.dumps(HEADER))

    client = make(OpenAIClient, handler, [])
    result = await client.generate_structured("sys", "user", SummaryHeader, 0.1)

    assert result.title == "T"
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://api.openai.com/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer sk-test"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "SummaryHeader"
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["temperature"] == 0.1
    assert (client.calls[0].prompt_tokens, client.calls[0].completion_tokens) == (11, 7)


async def test_azure_endpoint() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return openai_reply(json.dumps(HEADER))

    client = make(
        OpenAIClient,
        handler,
        [],
        azure=True,
        azure_openai_endpoint="https://x.openai.azure.com/",
        azure_openai_deployment="mom",
    )
    await client.generate_structured("s", "u", SummaryHeader)
    assert client.provider == "azure"
    assert str(seen[0].url) == (
        "https://x.openai.azure.com/openai/deployments/mom/chat/completions?api-version=2024-10-21"
    )
    assert seen[0].headers["api-key"] == "sk-test"


async def test_azure_requires_endpoint() -> None:
    client = make(OpenAIClient, lambda r: openai_reply("{}"), [], azure=True)
    with pytest.raises(LLMError, match="AZURE_OPENAI_ENDPOINT"):
        await client.generate_structured("s", "u", SummaryHeader)


async def test_missing_api_key() -> None:
    for cls in (OpenAIClient, AnthropicClient):
        client = cls(Settings(_env_file=None), "m")
        with pytest.raises(LLMError, match="LLM_API_KEY"):
            await client.generate_structured("s", "u", SummaryHeader)


async def test_anthropic_forced_tool_call() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "tool_use", "name": "record_output", "input": HEADER}],
                "usage": {"input_tokens": 5, "output_tokens": 3},
            },
        )

    client = make(AnthropicClient, handler, [])
    assert (await client.generate_structured("sys", "u", SummaryHeader)).title == "T"
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://api.anthropic.com/v1/messages"
    assert seen[0].headers["x-api-key"] == "sk-test"
    assert body["system"] == "sys"
    assert body["tool_choice"] == {"type": "tool", "name": "record_output"}
    assert body["tools"][0]["input_schema"]["title"] == "SummaryHeader"
    assert client.calls[0].prompt_tokens == 5


async def test_anthropic_text_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"content": [{"type": "text", "text": json.dumps(HEADER)}]})

    client = make(AnthropicClient, handler, [])
    assert (await client.generate_structured("s", "u", SummaryHeader)).title == "T"


async def test_ollama_schema_format() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps(HEADER)},
                "prompt_eval_count": 9,
                "eval_count": 4,
            },
        )

    client = make(OllamaClient, handler, [])
    assert (await client.generate_structured("s", "u", SummaryHeader)).title == "T"
    body = json.loads(seen[0].content)
    assert str(seen[0].url) == "http://localhost:11434/api/chat"
    assert body["format"]["title"] == "SummaryHeader"
    assert body["stream"] is False
    assert client.calls[0].completion_tokens == 4


async def test_retries_rate_limit_with_backoff() -> None:
    responses = [
        httpx.Response(429, headers={"retry-after": "3"}),
        httpx.Response(503),
        openai_reply(json.dumps(HEADER)),
    ]
    delays: list[float] = []
    client = make(OpenAIClient, lambda r: responses.pop(0), delays)
    assert (await client.generate_structured("s", "u", SummaryHeader)).title == "T"
    assert delays == [3.0, backoff(1)]
    assert backoff(0) == 1.0 and backoff(10) == 30.0


async def test_gives_up_after_retries() -> None:
    delays: list[float] = []
    client = make(OpenAIClient, lambda r: httpx.Response(500, text="boom"), delays)
    with pytest.raises(LLMError, match="HTTP 500"):
        await client.generate_structured("s", "u", SummaryHeader)
    assert len(delays) == 2


async def test_client_error_is_not_retried() -> None:
    delays: list[float] = []
    client = make(OpenAIClient, lambda r: httpx.Response(401, text="bad key"), delays)
    with pytest.raises(LLMError) as info:
        await client.generate_structured("s", "u", SummaryHeader)
    assert delays == []
    assert info.value.details == {"body": "bad key"}


async def test_timeouts_retry_then_fail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    delays: list[float] = []
    with pytest.raises(LLMError, match="timed out"):
        await make(OllamaClient, handler, delays).generate_structured("s", "u", SummaryHeader)
    assert delays == [1.0, 2.0]


async def test_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMError, match="request failed"):
        await make(OllamaClient, handler, []).generate_structured("s", "u", SummaryHeader)


@pytest.mark.parametrize(
    ("cls", "body"),
    [(OpenAIClient, {"choices": []}), (OllamaClient, {"nope": 1})],
)
async def test_unexpected_response_shape(cls: type[LLMClient], body: dict[str, Any]) -> None:
    client = make(cls, lambda r: httpx.Response(200, json=body), [])
    with pytest.raises(LLMError, match="Unexpected"):
        await client.generate_structured("s", "u", SummaryHeader)
