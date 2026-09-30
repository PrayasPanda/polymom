"""Anthropic Messages API; structured output through a forced tool call."""

import json

from pydantic import BaseModel

from app.core.exceptions import LLMConfigurationError
from app.services.llm.base import Completion, LLMClient, Message, schema_json

ANTHROPIC_VERSION = "2023-06-01"
TOOL_NAME = "record_output"


class AnthropicClient(LLMClient):
    """The schema becomes the ``input_schema`` of one tool the model must call."""

    provider = "anthropic"

    async def _complete(
        self, system: str, messages: list[Message], schema: type[BaseModel], temperature: float
    ) -> Completion:
        s = self.settings
        key = s.llm_api_key.get_secret_value() if s.llm_api_key else ""
        if not key:
            raise LLMConfigurationError("LLM_API_KEY is required for LLM_PROVIDER=anthropic.")
        payload = {
            "model": self.model,
            "max_tokens": s.llm_max_output_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "tools": [
                {
                    "name": TOOL_NAME,
                    "description": f"Record the {schema.__name__} result.",
                    "input_schema": schema_json(schema),
                }
            ],
            "tool_choice": {"type": "tool", "name": TOOL_NAME},
        }
        headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
        data = await self._post(f"{s.anthropic_base_url.rstrip('/')}/v1/messages", payload, headers)
        blocks = data.get("content") or []
        tool = next((b for b in blocks if b.get("type") == "tool_use"), None)
        if tool is not None:
            text = json.dumps(tool.get("input", {}), ensure_ascii=False)
        else:
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = data.get("usage") or {}
        return Completion(text, usage.get("input_tokens", 0), usage.get("output_tokens", 0))
