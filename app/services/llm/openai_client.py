"""OpenAI and Azure OpenAI chat completions with native JSON-schema output."""

from typing import Any

from pydantic import BaseModel

from app.core.exceptions import LLMConfigurationError, LLMError
from app.services.llm.base import Completion, LLMClient, Message, schema_json


class OpenAIClient(LLMClient):
    """``response_format: json_schema`` (non-strict, so Pydantic schemas work unchanged).

    With ``azure=True`` the same payload goes to
    ``{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}``.
    """

    provider = "openai"

    def __init__(self, *args: Any, azure: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.azure = azure
        if azure:
            self.provider = "azure"

    def _endpoint(self) -> tuple[str, dict[str, str]]:
        s = self.settings
        key = s.llm_api_key.get_secret_value() if s.llm_api_key else ""
        if not key:
            raise LLMConfigurationError(
                f"LLM_API_KEY is required for LLM_PROVIDER={self.provider}."
            )
        if self.azure:
            if not s.azure_openai_endpoint or not s.azure_openai_deployment:
                raise LLMConfigurationError(
                    "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT are required."
                )
            url = (
                f"{s.azure_openai_endpoint.rstrip('/')}/openai/deployments/"
                f"{s.azure_openai_deployment}/chat/completions"
                f"?api-version={s.azure_openai_api_version}"
            )
            return url, {"api-key": key}
        return f"{s.openai_base_url.rstrip('/')}/chat/completions", {
            "Authorization": f"Bearer {key}"
        }

    async def _complete(
        self, system: str, messages: list[Message], schema: type[BaseModel], temperature: float
    ) -> Completion:
        url, headers = self._endpoint()
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": self.settings.llm_max_output_tokens,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema_json(schema)},
            },
        }
        data = await self._post(url, payload, headers)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected OpenAI response shape.") from exc
        usage = data.get("usage") or {}
        return Completion(text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
