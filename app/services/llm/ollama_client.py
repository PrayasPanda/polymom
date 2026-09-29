"""Local models through Ollama (fully offline); ``format`` takes a JSON schema."""

from pydantic import BaseModel

from app.core.exceptions import LLMError
from app.services.llm.base import Completion, LLMClient, Message, schema_json


class OllamaClient(LLMClient):
    provider = "ollama"

    async def _complete(
        self, system: str, messages: list[Message], schema: type[BaseModel], temperature: float
    ) -> Completion:
        s = self.settings
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema_json(schema),
            "options": {"temperature": temperature, "num_predict": s.llm_max_output_tokens},
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
        }
        data = await self._post(f"{s.ollama_base_url.rstrip('/')}/api/chat", payload)
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError("Unexpected Ollama response shape.") from exc
        return Completion(text, data.get("prompt_eval_count", 0), data.get("eval_count", 0))
