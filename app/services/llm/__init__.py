"""LLM clients: OpenAI / Azure OpenAI, Anthropic, Ollama and a deterministic mock."""

from app.core.config import Settings
from app.services.llm.anthropic_client import AnthropicClient
from app.services.llm.base import LLMClient
from app.services.llm.mock_client import MockLLMClient
from app.services.llm.ollama_client import OllamaClient
from app.services.llm.openai_client import OpenAIClient

DEFAULT_LLM_MODELS = {
    "openai": "gpt-4o-mini",
    "azure": "gpt-4o-mini",  # informational; Azure routes by AZURE_OPENAI_DEPLOYMENT
    "anthropic": "claude-sonnet-5",
    "ollama": "qwen2.5:7b-instruct",
    "mock": "mock-llm",
}


def build_llm_client(settings: Settings, model: str | None = None) -> LLMClient:
    """Client for ``LLM_PROVIDER``; ``model`` overrides ``LLM_MODEL``."""
    provider = settings.llm_provider
    name = model or settings.llm_model or DEFAULT_LLM_MODELS[provider]
    if provider == "mock":
        return MockLLMClient(settings, name)
    if provider == "anthropic":
        return AnthropicClient(settings, name)
    if provider == "ollama":
        return OllamaClient(settings, name)
    return OpenAIClient(settings, name, azure=provider == "azure")
