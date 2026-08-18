"""Known OpenAI-compatible LLM endpoints.

The client only ever needs a base URL, a key and a model name, so supporting a
new provider is a row in this table rather than a code change. Presets supply
sensible defaults; the user can override any of them.

`price_in`/`price_out` are USD per million tokens for the preset's default
model and exist only to give a rough per-run cost. They are left None wherever
pricing depends too much on which model you pick — token counts are still
reported in that case.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    base_url: str
    default_model: str
    needs_key: bool = True
    price_in: float | None = None
    price_out: float | None = None
    # Cached-input price. Research prompts share long template prefixes, so
    # providers with context caching serve most input tokens at this rate —
    # ignoring it overestimated real DeepSeek runs by an order of magnitude.
    price_cache_in: float | None = None
    hint: str = ""


PROVIDERS: dict[str, Provider] = {
    "deepseek": Provider(
        "deepseek", "DeepSeek", "https://api.deepseek.com", "deepseek-chat",
        price_in=0.28, price_out=0.42, price_cache_in=0.028,
        hint="Cheap and strong at structured output. Key: platform.deepseek.com"),
    "openai": Provider(
        "openai", "OpenAI", "https://api.openai.com/v1", "gpt-4.1-mini",
        price_in=0.40, price_out=1.60, price_cache_in=0.10,
        hint="Key: platform.openai.com. Supports strict json_schema output."),
    "openrouter": Provider(
        "openrouter", "OpenRouter", "https://openrouter.ai/api/v1",
        "deepseek/deepseek-chat",
        hint="One key, hundreds of models. Pricing depends on the model."),
    "groq": Provider(
        "groq", "Groq", "https://api.groq.com/openai/v1",
        "llama-3.3-70b-versatile",
        hint="Very fast inference — good fit for the fast-model slot."),
    "together": Provider(
        "together", "Together AI", "https://api.together.xyz/v1",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        hint="Wide open-model selection."),
    "lmstudio": Provider(
        "lmstudio", "LM Studio (local)", "http://host.docker.internal:1234/v1",
        "", needs_key=False,
        hint="Start the LM Studio server, then paste the model id it shows."),
    "ollama": Provider(
        "ollama", "Ollama (local)", "http://host.docker.internal:11434/v1",
        "", needs_key=False,
        hint="Run `ollama serve`. Model id is e.g. qwen2.5:32b-instruct."),
    "local": Provider(
        "local", "Other OpenAI-compatible", "http://host.docker.internal:8080/v1",
        "", needs_key=False,
        hint="llama.cpp, vLLM, MLX, LiteLLM — anything speaking /v1/chat/completions."),
}

DEFAULT_PROVIDER = "deepseek"
# Providers that run on your own hardware: no API cost, and a missing key is
# normal rather than a misconfiguration.
LOCAL_PROVIDERS = {"lmstudio", "ollama", "local"}


def get(key: str) -> Provider:
    return PROVIDERS.get(key) or PROVIDERS[DEFAULT_PROVIDER]


def is_local(key: str) -> bool:
    return key in LOCAL_PROVIDERS
