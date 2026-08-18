"""Provider resolution, including the legacy path.

An existing install has DEEPSEEK_* / LOCAL_LLM_* in its .env. Those must keep
working untouched after the move to generic llm_* fields — silently switching
someone's endpoint on upgrade would be the worst possible regression.
"""
import pytest

from app.config import Settings, load_settings
from app.llm import providers
from app.llm.client import LLM


def test_preset_supplies_defaults():
    s = Settings(llm_provider="openai", llm_api_key="sk-x")
    assert s.resolved_base_url == "https://api.openai.com/v1"
    assert s.resolved_model == "gpt-4.1-mini"
    assert s.llm_is_configured


def test_explicit_fields_beat_the_preset():
    s = Settings(llm_provider="openai", llm_api_key="sk-x",
                 llm_base_url="https://proxy.internal/v1", llm_model="my-model")
    assert s.resolved_base_url == "https://proxy.internal/v1"
    assert s.resolved_model == "my-model"


def test_legacy_deepseek_env_still_works(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-legacy")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    s = load_settings(str(data_dir))
    assert s.resolved_api_key == "sk-legacy"
    assert s.resolved_model == "deepseek-reasoner"
    assert s.resolved_base_url == "https://api.deepseek.com"
    assert s.llm_is_configured


def test_legacy_local_env_still_works(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://host.docker.internal:8000/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "mlx-community/Qwen3.6-35B")
    monkeypatch.setenv("LOCAL_LLM_API_KEY", "sk-mlx-local")
    s = load_settings(str(data_dir))
    assert s.resolved_base_url == "http://host.docker.internal:8000/v1"
    assert s.resolved_model == "mlx-community/Qwen3.6-35B"
    assert s.resolved_api_key == "sk-mlx-local"
    assert s.llm_is_configured
    assert LLM(s).model == "mlx-community/Qwen3.6-35B"


def test_local_providers_need_no_key_but_do_need_a_model():
    assert not Settings(llm_provider="ollama").llm_is_configured  # no model
    assert Settings(llm_provider="ollama", llm_model="qwen2.5:32b").llm_is_configured
    # a cloud provider is not configured without a key
    assert not Settings(llm_provider="openai").llm_is_configured


def test_fast_model_only_applies_to_notes():
    llm = LLM(Settings(llm_provider="openai", llm_api_key="sk-x",
                       llm_model="big", fast_model="small"))
    assert llm.model_for("notes") == "small"
    for kind in ("planner", "gap", "synth", "followups", "ask", "chat"):
        assert llm.model_for(kind) == "big"


def test_fast_model_defaults_to_the_main_model():
    llm = LLM(Settings(llm_provider="openai", llm_api_key="sk-x", llm_model="big"))
    assert llm.model_for("notes") == "big"
    assert "fast_model" not in llm.usage_summary()


def test_cost_reported_only_where_priced():
    priced = LLM(Settings(llm_provider="deepseek", llm_api_key="k"))
    assert "est_cost_usd" in priced.usage_summary()
    unpriced = LLM(Settings(llm_provider="ollama", llm_model="m"))
    assert "est_cost_usd" not in unpriced.usage_summary()


def test_every_preset_is_coherent():
    for key, p in providers.PROVIDERS.items():
        assert p.key == key
        assert p.base_url.startswith("http")
        assert p.label
        if providers.is_local(key):
            assert not p.needs_key


async def test_unconfigured_provider_fails_with_a_useful_message():
    llm = LLM(Settings(llm_provider="openai"))  # no key, no model override
    with pytest.raises(Exception) as exc:
        await llm.chat("planner", [{"role": "user", "content": "hi"}])
    assert "OpenAI" in str(exc.value)


def test_deepseek_cost_prices_cache_hits(data_dir):
    """Matt's empirical bill: 24.9M tokens ≈ $0.57 — cache hits dominate.
    Costing every input token at the cache-miss rate overestimated ~10x."""
    from app.config import Settings
    from app.llm.client import LLM

    cfg = Settings(data_dir=str(data_dir), llm_provider="deepseek",
                   llm_api_key="sk-test")
    llm = LLM(cfg)

    class Usage:
        prompt_tokens = 1_000_000
        completion_tokens = 100_000
        prompt_cache_hit_tokens = 900_000

    class Resp:
        usage = Usage()

    llm._track("notes", Resp())
    s = llm.usage_summary()
    assert s["cached_tokens"] == 900_000
    # 100k miss @ $0.28 + 900k hit @ $0.028 + 100k out @ $0.42
    expected = round(0.1 * 0.28 + 0.9 * 0.028 + 0.1 * 0.42, 4)
    assert s["est_cost_usd"] == expected
    # sanity: the old all-miss math would have said ~2.4x more
    assert s["est_cost_usd"] < 0.1


def test_cost_without_cache_info_uses_miss_price(data_dir):
    from app.config import Settings
    from app.llm.client import LLM

    cfg = Settings(data_dir=str(data_dir), llm_provider="deepseek",
                   llm_api_key="sk-test")
    llm = LLM(cfg)

    class Usage:
        prompt_tokens = 1_000_000
        completion_tokens = 0

    class Resp:
        usage = Usage()

    llm._track("notes", Resp())
    s = llm.usage_summary()
    assert "cached_tokens" not in s
    assert s["est_cost_usd"] == 0.28
