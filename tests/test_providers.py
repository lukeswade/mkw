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
    # 100k miss @ $0.44 + 900k hit @ $0.014 + 100k out @ $1.32 (V4 Flash, peak)
    expected = round(0.1 * 0.44 + 0.9 * 0.014 + 0.1 * 1.32, 4)
    assert s["est_cost_usd"] == expected
    # sanity: the old all-miss math would have said ~2.4x more
    assert s["est_cost_usd"] < 0.25            # a 1M-token run with 90% cache hits stays cheap


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
    assert s["est_cost_usd"] == 0.44


async def test_a_streamed_call_reports_the_servers_token_counts(data_dir):
    """chat_stream fabricated usage from len(text)//3. Synthesis is the
    largest call in every run; its cost was a guess."""
    from types import SimpleNamespace as NS
    from app.config import Settings
    from app.llm.client import LLM
    llm = LLM(Settings(data_dir=str(data_dir), llm_provider="openai",
                       llm_api_key="sk-test", llm_model="m"))
    seen_kwargs = {}

    async def fake_create(**kw):
        seen_kwargs.update(kw)
        async def gen():
            yield NS(choices=[NS(delta=NS(content="Hel"))], usage=None)
            yield NS(choices=[NS(delta=NS(content="lo"))], usage=None)
            yield NS(choices=[], usage=NS(prompt_tokens=1234, completion_tokens=56))
        return gen()
    llm.client.chat.completions.create = fake_create

    class Bus:
        chunks = []
        def publish(self, run_id, typ, **f): self.chunks.append(f["chunk"])
    bus = Bus()
    text = await llm.chat_stream("synth", [{"role": "user", "content": "x"}], bus, "r1")
    assert text == "Hello" and bus.chunks == ["Hel", "lo"]
    assert seen_kwargs["stream_options"] == {"include_usage": True}
    assert llm.usage["synth"]["prompt_tokens"] == 1234
    assert llm.usage["synth"]["completion_tokens"] == 56


async def test_a_server_that_rejects_stream_options_still_streams(data_dir):
    """Some local servers 400 on stream_options. Strip it and go again —
    safe, since a 400 on the request precedes any streamed output."""
    import httpx
    from types import SimpleNamespace as NS
    from openai import APIStatusError
    from app.config import Settings
    from app.llm.client import LLM
    llm = LLM(Settings(data_dir=str(data_dir), llm_provider="openai",
                       llm_api_key="sk-test", llm_model="m"))
    calls = []

    async def fake_create(**kw):
        calls.append(dict(kw))
        if "stream_options" in kw:
            raise APIStatusError("unknown field stream_options",
                                 response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
                                 body=None)
        async def gen():
            # long enough that the len//3 estimate fallback is non-zero
            yield NS(choices=[NS(delta=NS(content="ok, streaming without usage"))], usage=None)
        return gen()
    llm.client.chat.completions.create = fake_create

    class Bus:
        def publish(self, *a, **k): pass
    text = await llm.chat_stream("synth", [{"role": "user", "content": "x"}], Bus(), "r1")
    assert text == "ok, streaming without usage"
    assert len(calls) == 2 and "stream_options" not in calls[1]
    assert llm.supports_stream_usage is False
    # and the estimate fallback still recorded something sane
    assert llm.usage["synth"]["completion_tokens"] > 0


async def test_a_call_that_never_returns_is_bounded_by_the_ceiling(data_dir, monkeypatch):
    """The bug: the SDK timeout is idle-only, so a server that trickles tokens
    forever is never cut off. wait_for makes the ceiling a real wall clock."""
    import asyncio, time
    from app.config import Settings
    from app.llm import client as client_mod
    from app.llm.client import LLM, LLMError
    monkeypatch.setattr(client_mod, "_BACKOFF", ())          # no retry sleeps in the test
    llm = LLM(Settings(data_dir=str(data_dir), llm_provider="openai",
                       llm_api_key="sk-test", llm_model="m",
                       llm_timeout=1, llm_call_ceiling=1))

    attempts = []

    async def hangs(**kw):
        attempts.append(1)
        await asyncio.sleep(30)                              # accepted, then silence
    llm.client.chat.completions.create = hangs

    t = time.monotonic()
    with pytest.raises(LLMError, match="ceiling"):
        await llm.chat_raw("notes", [{"role": "user", "content": "x"}])
    assert time.monotonic() - t < 5                          # bounded, not 30s
    # and not retried: a runaway is deterministic for its prompt, so three
    # attempts would just be three runaways (30 minutes at the real ceiling)
    assert len(attempts) == 1


async def test_a_stream_that_never_ends_is_bounded(data_dir):
    """A thinking model with no off switch streams reasoning without end; the
    idle read timeout never fires while it does."""
    import asyncio, time
    from types import SimpleNamespace as NS
    from app.config import Settings
    from app.llm.client import LLM, LLMError
    llm = LLM(Settings(data_dir=str(data_dir), llm_provider="openai",
                       llm_api_key="sk-test", llm_model="m",
                       llm_timeout=1, llm_call_ceiling=1))

    async def endless(**kw):
        async def gen():
            while True:
                yield NS(choices=[NS(delta=NS(content="think "))], usage=None)
                await asyncio.sleep(0.05)
        return gen()
    llm.client.chat.completions.create = endless

    class Bus:
        n = 0
        def publish(self, *a, **k): self.n += 1
    bus = Bus()
    t = time.monotonic()
    with pytest.raises(LLMError):
        await llm.chat_stream("synth", [{"role": "user", "content": "x"}], bus, "r1")
    assert time.monotonic() - t < 10 and bus.n > 0           # streamed, then cut off


async def test_a_failed_notes_call_skips_the_source_not_the_run(data_dir):
    """One timed-out notes call must skip that source, not raise out of the
    round — before the ceiling existed it hung the round; a bare LLMError now
    would kill it instead. take_notes swallows both into a skip."""
    from app.config import Settings
    from app.llm.client import LLM, LLMError
    from app.research.notes import take_notes
    llm = LLM(Settings(data_dir=str(data_dir), llm_provider="openai",
                       llm_api_key="sk-test", llm_model="m"))

    async def boom(*a, **k):
        raise LLMError("exceeded the 600s ceiling")
    llm.chat_json = boom

    out = await take_notes(llm, brief="b", recency_desc="all time",
                           today="2026-09-01", url="http://x/y", title="t",
                           detected_date=None, text="a page worth reading", keywords=None)
    assert out is None
