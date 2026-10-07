"""A server's refusal reaches the run as a clear, typed error.

2026-10-07: with the Mac out of RAM, oMLX's prefill memory guard refused
every call. A JSON-mode call had already been answered 200 (oMLX keeps those
alive with whitespace), so its refusal came as an {"error": ...} body with no
choices, and resp.choices[0] failed a claim check with "'NoneType' object is
not subscriptable". A plain call got a 400 whose advice ("...or reduce
context length") read as a too-long prompt. These tests replay the server's
own bytes through the real SDK, since the bug lived in how the SDK reads them.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI

from app.config import Settings
from app.llm.client import LLM, LLMError, PromptTooLong, ServerOutOfMemory

# oMLX 0.7.0's words, from its server log on 2026-10-07.
_GUARD = ("oMLX prefill memory guard rejected this prompt: Prefill would require "
          "~21.02 GB peak (current 20.96 GB + KV+SDPA 59.29 MB) but dynamic ceiling "
          "is 19.38 GB. Close other apps to free RAM (static cap is 44.16 GB but only "
          "617.36 MB is reclaimable right now), raise memory_guard_tier (safe → "
          "balanced → aggressive), or reduce context length.")
_TOO_LONG = "Prompt too long: 66568 tokens exceeds max context window of 65536 tokens"
_SCHEMA = {"type": "json_schema",
           "json_schema": {"name": "claims", "schema": {"type": "object"}}}
_MSGS = [{"role": "user", "content": "Extract the claims."}]
_BUS = SimpleNamespace(publish=lambda *a, **k: None)


def _error(message: str, code: str | None = None) -> dict:
    """oMLX's error body (_openai_error_body / _prefill_memory_openai_error_body)."""
    return {"error": {"message": message, "type": "invalid_request_error",
                      "param": None, "code": code, "omlx_code": code},
            "type": "error"}


def _keepalive(body: dict) -> str:
    """A JSON-mode reply refused after the 200: keepalive spaces, then the error."""
    return "   " + json.dumps(body)


class _Server:
    """Answers every request with one canned response and records the bodies."""

    def __init__(self, status: int, body: str):
        self.status, self.body = status, body
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return httpx.Response(self.status, text=self.body,
                              headers={"content-type": "application/json"})


def _llm(server: _Server) -> LLM:
    llm = LLM(Settings(llm_provider="local", llm_model="a-model",
                       llm_base_url="http://localhost:8000/v1", llm_api_key="k"))
    llm.client = AsyncOpenAI(
        base_url="http://localhost:8000/v1", api_key="k", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(server)))
    return llm


@pytest.mark.asyncio
async def test_a_refusal_inside_a_200_is_a_clear_error_not_a_type_error():
    server = _Server(200, _keepalive(_error(_GUARD, "prefill_memory_exceeded")))
    llm = _llm(server)
    with pytest.raises(ServerOutOfMemory) as caught:
        await llm.chat_raw("claims", _MSGS, json_mode=True)
    assert "short of memory" in str(caught.value)
    assert "Close other apps to free RAM" in str(caught.value)
    assert len(server.requests) == 1     # terminal: no retry, no field dropped


@pytest.mark.asyncio
async def test_a_memory_refusal_is_not_mistaken_for_a_long_prompt():
    """Its advice says "reduce context length"; shrinking the prompt cannot help."""
    server = _Server(400, json.dumps(_error(_GUARD, "prefill_memory_exceeded")))
    llm = _llm(server)
    with pytest.raises(ServerOutOfMemory) as caught:
        await llm.chat_raw("ask", _MSGS)
    assert not isinstance(caught.value, PromptTooLong)
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_a_refusal_costs_the_run_no_capability():
    server = _Server(400, json.dumps(_error(_GUARD, "prefill_memory_exceeded")))
    llm = _llm(server)
    with pytest.raises(ServerOutOfMemory):
        await llm.chat_raw("claims", _MSGS, response_format=_SCHEMA)
    assert llm.supports_json_schema and llm.suppress_thinking
    assert "chat_template_kwargs" in server.requests[0]


@pytest.mark.asyncio
async def test_a_long_prompt_refused_inside_a_200_is_still_prompt_too_long():
    server = _Server(200, _keepalive(_error(_TOO_LONG)))
    llm = _llm(server)
    with pytest.raises(PromptTooLong) as caught:
        await llm.chat_raw("claims", _MSGS, json_mode=True)
    assert caught.value.limit == 65_536 and llm.context_tokens == 65_536


@pytest.mark.asyncio
async def test_a_long_json_prompt_is_refused_once_not_after_two_fallbacks():
    """It used to step down the schema, then drop JSON mode, then give up:
    three requests, ten seconds of backoff, and no schema for the rest of the run."""
    server = _Server(400, json.dumps(_error(_TOO_LONG)))
    llm = _llm(server)
    with pytest.raises(PromptTooLong):
        await llm.chat_raw("claims", _MSGS, response_format=_SCHEMA)
    assert len(server.requests) == 1 and llm.supports_json_schema


@pytest.mark.asyncio
async def test_an_empty_reply_fails_clearly():
    server = _Server(200, json.dumps({"id": "x", "object": "chat.completion",
                                      "created": 0, "model": "a-model", "choices": []}))
    llm = _llm(server)
    with pytest.raises(LLMError) as caught:
        await llm.chat_raw("notes", _MSGS)
    assert type(caught.value) is LLMError
    assert "no reply" in str(caught.value)


@pytest.mark.asyncio
async def test_a_refused_stream_keeps_thinking_off():
    """chat_stream stripped its optional fields on any 400, so one memory
    refusal during synthesis turned thinking back on for the rest of the run."""
    server = _Server(400, json.dumps(_error(_GUARD, "prefill_memory_exceeded")))
    llm = _llm(server)
    with pytest.raises(ServerOutOfMemory):
        await llm.chat_stream("synth", _MSGS, _BUS, "run")
    assert llm.suppress_thinking and llm.supports_stream_usage
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_a_long_streamed_prompt_raises_prompt_too_long_itself():
    """The synthesizer digests on PromptTooLong; the stream now says so directly
    instead of failing generically and leaving it to an unstreamed retry."""
    server = _Server(400, json.dumps(_error(_TOO_LONG)))
    llm = _llm(server)
    with pytest.raises(PromptTooLong):
        await llm.chat_stream("synth", _MSGS, _BUS, "run")
    assert llm.suppress_thinking and len(server.requests) == 1
