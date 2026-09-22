"""A local server is told not to think.

Reasoning tokens come out of the same budget as the answer, so a model left
on its thinking default spends the whole allowance before it writes anything
— measured against Qwen3.6-35B-A3B-oQ4e-mtp, one notes prompt gave the same
answer in 59 tokens with thinking off and 1,469 with it on.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from openai import APIStatusError

from app.config import Settings
from app.llm.client import LLM


class _Refused(APIStatusError):
    """A 400 without the response/body plumbing the real class wants."""

    def __init__(self, message: str = "unknown field: chat_template_kwargs"):
        self.status_code = 400
        Exception.__init__(self, message)


def _reply(text: str = '{"ok": true}'):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                 finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


class _Server:
    """Records every request; optionally 400s the first N of them."""

    def __init__(self, refuse_first: int = 0):
        self.calls: list[dict] = []
        self.refuse_first = refuse_first

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.refuse_first:
            raise _Refused()
        return _reply()


def _llm(provider: str = "local", monkeypatch=None) -> LLM:
    cfg = Settings(llm_provider=provider, llm_model="a-model",
                   llm_base_url="http://localhost:8000/v1", llm_api_key="k")
    llm = LLM(cfg)
    return llm


def _attach(llm: LLM, server: _Server) -> None:
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=server.create)))


@pytest.mark.asyncio
async def test_a_local_server_is_asked_not_to_think():
    llm = _llm("local")
    server = _Server()
    _attach(llm, server)
    await llm.chat_raw("notes", [{"role": "user", "content": "hi"}])
    assert server.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.asyncio
async def test_a_cloud_provider_is_left_alone():
    """Clouds reject unknown body fields and price reasoning themselves."""
    llm = _llm("anthropic")
    server = _Server()
    _attach(llm, server)
    await llm.chat_raw("notes", [{"role": "user", "content": "hi"}])
    assert "extra_body" not in server.calls[0]


@pytest.mark.asyncio
async def test_the_env_var_restores_the_server_default(monkeypatch):
    monkeypatch.setenv("LLM_ENABLE_THINKING", "1")
    llm = _llm("local")
    server = _Server()
    _attach(llm, server)
    await llm.chat_raw("notes", [{"role": "user", "content": "hi"}])
    assert "extra_body" not in server.calls[0]


@pytest.mark.asyncio
async def test_a_server_that_rejects_the_flag_costs_one_retry_not_every_call(
        monkeypatch):
    monkeypatch.setattr("app.llm.client._BACKOFF", (0.0, 0.0))
    llm = _llm("local")
    server = _Server(refuse_first=1)
    _attach(llm, server)
    text, _finish = await llm.chat_raw("notes", [{"role": "user", "content": "hi"}])
    assert text == '{"ok": true}'                     # the call still succeeded
    assert "extra_body" in server.calls[0]            # tried once
    assert "extra_body" not in server.calls[1]        # dropped on the retry
    assert llm.suppress_thinking is False             # and not tried again
    server.refuse_first = 0
    await llm.chat_raw("notes", [{"role": "user", "content": "hi"}])
    assert "extra_body" not in server.calls[2]


@pytest.mark.asyncio
async def test_a_too_long_refusal_is_still_reported_as_too_long():
    """The flag fallback must not swallow the context-window refusal."""
    from app.llm.client import PromptTooLong

    class _TooLong(_Refused):
        def __init__(self):
            super().__init__("Prompt too long: max context window of 65,536")

    class _Server2(_Server):
        async def create(self, **kwargs):
            self.calls.append(kwargs)
            raise _TooLong()

    llm = _llm("local")
    server = _Server2()
    _attach(llm, server)
    with pytest.raises(PromptTooLong) as caught:
        await llm.chat_raw("notes", [{"role": "user", "content": "hi"}])
    assert len(server.calls) == 1                     # terminal, not retried
    assert caught.value.limit == 65_536                # the window it named
