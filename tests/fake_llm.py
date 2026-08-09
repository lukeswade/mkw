"""Scripted LLM double, keyed by the `kind` tag each call carries."""
from __future__ import annotations

from collections import defaultdict


class FakeLLM:
    """script: kind → list of payloads.

    - one-entry lists are reused for every call of that kind
    - multi-entry lists are consumed in order (last entry then sticks)
    - a payload may be a callable(messages) → payload
    - an Exception payload is raised
    """

    def __init__(self, script: dict[str, list]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: dict[str, int] = defaultdict(int)
        self.total_calls = 0
        self.usage: dict = {}

    def _next(self, kind: str, messages):
        self.calls[kind] += 1
        self.total_calls += 1
        seq = self.script.get(kind)
        if not seq:
            raise AssertionError(f"FakeLLM: no scripted response for kind={kind!r}")
        payload = seq[0] if len(seq) == 1 else seq.pop(0)
        if callable(payload):
            payload = payload(messages)
        if isinstance(payload, Exception):
            raise payload
        return payload

    async def chat(self, kind, messages, **_kw) -> str:
        return str(self._next(kind, messages))

    async def chat_json(self, kind, messages, schema, **_kw):
        return schema.model_validate(self._next(kind, messages))

    def usage_summary(self) -> dict:
        return {"provider": "fake", "model": "fake", "calls": self.total_calls,
                "prompt_tokens": 0, "completion_tokens": 0, "by_kind": {}}
