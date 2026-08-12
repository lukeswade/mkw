"""Provider-agnostic async LLM client.

DeepSeek and llama.cpp both speak the OpenAI chat-completions dialect, so one
AsyncOpenAI client with a configurable base_url covers both. Every call
carries a `kind` tag (planner/notes/gap/synth/...) used for usage accounting
and for routing canned responses in tests.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TypeVar

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.llm.json_utils import LLMJsonError, extract_json
from app.llm.schema_utils import response_format_for

log = logging.getLogger(__name__)

M = TypeVar("M", bound=BaseModel)

# Deliberately conservative estimate (English ≈ 4 chars/token) — used only
# for budgeting, where 25% headroom matters more than precision.
def est_tokens(text: str) -> int:
    return len(text) // 3


# USD per 1M tokens, deepseek-chat standard pricing — for the per-run cost
# estimate shown in stats. Rough by design.
_DEEPSEEK_IN_PER_M = 0.27
_DEEPSEEK_OUT_PER_M = 1.10

_BACKOFF = (2.0, 8.0)

# Ceiling for the truncation retry. DeepSeek caps output at 8k; local servers
# vary, but doubling past this wastes time rather than fixing anything.
_MAX_OUTPUT_TOKENS = 8000


class LLMError(Exception):
    pass


class LLM:
    def __init__(self, cfg: Settings):
        self.provider = cfg.llm_provider
        if self.provider == "local":
            base, key, self.model = (cfg.local_llm_base_url,
                                     cfg.local_llm_api_key or "sk-local",
                                     cfg.local_llm_model)
        else:
            base, key, self.model = (cfg.deepseek_base_url, cfg.deepseek_api_key,
                                     cfg.deepseek_model)
        self._configured = bool(key)
        self.client = AsyncOpenAI(base_url=base, api_key=key or "missing",
                                  timeout=cfg.llm_timeout, max_retries=0)
        self._sem = asyncio.Semaphore(cfg.llm_concurrency)
        self.usage: dict[str, dict[str, int]] = {}
        self.total_calls = 0
        # Constrained decoding, disabled automatically if the server 400s on it.
        self.supports_json_schema = True

    def _track(self, kind: str, resp) -> None:
        u = self.usage.setdefault(kind, {"calls": 0, "prompt_tokens": 0,
                                         "completion_tokens": 0})
        u["calls"] += 1
        if getattr(resp, "usage", None):
            u["prompt_tokens"] += resp.usage.prompt_tokens or 0
            u["completion_tokens"] += resp.usage.completion_tokens or 0

    def usage_summary(self) -> dict:
        total_in = sum(u["prompt_tokens"] for u in self.usage.values())
        total_out = sum(u["completion_tokens"] for u in self.usage.values())
        summary: dict = {
            "provider": self.provider,
            "model": self.model,
            "calls": self.total_calls,
            "prompt_tokens": total_in,
            "completion_tokens": total_out,
            "by_kind": self.usage,
        }
        if self.provider == "deepseek":
            summary["est_cost_usd"] = round(
                total_in / 1e6 * _DEEPSEEK_IN_PER_M
                + total_out / 1e6 * _DEEPSEEK_OUT_PER_M, 4)
        return summary

    async def chat(self, kind: str, messages: list[dict], *,
                   max_tokens: int = 2048, temperature: float = 0.3,
                   json_mode: bool = False,
                   response_format: dict | None = None) -> str:
        text, _finish = await self.chat_raw(
            kind, messages, max_tokens=max_tokens, temperature=temperature,
            json_mode=json_mode, response_format=response_format)
        return text

    async def chat_raw(self, kind: str, messages: list[dict], *,
                       max_tokens: int = 2048, temperature: float = 0.3,
                       json_mode: bool = False,
                       response_format: dict | None = None) -> tuple[str, str]:
        """Return (content, finish_reason).

        finish_reason matters: 'length' means the model was cut off mid-answer,
        so the JSON is truncated and no amount of repair prompting will fix it —
        the caller needs a bigger budget, not a retry.
        """
        if self.provider == "deepseek" and not self._configured:
            raise LLMError(
                "No DeepSeek API key configured — add it on the Settings page "
                "or as DEEPSEEK_API_KEY in .env (or switch to a local LLM)."
            )
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                async with self._sem:
                    resp = await self.client.chat.completions.create(**kwargs)
                self.total_calls += 1
                self._track(kind, resp)
                choice = resp.choices[0]
                return (choice.message.content or "",
                        getattr(choice, "finish_reason", "") or "")
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_err = e
            except APIStatusError as e:
                if e.status_code == 400 and "response_format" in kwargs:
                    fmt = kwargs["response_format"].get("type")
                    if fmt == "json_schema":
                        # server can't constrain to a schema — step down to
                        # plain json mode and stop trying for this session
                        self.supports_json_schema = False
                        kwargs["response_format"] = {"type": "json_object"}
                    else:
                        # some local servers reject json mode entirely
                        kwargs.pop("response_format")
                    last_err = e
                elif e.status_code >= 500:
                    last_err = e
                elif e.status_code in (401, 403):
                    raise LLMError(
                        f"LLM auth failed ({e.status_code}) — check the API key "
                        f"for provider '{self.provider}'."
                    ) from e
                else:
                    raise LLMError(f"LLM request rejected: {e}") from e
            except APIError as e:
                last_err = e
            if attempt < len(_BACKOFF):
                await asyncio.sleep(_BACKOFF[attempt])
        raise LLMError(f"LLM call '{kind}' failed after retries: {last_err}")

    async def chat_stream(self, kind: str, messages: list[dict], bus, run_id: str, *,
                          max_tokens: int = 2048, temperature: float = 0.3) -> str:
        """Stream chat completions and publish chunks to the progress bus."""
        if self.provider == "deepseek" and not self._configured:
            raise LLMError("No DeepSeek API key configured.")
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        
        # We don't retry streaming calls because partial output might have already been sent to the user.
        try:
            full_text = []
            async with self._sem:
                stream = await self.client.chat.completions.create(**kwargs)
                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_text.append(content)
                        bus.publish(run_id, "stream", chunk=content)
            self.total_calls += 1
            # We don't have accurate token counts for streams from all providers, so we can estimate
            final_text = "".join(full_text)
            self._track(kind, type("DummyResp", (), {"usage": type("DummyUsage", (), {"prompt_tokens": est_tokens(str(messages)), "completion_tokens": est_tokens(final_text)})}))
            return final_text
        except Exception as e:
            raise LLMError(f"LLM stream call '{kind}' failed: {e}") from e

    async def chat_json(self, kind: str, messages: list[dict], schema: type[M], *,
                        max_tokens: int = 2048, temperature: float = 0.2) -> M:
        """Structured call.

        Preference order, each step falling back to the next:
          1. constrained decoding against the model's JSON schema
          2. plain json mode, then extract_json's tolerant parsing
          3. one repair round-trip
        A truncated response (finish_reason 'length') is retried with a larger
        budget rather than repaired — repair re-sends the prompt plus the
        truncated text, so it just truncates again.
        """
        fmt = response_format_for(schema) if self.supports_json_schema else None
        budget = max_tokens

        for attempt in range(2):
            text, finish = await self.chat_raw(
                kind, messages, max_tokens=budget, temperature=temperature,
                json_mode=True,
                response_format=fmt if self.supports_json_schema else None)
            if finish != "length":
                break
            if attempt == 0:
                budget = min(budget * 2, _MAX_OUTPUT_TOKENS)
                log.warning("%s: response truncated, retrying with %d tokens",
                            kind, budget)

        try:
            return schema.model_validate(extract_json(text))
        except (LLMJsonError, ValidationError) as first_err:
            log.warning("%s: JSON parse failed (%s), attempting repair", kind,
                        str(first_err)[:200])
            repair_messages = [
                *messages,
                {"role": "assistant", "content": text[:6000]},
                {"role": "user", "content": (
                    f"Your previous output could not be used: {str(first_err)[:500]}\n"
                    "Respond again with ONLY the corrected JSON object — "
                    "no explanation, no markdown fences."
                )},
            ]
            text2 = await self.chat(
                kind, repair_messages, max_tokens=budget, temperature=0.0,
                json_mode=True,
                response_format=fmt if self.supports_json_schema else None)
            try:
                return schema.model_validate(extract_json(text2))
            except (LLMJsonError, ValidationError) as second_err:
                raise LLMJsonError(
                    f"{kind}: unusable JSON after repair attempt: {second_err}"
                ) from second_err
