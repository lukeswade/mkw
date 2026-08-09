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


class LLMError(Exception):
    pass


class LLM:
    def __init__(self, cfg: Settings):
        self.provider = cfg.llm_provider
        if self.provider == "local":
            base, key, self.model = (cfg.local_llm_base_url, "sk-local",
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
                   json_mode: bool = False) -> str:
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
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                async with self._sem:
                    resp = await self.client.chat.completions.create(**kwargs)
                self.total_calls += 1
                self._track(kind, resp)
                return resp.choices[0].message.content or ""
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_err = e
            except APIStatusError as e:
                if e.status_code == 400 and "response_format" in kwargs:
                    # some local servers reject json mode — retry without it
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

    async def chat_json(self, kind: str, messages: list[dict], schema: type[M], *,
                        max_tokens: int = 2048, temperature: float = 0.2) -> M:
        """Structured call: extract_json → validate → one repair round-trip."""
        text = await self.chat(kind, messages, max_tokens=max_tokens,
                               temperature=temperature, json_mode=True)
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
            text2 = await self.chat(kind, repair_messages, max_tokens=max_tokens,
                                    temperature=0.0, json_mode=True)
            try:
                return schema.model_validate(extract_json(text2))
            except (LLMJsonError, ValidationError) as second_err:
                raise LLMJsonError(
                    f"{kind}: unusable JSON after repair attempt: {second_err}"
                ) from second_err
