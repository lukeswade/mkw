"""Robust JSON extraction from LLM output.

llama.cpp's json mode is unreliable across builds and local reasoning models
prepend <think> blocks, so every structured response — regardless of provider
— goes through extract_json: strip thinking/fences, brace-scan, cheap repairs.
"""
from __future__ import annotations

import json
import re
from typing import Any


class LLMJsonError(Exception):
    pass


_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _balanced_slice(text: str, start: int) -> str:
    """Return the substring from `start` to its balanced closing brace/bracket."""
    opener = text[start]
    closer = {"{": "}", "[": "]"}[opener]
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]  # never balanced — let repairs have a go


def _cheap_repairs(text: str) -> str:
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    # normalize curly quotes that some models emit around keys
    text = text.replace("“", '"').replace("”", '"')
    return text


def extract_json(text: str) -> Any:
    if not text or not text.strip():
        raise LLMJsonError("empty response")
    t = _THINK_RE.sub("", text).strip()
    m = _FENCE_RE.search(t)
    if m and m.group(1).strip():
        t = m.group(1).strip()
    starts = [i for i in (t.find("{"), t.find("[")) if i != -1]
    if not starts:
        raise LLMJsonError("no JSON object in response")
    chunk = _balanced_slice(t, min(starts))
    for candidate in (chunk, _cheap_repairs(chunk)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise LLMJsonError("response contained unparseable JSON")
