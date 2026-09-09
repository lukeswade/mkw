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


def close_open(text: str) -> str:
    """Close an open string and any open containers, keeping partial content.

    The generous half of truncation salvage: a notes_md cut mid-sentence keeps
    the sentence. Invalid when the text stopped on a bare key or a half-typed
    number, which is what close_truncated() is for.
    """
    stack: list[str] = []
    in_string = escaped = False
    for ch in text:
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
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
    if not stack and not in_string:
        return text
    out = text + ('"' if in_string else "")
    return out.rstrip().rstrip(",") + "".join(reversed(stack))


def close_truncated(text: str) -> str:
    """Complete a JSON value that was cut off mid-flight.

    A response that hits the output ceiling is unsalvageable today: the
    brace-scan hands back the unbalanced remainder, _cheap_repairs only strips
    trailing commas, and json.loads fails. The caller then pays a repair
    round-trip that re-sends the prompt plus the truncated text and truncates
    again, so a long source costs three full model calls and is discarded
    anyway. Notes are the dominant cost in a run — 101 of 115 calls in one
    measured depth-10 run — so throwing away a 90%-complete NotesOut is
    expensive twice over.

    Cut back to the last COMPLETE element rather than closing wherever the
    text stopped, so a half-written key or number is dropped instead of being
    guessed at, then close whatever containers are still open. A model whose
    notes_md was cut mid-sentence keeps the sentence; nothing is invented.
    """
    def scan(s: str):
        stack: list[str] = []
        in_string = escaped = False
        safe: int | None = None
        for i, ch in enumerate(s):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if stack:
                    stack.pop()
                    safe = i + 1
            elif ch == ",":
                safe = i            # cut BEFORE the comma
        return stack, safe

    stack, safe = scan(text)
    if not stack:
        return text                 # already balanced; nothing to do
    if safe is None:
        return text                 # nothing complete to keep
    head = text[:safe].rstrip().rstrip(",")
    still_open, _ = scan(head)
    return head + "".join(reversed(still_open))


def _cheap_repairs(text: str) -> str:
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    # normalize curly quotes that some models emit around keys
    text = text.replace("“", '"').replace("”", '"')
    return text


def extract_json(text: str, *, allow_truncated: bool = False) -> Any:
    """Parse the JSON value out of a model response.

    `allow_truncated` is set ONLY by a caller that knows the response hit the
    output ceiling. Salvage is pure upside there, because the repair
    round-trip re-sends the prompt plus the truncated text at the same budget
    and truncates again. For a merely malformed response it is off, so a
    half-empty object cannot be accepted where a repair would have got the
    real answer.
    """
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
    # Truncation salvage, most content first: keep a half-written string if
    # closing it yields valid JSON, otherwise fall back to dropping the whole
    # incomplete element. For a NotesOut cut mid-sentence the first candidate
    # keeps the sentence, which is the payload; the second keeps the source.
    candidates = [chunk, _cheap_repairs(chunk)]
    if allow_truncated:
        # Most content first: keep a half-written string when closing it
        # yields valid JSON, else drop the whole incomplete element. For a
        # NotesOut cut mid-sentence the first keeps the sentence, which is the
        # payload; the second at least keeps the source.
        candidates += [_cheap_repairs(close_open(chunk)),
                       _cheap_repairs(close_truncated(chunk))]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise LLMJsonError("response contained unparseable JSON")
