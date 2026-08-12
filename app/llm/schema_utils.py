"""Turn a pydantic model into a strict JSON schema an LLM server can enforce.

OpenAI-compatible servers (including LM Studio / MLX and recent llama.cpp)
accept `response_format={"type":"json_schema", ...,"strict":true}` and
constrain decoding to it, which makes malformed JSON structurally impossible
instead of merely unlikely. Strict mode has rules pydantic does not follow by
default, so the raw schema needs adjusting:

  * every property must appear in "required" (optionality is expressed by
    allowing null, not by omitting the key)
  * every object must set "additionalProperties": false
  * $ref/$defs indirection is inlined — plenty of local servers mishandle it
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# Keywords a strict validator rejects or a constrained decoder cannot honour.
_STRIP_KEYS = {"default", "examples", "$comment", "format",
               "minLength", "maxLength", "minItems", "maxItems",
               "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}


def _inline(node: Any, defs: dict) -> Any:
    if isinstance(node, list):
        return [_inline(item, defs) for item in node]
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        ref = node["$ref"].rsplit("/", 1)[-1]
        target = defs.get(ref, {})
        merged = {k: v for k, v in node.items() if k != "$ref"}
        return _inline({**target, **merged}, defs)

    out = {k: _inline(v, defs) for k, v in node.items() if k not in _STRIP_KEYS}

    if out.get("type") == "object" or "properties" in out:
        out["type"] = "object"
        out["additionalProperties"] = False
        out["required"] = sorted(out.get("properties", {}))
    return out


def strict_schema(model: type[BaseModel]) -> dict:
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})
    schema = _inline(raw, defs)
    schema.pop("title", None)
    return schema


def response_format_for(model: type[BaseModel]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": strict_schema(model),
        },
    }
