"""Strict JSON schemas drive constrained decoding, so their shape matters:
a stray $ref or a missing 'required' entry makes a local server reject the
request and silently drop us back to unconstrained output."""
from app.llm.schema_utils import response_format_for, strict_schema
from app.models import FollowUpsOut, GapOut, NotesOut, PlannerOut

MODELS = [NotesOut, PlannerOut, GapOut, FollowUpsOut]


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_no_refs_survive_inlining():
    schema = strict_schema(NotesOut)
    assert "$defs" not in schema
    assert not any("$ref" in n for n in _walk(schema))


def test_nested_model_is_inlined():
    facts = strict_schema(NotesOut)["properties"]["key_facts"]
    assert facts["type"] == "array"
    item = facts["items"]
    assert set(item["properties"]) == {"claim", "evidence_quote", "confidence"}


def test_every_object_is_strict():
    for model in MODELS:
        for node in _walk(strict_schema(model)):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert sorted(node.get("properties", {})) == node["required"]


def test_response_format_envelope():
    fmt = response_format_for(NotesOut)
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["name"] == "NotesOut"


def test_unsupported_keywords_stripped():
    for model in MODELS:
        for node in _walk(strict_schema(model)):
            assert "default" not in node
            assert "maxLength" not in node
