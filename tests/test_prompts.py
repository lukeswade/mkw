"""Guard against unescaped braces in prompt templates.

Every template is rendered with str.format(), so a literal `{` in an embedded
JSON example silently becomes a replacement field and raises KeyError at
runtime — on every document, in production, with no test coverage. This
caught-it-once bug is now structurally impossible to reintroduce.
"""
import string

import pytest

from app.llm import prompts

TEMPLATES = {
    name: value
    for name, value in vars(prompts).items()
    if name.isupper() and isinstance(value, str) and not name.startswith("_")
}


def _fields(template: str) -> set[str]:
    return {
        field for _lit, field, _spec, _conv in string.Formatter().parse(template)
        if field is not None
    }


def test_templates_were_discovered():
    assert {"PLANNER", "NOTES", "GAP", "SYNTH", "FOLLOWUPS", "ASK"} <= set(TEMPLATES)


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_placeholders_are_plain_identifiers(name):
    """A JSON example leaking into the format string shows up as a field name
    like '\\n  "relevance"' — never a bare identifier."""
    for field in _fields(TEMPLATES[name]):
        assert field.isidentifier(), (
            f"{name} has placeholder {field!r} — this is almost certainly an "
            f"unescaped '{{' in an embedded JSON example. Double the braces."
        )


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_template_formats_without_error(name):
    template = TEMPLATES[name]
    args = {field: f"<{field}>" for field in _fields(template)}
    rendered = template.format(**args)
    assert "<" in rendered or not args
