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


def test_notes_prompt_carries_the_economy_rule():
    """Half of recent runs' doc time went to writing full notes for sources
    that scored ≤2 and were discarded — the rule must stay in the prompt,
    and must stay ABOVE the thin-fallback band (3s keep their notes)."""
    from app.llm import prompts
    assert "ECONOMY RULE" in prompts.NOTES
    assert "2 or lower" in prompts.NOTES


def test_planner_and_gap_demand_facet_spread():
    """Three of four queries in a live run asked for specs, so three of four
    keeps were spec sheets scoring 4/10 against a procedural brief; the one
    procedural query produced the one procedural source (6/10)."""
    from app.llm import prompts
    planner = prompts.PLANNER.replace("\n", " ")
    assert "SPREAD ACROSS THE DIFFERENT FACETS" in planner
    assert "at most one query" in planner            # specs are one facet
    assert "step by step" in planner                 # how-to phrasing required
    gap = prompts.GAP.replace("\n", " ")
    assert "Spread them across DIFFERENT gaps" in gap
