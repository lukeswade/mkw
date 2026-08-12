"""Fact rendering must never leak dict reprs or literal escape sequences into
prompts or generated markdown — both shipped as silent regressions once."""
from app.research.gap import render_round_findings
from app.research.notes import Finding, finding_markdown, render_facts
from app.research.synthesizer import _note_block

FACTS = [
    {"claim": "Energy density reached 400 Wh/kg.",
     "evidence_quote": "cells exceeded 400 Wh/kg in testing", "confidence": 9},
    {"claim": "Yield remains the blocker.", "evidence_quote": None,
     "confidence": 6},
]


def _finding(**kw) -> Finding:
    base = dict(idx=1, url="https://example.com/a", title="A Source",
                domain="example.com", published="2026-05-01", relevance=8,
                summary="A summary.", notes_md="Some notes.", key_facts=FACTS)
    base.update(kw)
    return Finding(**base)


def test_render_facts_produces_real_newlines():
    out = render_facts(FACTS)
    assert "\\n" not in out            # literal backslash-n, the old bug
    assert out.count("\n") >= 2
    assert "Energy density reached 400 Wh/kg. (confidence 9/10)" in out
    assert '> "cells exceeded 400 Wh/kg in testing"' in out


def test_render_facts_options():
    assert '"' not in render_facts(FACTS, quotes=False)
    assert len(render_facts(FACTS, limit=1).splitlines()) == 2  # claim + quote
    assert render_facts(FACTS, indent="  ").startswith("  - ")
    assert render_facts([]) == ""
    assert render_facts([{"claim": "  "}]) == ""  # blank claims dropped


def test_finding_markdown_has_no_escape_artifacts():
    md = finding_markdown(_finding())
    assert "\\n" not in md
    assert "{'claim'" not in md
    assert "## Key facts" in md
    body = md.split("## Key facts", 1)[1]
    assert body.count("\n- ") == 2  # two facts on their own lines


def test_finding_markdown_without_facts():
    md = finding_markdown(_finding(key_facts=[]))
    assert "_none extracted_" in md


def test_gap_prompt_gets_prose_not_dicts():
    rendered = render_round_findings([_finding()])
    assert "{'claim'" not in rendered
    assert "confidence" in rendered
    assert "Energy density reached 400 Wh/kg." in rendered
    # quotes are omitted from the gap prompt to save context budget
    assert "cells exceeded 400 Wh/kg in testing" not in rendered


def test_synthesis_receives_verbatim_evidence():
    block = _note_block(_finding())
    assert "{'claim'" not in block
    assert '> "cells exceeded 400 Wh/kg in testing"' in block
    assert "https://example.com/a" in block
