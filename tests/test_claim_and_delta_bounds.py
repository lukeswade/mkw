"""Bounds that clamp instead of abort."""
from __future__ import annotations

from app.models import ClaimsOut
from app.research import synthesizer


def test_one_long_claim_no_longer_aborts_the_check():
    long = "x" * 2000
    out = ClaimsOut.model_validate({"claims": [
        {"text": long, "importance": 9},
        {"text": "   ", "importance": 3},
        "a bare string claim",
        42,
        {"text": "fine", "importance": 5, "checkable": False},
    ]})
    texts = [c.text for c in out.claims]
    assert texts == ["x" * 600, "a bare string claim", "fine"]
    assert out.claims[0].importance == 9
    assert out.claims[-1].checkable is False


def test_claims_list_is_capped_not_rejected():
    out = ClaimsOut.model_validate({"claims": [{"text": f"c{i}"} for i in range(200)]})
    assert len(out.claims) == 80


def test_a_score_on_the_percent_scale_is_read_as_one():
    """2026-10-07..08: the claim judge answered confidence 95 six times; the
    field had no validator, so each cost a repair call. The schema the server
    enforces carries no range at all (schema_utils strips it)."""
    from app.models import Claim, Fact, NotesOut, VerdictOut, score_10
    assert VerdictOut.model_validate(
        {"verdict": "supported", "confidence": 95}).confidence == 9
    assert VerdictOut.model_validate({"confidence": "7"}).confidence == 7
    assert VerdictOut.model_validate({"confidence": "high"}).confidence == 0
    # clamping alone used to read a percentage as a perfect score
    notes = NotesOut.model_validate({"relevance": 70, "part_relevance": 40,
                                     "key_facts": [{"claim": "c", "confidence": 85}]})
    assert (notes.relevance, notes.part_relevance, notes.key_facts[0].confidence) == (7, 4, 8)
    assert NotesOut.model_validate({"relevance": 7}).relevance == 7
    assert NotesOut.model_validate({"relevance": 400}).relevance == 10
    assert NotesOut.model_validate({"relevance": -3}).relevance == 0
    assert Claim.model_validate({"text": "t", "importance": 80}).importance == 8
    assert Fact.model_validate({"claim": "c", "confidence": None}).confidence == 5
    assert score_10(float("nan"), 3) == 3


def test_an_over_long_field_is_clipped_not_fatal():
    from app.models import GapOut, MatrixOut, NotesOut, PlannerOut, VerdictOut
    v = VerdictOut.model_validate({"reasoning": "r" * 2000, "quote": "q" * 900,
                                   "sources": list(range(30))})
    assert (len(v.reasoning), len(v.quote), len(v.sources)) == (900, 400, 10)
    # one run-on claim used to cost the whole page its notes
    n = NotesOut.model_validate({"relevance": 6, "key_facts": [{"claim": "c" * 1200}]})
    assert len(n.key_facts[0].claim) == 500
    p = PlannerOut.model_validate({"title": "t", "subqueries": [f"q{i}" for i in range(15)],
                                   "keywords": [f"k{i}" for i in range(30)],
                                   "query_scopes": ["web"] * 15})
    assert (len(p.subqueries), len(p.keywords), len(p.query_scopes)) == (12, 20, 12)
    g = GapOut.model_validate({"next_queries": [f"q{i}" for i in range(15)],
                               "keywords": [f"k{i}" for i in range(25)]})
    assert (len(g.next_queries), len(g.keywords)) == (12, 20)
    m = MatrixOut.model_validate({"entities": [f"e{i}" for i in range(10)],
                                  "cells": [{"entity": "e" * 300, "value": "v" * 500,
                                             "sources": list(range(20))}]})
    assert len(m.entities) == 8
    assert (len(m.cells[0].entity), len(m.cells[0].value), len(m.cells[0].sources)) == (140, 300, 12)


def _compose(previous: str) -> str:
    return synthesizer._compose_prompt(
        ["- note [1]"], query="q", title="t", brief="b", recency_desc="any",
        today="2026-09-20", state_md="", findings=[], groups=[], candidates=[],
        uncovered_facets=None, premises=None, previous_overview=previous,
        deliverables=None)


def test_delta_prompt_marks_a_clipped_parent_overview():
    short = "## Earlier\n" + "word " * 500
    assert synthesizer._PREVIOUS_OVERVIEW_CLIP_NOTE not in _compose(short)
    long = "## Earlier\n" + "word " * 20_000
    prompt = _compose(long)
    assert synthesizer._PREVIOUS_OVERVIEW_CLIP_NOTE in prompt
    shown = prompt.split(synthesizer._PREVIOUS_OVERVIEW_CLIP_NOTE)[0]
    assert long[:24_000] in shown
    assert synthesizer._PREVIOUS_OVERVIEW_CHARS >= 24_000
