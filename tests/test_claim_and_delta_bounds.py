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
