"""Which extracted facts synthesis is shown, and in what order."""
from __future__ import annotations

from app.research import synthesizer
from app.research.notes import Finding, render_facts


def _facts():
    return [{"claim": f"fact {i}", "confidence": c}
            for i, c in enumerate([5, 9, 2, 7, 8, 3, 10, 6])]


def test_render_facts_can_order_by_confidence_before_limiting():
    plain = render_facts(_facts(), limit=3)
    assert plain.splitlines() == ["- fact 0 (confidence 5/10)",
                                  "- fact 1 (confidence 9/10)",
                                  "- fact 2 (confidence 2/10)"]
    ranked = render_facts(_facts(), limit=3, by_confidence=True)
    assert ranked.splitlines() == ["- fact 6 (confidence 10/10)",
                                   "- fact 1 (confidence 9/10)",
                                   "- fact 4 (confidence 8/10)"]
    # a fact with no confidence sorts last, never crashes
    assert render_facts([{"claim": "x"}, {"claim": "y", "confidence": 1}],
                        by_confidence=True).splitlines()[0] == "- y (confidence 1/10)"


def _finding():
    return Finding(idx=1, url="https://a.example/p", title="T", domain="a.example",
                   published=None, relevance=8, notes_md="notes", summary="s",
                   key_facts=_facts())


def test_note_block_follows_the_module_knobs(monkeypatch):
    base = synthesizer._note_block(_finding())
    assert base.count("- fact") == 6 and "fact 6" not in base    # today's default
    monkeypatch.setattr(synthesizer, "_FACTS_PER_SOURCE", 8)
    monkeypatch.setattr(synthesizer, "_FACTS_BY_CONFIDENCE", True)
    ranked = synthesizer._note_block(_finding())
    assert ranked.count("- fact") == 8
    assert ranked.index("fact 6") < ranked.index("fact 2")     # 10/10 before 2/10
