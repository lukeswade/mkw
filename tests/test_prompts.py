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


def test_gap_queries_must_stay_searchable():
    """Depth-10 round 3 searched "valve cover torque 7 ft-lbs vs 11 ft-lbs
    factory manual" and similar, matched nothing useful, and burned 14 full
    document analyses for zero keeps."""
    from app.llm import prompts
    gap = prompts.GAP.replace("\n", " ")
    assert "KEEP EACH QUERY SHORT AND SEARCHABLE" in gap
    assert "A narrow gap still needs a broad query" in gap


def test_triage_sees_the_round_queries():
    """Triage dropped a Tundra 4.7L thread and an NGK IFR6A11 discussion —
    both answering queries the round was actively running."""
    from app.llm import prompts
    assert "{queries}" in prompts.TRIAGE
    t = prompts.TRIAGE.replace("\n", " ")
    assert "answers ANY of these is worth keeping" in t


async def test_the_anchored_planner_variant_is_opt_in(data_dir):
    """Both A/B arms run identical code; only the planner's instructions
    differ, and only when asked for by PLANNER_VARIANT=anchored."""
    from app.models import PlannerOut
    from app.research import planner

    class Capture:
        def __init__(self): self.prompts = []
        async def chat_json(self, kind, messages, schema, **kw):
            self.prompts.append(messages[0]["content"])
            return PlannerOut(title="t", brief="b", subqueries=["q1", "q2"], keywords=["k"])

    for variant, expected in (("default", False), ("anchored", True)):
        llm = Capture()
        out = await planner.plan(llm, query="Jailbreak a Kindle", recency_desc="all time",
                                 today="2026-09-02", breadth=2, variant=variant)
        assert out.subqueries == ["q1", "q2"]
        assert ("Anchoring rules" in llm.prompts[0]) is expected, variant
        assert "Research question: Jailbreak a Kindle" in llm.prompts[0]


def test_the_anchored_planner_is_now_the_default_with_a_kill_switch(data_dir, monkeypatch):
    from app.config import Settings, load_settings
    assert Settings(data_dir=str(data_dir)).planner_variant == "anchored"
    monkeypatch.setenv("DATA_DIR", str(data_dir)); monkeypatch.setenv("PLANNER_VARIANT", "default")
    assert load_settings(str(data_dir)).planner_variant == "default"


async def test_the_anchored_gap_variant_is_opt_in_and_reaches_the_retry(data_dir):
    from app.models import GapOut
    from app.research import gap

    class Capture:
        def __init__(self): self.prompts = []
        async def chat_json(self, kind, messages, schema, **kw):
            self.prompts.append(messages[0]["content"])
            # first answer: not saturated, no queries -> forces the stern re-ask
            return GapOut(state_md="s", saturated=False, next_queries=[] if len(self.prompts) == 1 else ["Cinque trackball ZMK"])

    for variant, expected in (("default", False), ("anchored", True)):
        llm = Capture()
        out = await gap.analyze(llm, query="DIY trackball", brief="b", recency_desc="all time",
                                round_no=1, depth=2, breadth=3, state_md="", new_findings=[],
                                searched=["x"], variant=variant)
        assert len(llm.prompts) == 2                                   # the re-ask happened
        assert all(("Anchoring rules" in p) is expected for p in llm.prompts), variant
        assert out.next_queries == ["Cinque trackball ZMK"]


def test_instructions_first_keeps_every_placeholder_and_moves_the_rubric_ahead_of_the_document():
    import re
    from app.llm import prompts
    a, b = prompts.NOTES, prompts.NOTES_INSTRUCTIONS_FIRST
    assert set(re.findall(r"\{(\w+)\}", a)) == set(re.findall(r"\{(\w+)\}", b))
    assert b.index("Produce a JSON object") < b.index("SOURCE DOCUMENT (untrusted")
    assert a.index("Produce a JSON object") > a.index("SOURCE DOCUMENT (untrusted")
    # the shared prefix before the document is now long enough to be worth caching
    assert b.index("SOURCE DOCUMENT") > 1500 and a.index("SOURCE DOCUMENT") < 400


async def test_a_borderline_score_is_rechecked_once_and_averaged(data_dir):
    from app.models import NotesOut
    from app.research.notes import take_notes

    class Capture:
        def __init__(self, scores): self.scores = list(scores); self.calls = 0
        async def chat_json(self, kind, messages, schema, **kw):
            self.calls += 1
            return NotesOut(relevance=self.scores.pop(0), summary=f"s{self.calls}", notes_md="n", key_facts=[])

    kw = dict(brief="b", recency_desc="all time", today="2026-09-03", url="u", title="t",
              detected_date=None, text="page text")
    llm = Capture([4, 8]); out = await take_notes(llm, recheck=True, **kw)
    assert llm.calls == 2 and out.relevance == 6 and out.summary == "s2"      # averaged, notes from the higher answer
    llm = Capture([4, 8]); out = await take_notes(llm, recheck=False, **kw)
    assert llm.calls == 1 and out.relevance == 4
    llm = Capture([7, 9]); out = await take_notes(llm, recheck=True, **kw)
    assert llm.calls == 1 and out.relevance == 7                               # not borderline: no second look
