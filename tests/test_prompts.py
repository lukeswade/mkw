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


def test_instructions_first_is_the_default_notes_layout(data_dir):
    from app.config import Settings
    assert Settings(data_dir=str(data_dir)).notes_order == "instructions_first"


# ---- what the asker said about the shape of the answer ----------------------

def test_deliverables_are_capped_and_cleaned_in_code():
    """Bound in code, not in the prompt: one runaway instruction must not be
    able to rewrite the synthesis prompt."""
    from app.models import PlannerOut
    p = PlannerOut(
        title="t", brief="b", facets=["a"], subqueries=["q"],
        deliverables=["  include a comparison table  ", "", None,
                      "x" * 500, "two", "three", "four", "five"])
    assert p.deliverables[0] == "include a comparison table"
    assert len(p.deliverables) == 4          # list capped
    assert max(len(d) for d in p.deliverables) <= 200   # each clamped
    assert "" not in p.deliverables


def test_a_question_with_no_formatting_instruction_has_no_deliverables():
    from app.models import PlannerOut
    p = PlannerOut(title="t", brief="b", facets=["a"], subqueries=["q"])
    assert p.deliverables == []


async def test_synthesis_is_told_what_shape_the_asker_wanted():
    """2026-09-11: a question asking for a comprehensive comparison table got
    a good document with no table in it. The brief — the only thing that
    reached synthesis — describes what to research, never what to produce."""
    from app.research.notes import Finding
    from app.research.synthesizer import synthesize
    from tests.fake_llm import FakeLLM
    seen = []

    def capture(messages):
        seen.append(messages[-1]["content"])
        return "# Doc\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nBody [1]."

    f = Finding(idx=1, url="https://a.com/x", title="T", domain="a.com",
                published=None, relevance=8, summary="s", notes_md="notes")
    await synthesize(FakeLLM({"synth": [capture]}), query="q", title="T",
                     brief="b", recency_desc="any", today="2026-09-11",
                     state_md="", findings=[f],
                     deliverables=["Include a comprehensive comparison table"])
    assert "comprehensive comparison table" in seen[0]
    # and it comes last, so it has the final say over the generic instructions
    assert seen[0].rindex("comprehensive comparison table") > seen[0].rindex("Research question")


async def test_no_deliverables_block_when_the_asker_named_none():
    from app.research.notes import Finding
    from app.research.synthesizer import synthesize
    from tests.fake_llm import FakeLLM
    seen = []

    def capture(messages):
        seen.append(messages[-1]["content"])
        return "# Doc\n\nBody [1]."

    f = Finding(idx=1, url="https://a.com/x", title="T", domain="a.com",
                published=None, relevance=8, summary="s", notes_md="notes")
    await synthesize(FakeLLM({"synth": [capture]}), query="q", title="T",
                     brief="b", recency_desc="any", today="2026-09-11",
                     state_md="", findings=[f])
    assert "shape" not in seen[0].lower().split("research question")[0]
    assert "the asker also said" not in seen[0].lower()


# ---- document length scales with how much was gathered ----------------------

def test_target_words_scales_and_clamps():
    """Measured 2026-09-11: documents came out 1000-2100 words whether a run
    kept 20 sources or 89, so a deep run's extra sources had nowhere to go —
    66 sources cited 36%, 23 sources cited 74%."""
    from app.research.synthesizer import (target_words, _MIN_TARGET_WORDS,
                                          _MAX_TARGET_WORDS)
    assert target_words(6) == _MIN_TARGET_WORDS        # floor: never padded
    assert target_words(200) == _MAX_TARGET_WORDS      # ceiling: never a book
    assert target_words(66) == 3300
    assert target_words(29) < target_words(66) < target_words(89)
    assert target_words(0) == _MIN_TARGET_WORDS        # no findings, no crash


async def test_synthesis_is_given_a_length_target_from_the_source_count():
    from app.research.notes import Finding
    from app.research.synthesizer import synthesize, target_words
    from tests.fake_llm import FakeLLM
    seen = []

    def capture(messages):
        seen.append(messages[-1]["content"])
        return "# Doc\n\nBody [1]."

    fs = [Finding(idx=i, url=f"https://a.com/{i}", title=f"T{i}", domain="a.com",
                  published=None, relevance=7, summary="s", notes_md="notes")
          for i in range(1, 41)]
    await synthesize(FakeLLM({"synth": [capture]}), query="q", title="T",
                     brief="b", recency_desc="any", today="2026-09-11",
                     state_md="", findings=fs)
    assert f"{target_words(40):,}" in seen[0]
    assert "40 sources" in seen[0]
    # Measured 2026-09-11: a hedged target ("write the shorter document if the
    # sources do not carry it") was inert — asked for 3,300 words, got 1,480.
    # Removing the escape hatch moved it to 2,197 words and 38 of 66 cited
    # against 33. The instruction has to be a floor, with the anti-padding
    # guards kept so length comes from coverage.
    assert "at least" in seen[0]
    assert "not padding" in seen[0]
    assert "never cite a source you did not draw on" in seen[0]


def test_the_output_cap_can_never_be_tighter_than_the_length_asked_for():
    """A ceiling below the target would truncate the document the prompt just
    requested."""
    from app.research.synthesizer import target_words
    for n in (1, 6, 29, 66, 89, 150, 400):
        want = target_words(n)
        max_out = min(16_000, max(8_000, int(want * 2.5)))
        assert max_out >= want * 1.5, (n, want, max_out)


# ---- a section per part of the question --------------------------------------

def _fs(n_per_facet: dict[str, int]):
    from app.research.notes import Finding
    out, i = [], 0
    for facet, n in n_per_facet.items():
        for _ in range(n):
            i += 1
            out.append(Finding(idx=i, url=f"https://a.com/{i}", title=f"T{i}",
                               domain="a.com", published=None, relevance=7,
                               summary="s", notes_md="notes", query=facet))
    return out


async def _prompt_for(findings, **kw):
    from app.research.synthesizer import synthesize
    from tests.fake_llm import FakeLLM
    seen = []

    def capture(messages):
        seen.append(messages[-1]["content"])
        return "# Doc\n\nBody [1]."

    await synthesize(FakeLLM({"synth": [capture]}), query="q", title="T",
                     brief="b", recency_desc="any", today="2026-09-11",
                     state_md="", findings=findings, **kw)
    return seen[0]


async def test_each_part_of_the_question_is_asked_for_its_own_section():
    """Models follow structure more reliably than word counts, and the parts
    with two sources are the ones that vanish into a passing sentence."""
    fs = _fs({"agent interface": 37, "local hosting": 12, "manual editing": 2})
    p = await _prompt_for(fs, facet_of={"agent interface": "agent interface",
                                        "local hosting": "local hosting",
                                        "manual editing": "manual editing"})
    assert "agent interface — 37 sources" in p
    assert "manual editing — 2 sources" in p
    # ordered by weight, so the biggest part is obviously the longest section
    assert p.index("agent interface — 37") < p.index("manual editing — 2")
    assert "section of its own" in p


async def test_a_part_synthesis_is_forbidden_to_write_about_is_never_listed():
    """uncovered_facets comes from the credit-gated counter and these groups
    from attribution, so a part can appear in both. Naming it here while
    SYNTH_COVERAGE_BLOCK forbids a section on it would put two contradictory
    instructions in one prompt."""
    # three parts so the block still fires after one is subtracted
    fs = _fs({"agent interface": 9, "local hosting": 4, "manual editing": 3})
    p = await _prompt_for(fs, facet_of={"agent interface": "agent interface",
                                        "local hosting": "local hosting",
                                        "manual editing": "manual editing"},
                          uncovered_facets=["manual editing"])
    assert "agent interface — 9 sources" in p
    assert "local hosting — 4 sources" in p
    assert "manual editing — 3 sources" not in p     # never asked for
    assert "manual editing" in p                     # still gagged by coverage
    assert "Do not write a section on any of them" in p


async def test_no_structure_block_when_there_is_only_one_part():
    fs = _fs({"only thing": 5})
    p = await _prompt_for(fs, facet_of={"only thing": "only thing"})
    assert "section of its own" not in p
