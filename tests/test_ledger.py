"""The fact sheet: what the sources establish about each named thing,
reconciled before the report is written (research/ledger.py).

Fixtures are cut from the 2026-11-15 finance run, where the report listed two
practices as acquisition targets that one source said were already bought.
"""
from __future__ import annotations

import dataclasses

import httpx
import respx

from app.db import Repo, connect
from app.models import LedgerNamesOut, LedgerProfileOut, RunParams
from app.research import ledger, synthesizer
from app.research.ledger import Entry, evidence, mention_re, mine_names, pick_names
from app.research.notes import Finding
from app.research.orchestrator import Orchestrator
from app.research.pipeline import Pipeline
from app.research.progress import ProgressBus
from tests.fake_llm import FakeLLM
from tests.test_pipeline_e2e import SX, article, make_cfg, script, sx_payload, sx_result


def src(idx, title, notes="", facts=(), relevance=5, published="2026-01-10"):
    return Finding(idx=idx, url=f"https://s{idx}.example/x", title=title,
                   domain=f"s{idx}.example", published=published, relevance=relevance,
                   summary="", notes_md=notes,
                   key_facts=[{"claim": c, "evidence_quote": None, "confidence": 9} for c in facts])


def sources() -> list[Finding]:
    return [
        src(32, "U.S. Urology Partners 2026 Company Profile | PitchBook",
            notes="- Oct 20, 2025: Buyout/LBO of New Jersey Urology (Central and South Divisions).\n"
                  "- May 6, 2022: Buyout/LBO of Florida Urology Center.",
            facts=["Major investors in U.S. Urology Partners include Ares Capital and General Atlantic."]),
        src(18, "Urology MSO tracker",
            facts=["General Atlantic recapitalized US Urology Partners in April 2025, exiting NMS Capital.",
                   "Cardinal Health acquired Solaris Health for $1.9 billion, closing in November 2025."]),
        src(27, "Urology M&A dynamics", published="2024-03-01",
            facts=["US Urology Partners (NMS Capital) is the largest US urology MSO with over 500 providers."]),
        src(43, "GI platforms guide",
            facts=["PE-backed gastroenterology MSOs trade at multiples of 6x-13x EBITDA.",
                   "Active platforms include One GI and Gastro Health; for some, none given a valuation."]),
        src(63, "One GI profile",
            facts=["One GI is backed by Webster Equity Partners and Goldman Sachs."]),
        src(49, "Law firm bio",
            facts=["However New Jersey Urology formed a joint venture with J.W. Childs Associates."]),
        src(58, "Gastro Health history",
            facts=["Gastro Health grew from 21 to 212 sites under Audax before the OMERS sale."]),
        src(5, "Medicare outlook", facts=["Medicare site-neutral payments expand in 2026."]),
        src(6, "Medicare payment note",
            facts=["Medicare pays about twice as much in HOPDs, which site-neutral payments would end."]),
    ]


USUP = Entry("U.S. Urology Partners", [], [18, 27, 32])


# --- names ---------------------------------------------------------------------

def test_a_name_is_matched_however_it_is_spelled_and_only_as_whole_words():
    assert mention_re("U.S. Urology Partners").search("General Atlantic recapitalized US Urology Partners")
    # and the other way round: an entry is called whichever spelling is commonest
    assert mention_re("US Urology Partners").search("U.S. Urology Partners 2026 Company Profile")
    assert mention_re("One GI").search("One GI's footprint")
    assert not mention_re("One GI").search("for some, none given a valuation")


def test_mined_names_count_sources_merge_spellings_and_drop_openers():
    mined = dict(mine_names(sources()))
    usup = next(n for n in mined if "Urology Partners" in n)
    assert mined[usup] == 3                              # [18] [27] [32], two spellings
    assert mined["New Jersey Urology"] == 2              # "However New Jersey Urology" too
    assert mined["Medicare"] == 2                        # generous on purpose; the call filters
    assert "Florida Urology Center" not in mined         # one source
    assert not any(n.startswith("However") for n in mined)


def test_a_name_that_also_opens_sentences_is_kept_whole():
    found = dict(mine_names([
        src(1, "a", facts=["One GI is backed by Webster Equity Partners.",
                           "Key Bank financed the recapitalisation."]),
        src(2, "b", facts=["Active platforms include One GI and others.",
                           "The deal was financed by Key Bank last year."])]))
    assert found["One GI"] == 2          # "One" is not an opener
    assert found["Key Bank"] == 2        # "Key" is, but the notes write the name mid-sentence too


def test_articles_and_possessives_do_not_split_a_name():
    found = dict(mine_names([
        src(1, "a", facts=["The Specialty Alliance is Cardinal Health's MSO for urology."]),
        src(2, "b", facts=["Solaris joined the Specialty Alliance under Cardinal Health."]),
        src(3, "c", facts=["Urology America now sits under The Specialty Alliance umbrella."])]))
    assert found["Specialty Alliance"] == 3 and found["Cardinal Health"] == 2


async def test_names_are_ranked_by_sources_and_capped(monkeypatch):
    monkeypatch.setattr(ledger, "_MAX_NAMES", 1)
    llm = FakeLLM({"ledger_names": [{"names": ["Gastro Health", "US Urology Partners"]}]})
    entries = await pick_names(llm, query="q", findings=sources())
    assert len(entries) == 1 and "Urology Partners" in entries[0].name     # 3 sources beat 2


def test_numbered_names_are_mined_whole():
    found = dict(mine_names([
        src(1, "a", facts=["An RTX 3080 Ti runs Qwen3.6-35B-A3B at 24 tokens/second."]),
        src(2, "b", facts=["Qwen3.6-35B-A3B needs offloading on the RTX 3080 Ti with llama.cpp."])]))
    assert found["RTX 3080 Ti"] == 2 and found["Qwen3.6-35B-A3B"] == 2


async def test_the_model_picks_from_the_mined_list_and_cannot_add_to_it():
    llm = FakeLLM({"ledger_names": [{"names": [
        {"name": "US Urology Partners", "aliases": ["U.S. Urology Partners", "Made Up Holdings"]},
        {"name": "Acme Robotics", "aliases": []},                 # in no note
        {"name": "site-neutral payments", "aliases": []},         # in two notes, but a concept, not a mined name
        {"name": "Florida Urology Center", "aliases": []},        # one source: not mined
        "Gastro Health"]}]})                                      # a bare string is a name
    entries = await pick_names(llm, query="q", findings=sources())
    assert [e.name for e in entries][0].endswith("Urology Partners")
    assert {e.name for e in entries} == {entries[0].name, "Gastro Health"}
    assert sorted(entries[0].sources) == [18, 27, 32]
    assert "Made Up Holdings" not in entries[0].aliases


async def test_candidates_are_kept_when_the_naming_call_fails():
    llm = FakeLLM({"ledger_names": [RuntimeError("down")]})
    entries = await pick_names(llm, query="q", findings=sources(),
                               also=["Gastro Health", "Nobody Mentions This"])
    assert [e.name for e in entries] == ["Gastro Health"]


# --- evidence ------------------------------------------------------------------

def test_evidence_is_every_statement_naming_the_thing_with_its_source():
    got = evidence(USUP, sources())
    by = {i: [t for j, t in got if j == i] for i, _ in got}
    assert set(by) == {18, 27, 32}
    assert any("exiting NMS Capital" in t for t in by[18])
    assert not any("Solaris Health" in t for t in by[18])        # the same source, another subject


def test_a_page_titled_for_the_thing_gives_all_its_lines():
    # "Buyout/LBO of New Jersey Urology" names its buyer only in the page title
    got = [t for i, t in evidence(USUP, sources()) if i == 32]
    assert any("New Jersey Urology" in t for t in got) and any("Florida Urology Center" in t for t in got)


def test_one_source_cannot_fill_the_evidence(monkeypatch):
    monkeypatch.setattr(ledger, "_PER_SOURCE", 2)
    chatty = src(1, "notes", facts=[f"Gastro Health fact number {n} of many." for n in range(9)])
    assert len(evidence(Entry("Gastro Health"), [chatty])) == 2


# --- reconciled lines -----------------------------------------------------------

PICK = {"names": [{"name": "US Urology Partners", "aliases": ["U.S. Urology Partners"]},
                  {"name": "Gastro Health", "aliases": []}]}
GOOD = "U.S. Urology Partners has been backed by General Atlantic since April 2025, when NMS Capital exited."


def profile(*lines):
    def reply(messages):
        if "Gastro Health" in messages[-1]["content"].split("Everything the kept sources say about")[1][:40]:
            return {"lines": [{"text": "Gastro Health grew from 21 to 212 sites under Audax.", "sources": [58]}]}
        return {"lines": list(lines)}
    return reply


async def build(llm, **kw):
    return await ledger.build(llm, query=kw.pop("query", "Which urology platforms could we buy?"),
                              today="2026-11-15", findings=sources(), **kw)


async def test_a_line_is_kept_with_its_sources_and_markers_are_stripped():
    llm = FakeLLM({"ledger_names": [PICK],
                   "ledger": [profile({"text": GOOD + " [18]", "sources": [18, "[27]"]})]})
    entries = await build(llm)
    usup = next(e for e in entries if "Urology Partners" in e.name)
    assert usup.lines == [(GOOD, [18, 27])]
    block = ledger.render(entries)
    assert f"- {GOOD} [18][27]" in block and block.startswith(usup.name)


async def test_a_line_with_a_figure_its_sources_do_not_state_is_dropped():
    llm = FakeLLM({"ledger_names": [PICK], "ledger": [profile(
        {"text": "U.S. Urology Partners was valued at $4.2 billion in 2025.", "sources": [18]},
        {"text": GOOD, "sources": [18]})]})
    usup = next(e for e in await build(llm) if "Urology Partners" in e.name)
    assert [t for t, _ in usup.lines] == [GOOD]


async def test_a_line_resting_on_sources_that_were_not_shown_is_dropped():
    llm = FakeLLM({"ledger_names": [PICK], "ledger": [profile(
        {"text": "U.S. Urology Partners is backed by Webster Equity Partners.", "sources": [63]},
        {"text": GOOD, "sources": [18, 63]})]})
    usup = next(e for e in await build(llm) if "Urology Partners" in e.name)
    assert usup.lines == [(GOOD, [18])]           # [63] never mentions it, so it was never shown


async def test_a_line_that_does_not_name_the_thing_is_dropped():
    llm = FakeLLM({"ledger_names": [PICK], "ledger": [profile(
        {"text": "Solaris Health was acquired for $1.9 billion in November 2025.", "sources": [18]},
        {"text": GOOD, "sources": [18]})]})
    usup = next(e for e in await build(llm) if "Urology Partners" in e.name)
    assert [t for t, _ in usup.lines] == [GOOD]


async def test_too_few_sources_or_any_failure_means_no_fact_sheet(monkeypatch):
    quiet = FakeLLM({})
    assert await ledger.build(quiet, query="q", today="2026-11-15", findings=sources()[:5]) == []
    assert quiet.total_calls == 0

    def boom(*a, **k):
        raise ValueError("bug")
    monkeypatch.setattr(ledger, "mine_names", boom)
    assert await build(FakeLLM({})) == []


async def test_a_name_whose_call_fails_or_yields_nothing_is_left_out():
    llm = FakeLLM({"ledger_names": [PICK], "ledger": [profile()]})     # no lines for USUP
    assert [e.name for e in await build(llm)] == ["Gastro Health"]


def test_model_output_is_read_forgivingly():
    assert LedgerNamesOut.model_validate({"names": "nope"}).names == []
    assert LedgerProfileOut.model_validate({"lines": [{"text": "x", "sources": ["[3]", "n/a", 4]}, "junk"]}
                                           ).lines[0].sources == [3, 4]


# --- in synthesis --------------------------------------------------------------

def _prompts(seen):
    def reply(messages):
        seen.append(messages[-1]["content"])
        return "# T\n\n## TL;DR\n\n- Done [18].\n\n## Open questions\n\n- None.\n"
    return reply


async def _synth(llm, **kw):
    return await synthesizer.synthesize(
        llm, query="Which urology platforms could we buy?", title="T", brief="b",
        recency_desc="any", today="2026-11-15", state_md="", findings=sources(), **kw)


async def test_synthesis_is_given_the_fact_sheet_only_when_it_is_switched_on():
    seen: list[str] = []
    llm = FakeLLM({"candidates": [{"candidates": []}], "synth": [_prompts(seen)],
                   "ledger_names": [PICK],
                   "ledger": [profile({"text": GOOD, "sources": [18]})]})
    await _synth(llm, fact_ledger=True)
    assert "Hold to these lines" in seen[0] and f"- {GOOD} [18]" in seen[0]

    seen.clear()
    off = FakeLLM({"candidates": [{"candidates": []}], "synth": [_prompts(seen)]})
    await _synth(off)
    assert "Hold to these lines" not in seen[0] and off.calls["ledger_names"] == 0


async def test_the_second_pass_is_held_to_the_same_fact_sheet():
    seen: list[str] = []
    strong = [dataclasses.replace(f, relevance=8) for f in sources()]
    llm = FakeLLM({"candidates": [{"candidates": []}], "synth": [_prompts(seen)],
                   "ledger_names": [PICK],
                   "ledger": [profile({"text": GOOD, "sources": [18]})]})
    await synthesizer.synthesize(
        llm, query="Which urology platforms could we buy?", title="T", brief="b",
        recency_desc="any", today="2026-11-15", state_md="", findings=strong, fact_ledger=True)
    assert len(seen) == 2 and "revising a research overview" in seen[1]
    assert f"- {GOOD} [18]" in seen[1]


@respx.mock
async def test_runs_and_resynthesis_pass_the_setting_through(data_dir, monkeypatch):
    cfg = dataclasses.replace(make_cfg(data_dir), fact_ledger=True)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://example-a.com/article", "Article A")])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))
    passed: list[bool] = []
    real = synthesizer.synthesize

    async def spy(*a, **kw):
        passed.append(kw.get("fact_ledger"))
        return await real(*a, **kw)
    monkeypatch.setattr(synthesizer, "synthesize", spy)
    repo = Repo(connect(cfg.db_path))
    llm = lambda: FakeLLM(script([{"state_md": "s", "saturated": True, "next_queries": []}]))  # noqa: E731
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(), llm_factory=llm)
    run_id = orch.enqueue(RunParams(query="widgets", depth=1, recency="all", origin="cli"))
    await orch.execute_now(run_id)
    await Pipeline(cfg, repo, ProgressBus(), llm_factory=llm).resynthesize(run_id)
    off = dataclasses.replace(cfg, fact_ledger=False)
    await Pipeline(off, repo, ProgressBus(), llm_factory=llm).resynthesize(run_id)
    assert passed == [True, True, False]
