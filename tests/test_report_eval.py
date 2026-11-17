"""The report eval's own measures (scripts/eval/report_eval.py).

Its numbers decide whether a change to synthesis ships, so each measure is
pinned on the untidy cases it was built from: the 2026-11-15 finance report
that named two already-acquired practices as acquisition targets.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest
import respx

from app.research.notes import Finding
from tests.fake_llm import FakeLLM

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval" / "report_eval.py"


def _load():
    # by path: scripts/ is not a package. Registered before exec, or the
    # module's dataclasses cannot resolve their own annotations.
    spec = importlib.util.spec_from_file_location("report_eval", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["report_eval"] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _load()

REPORT = """# Targets

## TL;DR

*   **Top Acquisition Targets (Add-ons):** Smaller practices are ideal. Examples include **Florida Urology Center** and **New Jersey Urology** [2][44].

## Platform Candidates

### 1. U.S. Urology Partners (NMS Capital / General Atlantic)

*   **Assets:** Includes New Jersey Urology and Florida Urology Center [32].
*   **Profile:** Backed by NMS Capital, with a growth investment from General Atlantic in April 2025 [1][18].

## Add-On Candidates

*   **Examples:** Florida Urology Center, Arizona Urology Specialists.

## Researched but not used

- [9] New Jersey Urology is a standalone add-on target.
"""

OWNED = {"id": "owned", "kind": "substantive", "terms": [r"Florida Urology Center"],
         "violation": [{"all": [r"Florida Urology Center|New Jersey Urology",
                                r"add-on|standalone|target|candidate"],
                        "none": [r"U\.?S\.? Urology Partners|owned by"]}]}


def src(idx, notes, relevance=8, query=""):
    return Finding(idx=idx, url=f"https://s{idx}.example/x", title=f"Source {idx}",
                   domain=f"s{idx}.example", published="2026-10-10", relevance=relevance,
                   summary="", notes_md=notes, query=query)


# --- reading a report ----------------------------------------------------------

def test_blocks_carry_their_headings_and_stop_at_the_closing_lists():
    found = ev.blocks(REPORT)
    assets = next(b for b in found if b.text.startswith("*   **Assets"))
    assert assets.section == "Platform Candidates"
    assert "U.S. Urology Partners" in assets.headings
    assert not any("[9]" in b.text for b in found)        # the "not used" list is not the report


def test_table_separators_and_headings_are_not_blocks():
    found = ev.blocks("## A\n\n| x | y |\n|---|:--|\n| One GI | 6-13x |\n\n---\n\n### B\n\ntext\n")
    assert [b.text for b in found] == ["| x | y |", "| One GI | 6-13x |", "text"]
    assert found[-1].headings == "A > B"


def test_a_nested_bullet_and_a_bold_label_carry_their_subject_down():
    # "Performance: ~24 tok/s" was about the model named one bullet up
    md = ("## Coding\n\n3.  **Qwen3-Coder 30B-A3B (MoE):**\n"
          "    *   **Performance:** ~22-24 tok/s on a Ryzen 5600X [81].\n"
          "4.  **Qwen3.5-9B:**\n    *   **Performance:** 40 tok/s [1].\n\n"
          "**Alternative: Gemma 4 12B**\nIt runs fully on the GPU at 30 tok/s [2].\n\n"
          "### Next\n\nUnrelated paragraph.\n")
    by = {b.text: b.headings for b in ev.blocks(md)}
    assert "Qwen3-Coder" in by["*   **Performance:** ~22-24 tok/s on a Ryzen 5600X [81]."]
    assert "Qwen3-Coder" not in by["*   **Performance:** 40 tok/s [1]."]       # its own parent only
    assert by["It runs fully on the GPU at 30 tok/s [2]."].endswith("Alternative: Gemma 4 12B")
    assert by["Unrelated paragraph."] == "Coding > Next"                        # a heading ends both


# --- known errors ---------------------------------------------------------------

def test_an_error_is_read_with_the_headings_above_it():
    status, evidence = ev.fact_status(OWNED, REPORT)
    assert status == "present" and evidence.startswith("*   **Top Acquisition Targets")
    # the same names under their owner's heading are not the error
    fixed = REPORT.replace("Examples include **Florida Urology Center** and **New Jersey Urology**",
                           "U.S. Urology Partners already owns Florida Urology Center")
    fixed = fixed.replace("*   **Examples:** Florida Urology Center, Arizona",
                          "*   **Examples:** Arizona")
    assert ev.fact_status(OWNED, fixed)[0] == "clear"


def test_a_report_that_never_mentions_the_subject_is_absent_not_clear():
    assert ev.fact_status(OWNED, "# T\n\n## A\n\nNothing about urology groups.\n")[0] == "absent"


def test_every_condition_must_be_met_somewhere():
    fact = {"id": "two", "kind": "substantive", "terms": [r"One GI"],
            "violation": [{"all": [r"One GI", r"mid-teens"]}, {"all": [r"One GI", r"6-13x"]}]}
    one = "## T\n\n### One GI\n\nLikely trading at mid-teens multiples.\n"
    assert ev.fact_status(fact, one)[0] == "clear"
    assert ev.fact_status(fact, one + "\n| One GI | 6-13x EBITDA |\n")[0] == "present"


def test_a_wrong_citation_needs_a_citation_to_be_wrong_about():
    fact = {"id": "cite", "kind": "citation", "terms": [r"3\.2 ?%"],
            "violation": [{"scope": "sentence", "all": [r"3\.2 ?%"], "needs_cite": [7, 2]}]}
    assert ev.fact_status(fact, "## A\n\nWhisper reaches 3.2% WER [18].\n")[0] == "present"
    assert ev.fact_status(fact, "## A\n\nWhisper reaches 3.2% WER [2, 18].\n")[0] == "clear"
    assert ev.fact_status(fact, "## A\n\nWhisper reaches 3.2% WER.\n")[0] == "clear"


def test_a_sentence_can_take_its_subject_from_the_bullet_around_it():
    fact = {"id": "lw", "kind": "citation", "terms": [r"Linkwarden"],
            "violation": [{"scope": "sentence", "context": [r"Linkwarden"],
                           "all": [r"recommended[^.]{0,40}4 ?GB"], "needs_cite": [33]}]}
    bullet = ("## A\n\n*   **Linkwarden**: needs PostgreSQL [32] [33]. "
              "The recommended deployment is 4GB RAM [35].\n")
    assert ev.fact_status(fact, bullet)[0] == "present"   # [33] sits in another sentence
    assert ev.fact_status(fact, bullet.replace("Linkwarden", "Karakeep")
                          + "\nLinkwarden is fine.\n")[0] == "clear"


def test_a_whole_word_exclusion_does_not_match_inside_another_word():
    # "dated" once matched inside "consolidated" and hid the error
    fact = {"id": "stale", "kind": "substantive", "terms": [r"3,600"],
            "violation": [{"scope": "sentence", "all": [r"3,600", r"2025"],
                           "none": [r"\b(2021|dated)\b"]}]}
    md = "## A\n\nA shortage of 3,600 urologists by 2025 lifts consolidated platforms [48].\n"
    assert ev.fact_status(fact, md)[0] == "present"


# --- measures -------------------------------------------------------------------

def test_a_restatement_in_a_later_section_counts_once_and_summaries_do_not():
    md = ("## TL;DR\n\nPlatform acquisitions in urology trade at 10-14x EV/EBITDA while add-ons trade lower.\n\n"
          "## Market\n\nPlatform acquisitions in urology trade at 10-14x EV/EBITDA, while add-on targets trade at 5-9x [2].\n\n"
          "## Metrics\n\n*   **Platform Targets:** Large multi-state urology platforms with ASC ownership trade at **10-14x EV/EBITDA** [2].\n"
          "*   **Platform Targets again:** Urology platform acquisitions trade at 10-14x EV/EBITDA [2].\n")
    assert ev.repeats(md) == 2          # both Metrics bullets restate Market; TL;DR is exempt


def test_shared_wording_about_different_things_is_not_a_restatement():
    gemma = ("It requires roughly 14 GB of VRAM at Q4 quantization, necessitating partial "
             "layer offloading to system memory on this card")
    md = f"## Gemma\n\n{gemma} [6].\n\n## Qwen\n\n{gemma.replace('14', '17')} [22].\n"
    # a known limit: the same template about another model reads as a
    # restatement when it is word for word
    assert ev.repeats(md) == 1
    md = md.replace(gemma.replace("14", "17"),
                    "Qwen needs about 17 GB at Q4, so some expert layers are streamed from "
                    "system memory during generation on this card")
    assert ev.repeats(md) == 0


def test_figures_from_the_question_do_not_anchor_a_restatement():
    # most of the shorter sentence's words are shared, under half of all of
    # them: the shared figure decides, and "6700" is only the asker's GPU
    md = ("## A\n\nOn the RX 6700 XT, offloading expert layers to system memory creates a "
          "bandwidth bottleneck during generation [1].\n\n"
          "## B\n\nOn the RX 6700 XT, the bandwidth bottleneck from offloading expert layers to "
          "system memory is worse under DDR4 than DDR5 and limits practical token throughput "
          "considerably [2].\n")
    assert ev.repeats(md) == 1
    assert ev.repeats(md, query="I have an RX 6700 XT") == 0


def test_figure_stats_counts_what_the_cited_sources_do_not_state():
    md = "# T\n\n## A\n\nWhisper reaches 3.2% WER [1]. Granite reaches 5.52% WER [1].\n"
    figures, unmatched = ev.figure_stats(md, [src(1, "Granite: 5.52% average WER.")], "")
    assert (figures, unmatched) == (2, 1)


def test_model_free_reports_lost_parts_and_known_errors():
    meta = {"query": "q", "query_facet": {"q1": "targets", "q2": "market"}}
    findings = [src(1, "n", query="q1"), src(2, "n", query="q2"), src(32, "n", query="q1")]
    m = ev.model_free(REPORT, {"facts": [OWNED]}, meta, findings)
    assert m["kept"] == 3 and m["cited"] == 3
    assert m["known"]["owned"]["status"] == "present"
    assert m["lost_parts"] == []
    assert ev.model_free("# T\n\n## A\n\nOnly this [1].\n", {"facts": []}, meta, findings)["lost_parts"] == ["market"]


# --- the judge ------------------------------------------------------------------

def _passage_judge(quote: str):
    """Calls the TL;DR passage a contradiction, with `quote`; the rest fine."""
    def reply(messages):
        if "Top Acquisition Targets (Add-ons)" in messages[-1]["content"]:
            return {"verdict": "contradicted", "quote": quote, "why": "listed as targets"}
        return {"verdict": "consistent", "quote": "", "why": "names the owner"}
    return reply


async def test_one_wrong_passage_is_not_excused_by_the_right_ones():
    # judged all at once, a report right in one place and wrong in another
    # came back "consistent"
    fact = {**OWNED, "fact": "U.S. Urology Partners owns both practices."}
    real = "Examples include Florida Urology Center and New Jersey Urology [2][44]."
    llm = FakeLLM({"eval_fact": [_passage_judge(real)]})
    out = await ev.judge_fact(llm, fact, REPORT)
    assert out["verdict"] == "contradicted" and out["quote"] == real      # bold marks ignored
    assert llm.calls["eval_fact"] == 3                                    # one call per passage


async def test_a_contradiction_counts_only_when_its_quote_is_in_the_report():
    fact = {**OWNED, "fact": "U.S. Urology Partners owns both practices."}
    llm = FakeLLM({"eval_fact": [_passage_judge("Florida Urology Center is for sale.")]})
    assert (await ev.judge_fact(llm, fact, REPORT))["verdict"] == "unverified"


async def test_a_fact_the_report_never_mentions_is_not_sent_to_the_judge():
    llm = FakeLLM({})
    out = await ev.judge_fact(llm, {**OWNED, "fact": "x", "terms": [r"Gastro Health"]}, REPORT)
    assert out["verdict"] == "not_addressed" and llm.total_calls == 0


def test_a_conflict_needs_both_quotes_in_the_report_and_the_subject_in_both():
    a = "**Profile:** Backed by NMS Capital, with a growth investment from General Atlantic in April 2025 [1][18]."
    b = "**Assets:** Includes New Jersey Urology and Florida Urology Center [32]."
    pairs = [
        ev.Pair(subject="U.S. Urology Partners", a=a, b=b, why="ok"),          # kept
        ev.Pair(subject="U.S. Urology Partners", a=a, b=a, why="same"),        # one statement
        ev.Pair(subject="U.S. Urology Partners", a=a, b="It is backed by Ares alone.", why="invented"),
        # its own pair of quotes, so only the subject can be what rejects it
        ev.Pair(subject="Gastro Health", a=b,
                b="**Examples:** Florida Urology Center, Arizona Urology Specialists.",
                why="subject in neither"),
        ev.Pair(subject="U.S. Urology Partners", a=b, b=a, why="same pair again"),
    ]
    kept = ev.verify_pairs(pairs, REPORT)
    assert [p["why"] for p in kept] == ["ok"]


async def test_an_uncheckable_pair_is_dropped_before_the_model_is_asked_again():
    llm = FakeLLM({"eval_conflicts": [{"pairs": [
        {"subject": "nothing", "a": "not in the report at all", "b": "also not in it at all", "why": "x"}]}]})
    out = await ev.judge_conflicts(llm, REPORT)
    assert (out["claimed"], out["checkable"], out["pairs"]) == (1, 0, [])
    assert llm.calls["eval_confirm"] == 0


async def test_a_checkable_pair_counts_only_when_confirmed_alone():
    # 6 of the first 7 checkable pairs were one statement made twice
    pair = {"subject": "U.S. Urology Partners", "why": "x",
            "a": "**Profile:** Backed by NMS Capital, with a growth investment from General Atlantic in April 2025 [1][18].",
            "b": "**Assets:** Includes New Jersey Urology and Florida Urology Center [32]."}
    llm = FakeLLM({"eval_conflicts": [{"pairs": [pair]}],
                   "eval_confirm": [{"verdict": "compatible", "why": "different attributes"},
                                    {"verdict": "conflict", "why": "two owners"}]})
    out = await ev.judge_conflicts(llm, REPORT)
    assert (out["claimed"], out["checkable"], out["pairs"]) == (1, 1, [])
    out = await ev.judge_conflicts(llm, REPORT)
    assert len(out["pairs"]) == 1 and out["pairs"][0]["confirmed_why"] == "two owners"


def test_only_a_clear_conflict_verdict_counts():
    assert ev.PairVerdict.model_validate({"verdict": "Conflict"}).verdict == "conflict"
    assert ev.PairVerdict.model_validate({"verdict": "probably"}).verdict == "compatible"
    assert ev.PairVerdict.model_validate({}).verdict == "compatible"


# --- the frozen set -------------------------------------------------------------

def _run(root: Path) -> Path:
    run = root / "runs" / "r1"
    (run / "findings").mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({"query": "q", "title": "T"}))
    (run / "overview.md").write_text(REPORT)
    (run / "findings" / "032_x.md").write_text(
        "# [32] PitchBook profile\n\n- **URL:** https://p.example/x\n- **Domain:** p.example\n"
        "- **Published:** undated\n- **Relevance:** 8/10\n- **Found via:** q1\n\n"
        "**Summary:** A profile.\n\n## Notes\n\nBought Florida Urology Center in 2022.\n\n"
        "## Key facts\n\n- Bought New Jersey Urology. (confidence 9/10)\n  > \"Buyout/LBO of New Jersey Urology\"\n")
    return run


def test_freeze_copies_once_and_check_proves_each_pattern(tmp_path, monkeypatch, capsys):
    _run(tmp_path)
    monkeypatch.setattr(ev, "SET", tmp_path / "set")
    monkeypatch.setattr(ev, "RUNS", tmp_path / "runs")
    (tmp_path / "set").mkdir()
    never = {"id": "never", "kind": "substantive", "sources": [32], "terms": [r"Gastro"],
             "violation": [{"all": [r"Gastro Health is a target"]}]}
    ghost = {**OWNED, "id": "ghost", "sources": [99]}
    (tmp_path / "set" / "cases.json").write_text(json.dumps({"cases": [
        {"id": "c1", "run": "r1", "facts": [{**OWNED, "sources": [32]}, never, ghost]}]}))
    assert ev.freeze() == 0
    (tmp_path / "runs" / "r1" / "overview.md").write_text("changed since")
    assert ev.freeze() == 0                                  # a second freeze keeps the first
    assert (tmp_path / "set" / "c1" / "baseline.md").read_text() == REPORT
    (found,) = ev.load_findings(tmp_path / "set" / "c1")
    assert found.idx == 32 and found.published is None and found.query == "q1"
    assert found.key_facts and "Florida Urology Center" in found.notes_md
    assert ev.check() == 2          # one pattern never fires; one names a source the run lacks
    out = capsys.readouterr().out
    assert "ok   c1 / owned" in out
    assert "FAIL c1 / never: not detected in baseline" in out
    assert "FAIL c1 / ghost: names sources not in the run: [99]" in out


# --- giving the model back ------------------------------------------------------

class _Cfg:
    resolved_model = "Jundot/Qwen3.6-35B-A3B-oQ4e-mtp"
    resolved_base_url = "http://model.test/v1"
    resolved_api_key = "k"


def _health(monkeypatch, body=None, error=False):
    monkeypatch.setenv("REPORT_EVAL_HEALTH", "http://app.test/health")
    route = respx.get("http://app.test/health")
    route.mock(side_effect=httpx.ConnectError("down")) if error else route.mock(
        return_value=httpx.Response(200, json=body))


@respx.mock
def test_the_model_is_unloaded_when_nothing_else_is_using_it(monkeypatch):
    _health(monkeypatch, {"status": "ok", "active": 0, "queued": 0})
    # the server knows the model by its own name, not the "owner/" form
    unload = respx.post("http://model.test/v1/models/Qwen3.6-35B-A3B-oQ4e-mtp/unload").mock(
        return_value=httpx.Response(200, json={"status": "ok"}))
    assert ev.release_model(_Cfg, []) == "model unloaded"
    assert unload.called and unload.calls[0].request.headers["authorization"] == "Bearer k"


@respx.mock
def test_the_model_is_left_alone_when_a_run_is_using_it_or_that_cannot_be_told(monkeypatch):
    unload = respx.post("http://model.test/v1/models/Qwen3.6-35B-A3B-oQ4e-mtp/unload").mock(
        return_value=httpx.Response(200))
    _health(monkeypatch, {"status": "ok", "active": 1, "queued": 0})
    assert "a research run is using it" in ev.release_model(_Cfg, [])
    _health(monkeypatch, error=True)
    assert "could not tell" in ev.release_model(_Cfg, [])
    _health(monkeypatch, {"status": "ok", "active": 0, "queued": 0})
    assert "--keep-loaded" in ev.release_model(_Cfg, ["--keep-loaded"])
    assert not unload.called


@respx.mock
def test_a_server_without_an_unload_route_is_not_an_error(monkeypatch):
    _health(monkeypatch, {"status": "ok", "active": 0, "queued": 0})
    respx.post("http://model.test/v1/models/Qwen3.6-35B-A3B-oQ4e-mtp/unload").mock(
        return_value=httpx.Response(404))
    assert ev.release_model(_Cfg, []) == "model left loaded: the server answered 404"
    respx.post("http://model.test/v1/models/Qwen3.6-35B-A3B-oQ4e-mtp/unload").mock(
        return_value=httpx.Response(400))
    assert ev.release_model(_Cfg, []) == "model was not loaded"


def test_the_script_is_where_the_tests_expect_it():
    # a missing script must fail the suite, not skip it
    assert _SCRIPT.exists()
