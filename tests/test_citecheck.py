"""The citation check: figures in a finished report against the sources cited.

Built from the 2026-11-01 measurement: about one cited sentence with a
figure in ten misstated its source. The string pass decides what the model
sees; nothing the model proposes is applied unless code can confirm it.
"""
from __future__ import annotations

import dataclasses
import json

import httpx
import respx

from app.db import Repo, connect
from app.models import CiteCheckOut, RunParams
from app.research import citecheck
from app.research.citecheck import check, claim_tokens
from app.research.notes import Finding
from app.research.orchestrator import Orchestrator
from app.research.pipeline import Pipeline
from app.research.progress import ProgressBus
from app.research.storage import citation_ids, validate_citations
from app.research.synthesizer import cited_ids
from app.web.markdown import render_overview
from tests.fake_llm import FakeLLM
from tests.test_pipeline_e2e import SX, article, make_cfg, script, sx_payload, sx_result


def src(idx: int, notes: str, facts=(), relevance: int = 8,
        title: str = "Source") -> Finding:
    return Finding(idx=idx, url=f"https://s{idx}.example/x", title=f"{title} {idx}",
                   domain=f"s{idx}.example", published="2026-10-10",
                   relevance=relevance, summary="", notes_md=notes,
                   key_facts=[{"claim": c, "evidence_quote": q, "confidence": 9}
                              for c, q in facts])


def recorder(payload):
    """A scripted reply that also keeps every prompt it was sent."""
    seen: list[str] = []

    def reply(messages):
        seen.append(messages[-1]["content"])
        return payload
    return reply, seen


# --- citation markers ------------------------------------------------------

def test_citation_ids_read_grouped_markers():
    assert citation_ids("a [1] b [2, 25] c [4; 9] d [7][8]") == [1, 2, 25, 4, 9, 7, 8]


def test_a_source_cited_only_inside_a_group_counts_as_cited():
    # Matt's 10-15 report listed [25] as unused beside "... ~13-16GB [2, 25]".
    assert cited_ids("It needs ~13-16GB [2, 25].") == {2, 25}


def test_validate_citations_strips_bad_ids_inside_a_group():
    md, removed = validate_citations("A [3, 99]. B [98, 99]. C [2, 25].", 30)
    assert md == "A [3]. B . C [2, 25]."
    assert removed == {98, 99}


def test_grouped_markers_render_as_links():
    html = render_overview("Fits ~13-16GB [2, 25].", 30)
    assert 'href="#src-2"' in html and 'href="#src-25"' in html


# --- what the string pass reads -------------------------------------------

def keys(text: str) -> set[str]:
    return {t.key for t in claim_tokens(text)}


def test_claim_tokens_are_figures_and_numbered_names():
    got = keys("On a Qwen3.6-35B-A3B it runs ~24 tok/s at 53.2%, 1,200 tokens, "
               "4GB, 3-12 range, build b9615+ and llama.cpp 0.7.3 [10].")
    assert {t.unit for t in claim_tokens("needs 4GB and 53.2%")} == {"gb", "%"}
    assert {"qwen3635ba3b", "24", "53.2", "1200", "4", "3", "12", "b9615",
            "0.7.3"} <= got
    assert "10" not in got                      # a citation, not a claim


def test_years_single_digits_and_ordinals_are_not_claims():
    assert keys("In 2026 there were 3 options, the 1st of them free.") == set()


# --- the string pass -------------------------------------------------------

async def test_a_sentence_the_string_pass_clears_is_never_sent():
    llm = FakeLLM({})                            # any call would fail the test
    md = "# T\n\nQwen3.6-35B-A3B runs ~24 tok/s over 1200 tokens [1].\n"
    f = src(1, "Qwen 3.6 35B A3B ran at 24.0 tokens/second across 1,200 tokens.")
    out, summary = await check(llm, md, [f], query="q")
    assert out == md and summary.sentences == 1 and summary.flagged == 0
    assert llm.total_calls == 0


async def test_every_source_in_a_group_counts_as_evidence():
    llm = FakeLLM({})
    md = "# T\n\nIt reaches 24 tok/s [1, 2].\n"
    out, summary = await check(llm, md, [src(1, "No figures here."),
                                         src(2, "Measured 24 tokens/second.")],
                               query="q")
    assert summary.flagged == 0 and llm.total_calls == 0


async def test_tables_headings_quotes_and_closing_sections_are_not_checked():
    llm = FakeLLM({})
    md = ("# T\n\n## 2026 results [1]\n\n| Model | 99 tok/s [1] |\n|---|---|\n"
          "> 77 widgets [1].\n\n## Researched but not used\n\n- [2] 55 widgets [1].\n")
    out, summary = await check(llm, md, [src(1, "Nothing numeric."), src(2, "n")],
                               query="q")
    assert out == md and summary.sentences == 0 and llm.total_calls == 0


# --- what the model may change ---------------------------------------------

WRONG = "# T\n\n## Speed\n\nQwen3.6 runs at 24 tok/s on a 3080 Ti [1].\n"


async def test_recite_moves_the_citation_to_the_source_stating_the_figure():
    reply, seen = recorder({"action": "recite", "cite": [2], "reason": "[2] states it"})
    llm = FakeLLM({"citecheck": [reply]})
    found = [src(1, "Qwen3.6 is a 35B MoE."),
             src(2, "Qwen3.6 runs at 24 tokens/second on an RTX 3080 Ti.")]
    out, summary = await check(llm, WRONG, found, query="q")
    assert "Qwen3.6 runs at 24 tok/s on a 3080 Ti [2]." in out
    assert summary.recited == 1 and summary.changes[0]["outcome"] == "recited"
    assert "[2] Source 2" in seen[0]             # offered as a source stating it


async def test_recite_is_refused_when_the_new_source_does_not_state_every_figure():
    # [3] is offered (it states 24) but not the 3080 the sentence also needs
    llm = FakeLLM({"citecheck": [{"action": "recite", "cite": [3]}]})
    found = [src(1, "Qwen3.6 is a 35B MoE."),
             src(3, "Some other model runs at 24 tokens/second.")]
    out, summary = await check(llm, WRONG, found, query="q")
    assert out == WRONG and summary.rejected == 1


async def test_recite_is_refused_for_a_source_that_was_not_offered():
    llm = FakeLLM({"citecheck": [{"action": "recite", "cite": [99]}]})
    out, summary = await check(llm, WRONG, [src(1, "Qwen3.6 is a 35B MoE.")],
                               query="q")
    assert out == WRONG and summary.rejected == 1


WER = "# T\n\nWhisper Large V3 has a WER of ~3.2% on clean speech [1].\n"
WER_SRC = [src(1, "Whisper Large V3 scored a WER of about 7.4% on clean speech.")]


async def test_a_correction_is_applied_when_it_passes_the_string_check():
    llm = FakeLLM({"citecheck": [{
        "action": "rewrite",
        "revised": "Whisper Large V3 has a WER of ~7.4% on clean speech [1].",
        "reason": "source says 7.4%"}]})
    out, summary = await check(llm, WER, WER_SRC, query="q")
    assert "WER of ~7.4% on clean speech [1]." in out and "3.2" not in out
    assert summary.rewritten == 1


async def test_a_correction_stating_another_unsourced_figure_is_refused():
    llm = FakeLLM({"citecheck": [{
        "action": "rewrite",
        "revised": "Whisper Large V3 has a WER of ~5.1% on clean speech [1]."}]})
    out, summary = await check(llm, WER, WER_SRC, query="q")
    assert out == WER and summary.rejected == 1


async def test_a_correction_without_a_citation_is_refused():
    # with its figure dropped too, nothing else would stop an uncited sentence
    llm = FakeLLM({"citecheck": [
        {"action": "rewrite", "revised": "Whisper Large V3 has a WER of ~7.4% on clean speech."},
        {"action": "rewrite", "revised": "Whisper is accurate on clean speech."}]})
    for _ in range(2):
        out, summary = await check(llm, WER, WER_SRC, query="q")
        assert out == WER and summary.rejected == 1


async def test_a_correction_keeps_the_bullet_and_its_label():
    md = "# T\n\n*   **Fastest:** Parakeet runs at an RTF of 103x [1].\n"
    llm = FakeLLM({"citecheck": [{
        "action": "rewrite", "revised": "Parakeet runs at an RTF of 81x [1]."}]})
    out, _ = await check(llm, md, [src(1, "Parakeet reached an RTF of 81x.")],
                         query="q")
    assert "*   **Fastest:** Parakeet runs at an RTF of 81x [1]." in out


async def test_the_askers_own_setup_may_stay_in_a_correction():
    md = "# T\n\nOn an RX 6700 XT, Qwen3-Coder runs at 24 tok/s [1].\n"
    llm = FakeLLM({"citecheck": [{
        "action": "rewrite",
        "revised": "On an RX 6700 XT, Qwen3.6 ran 24 tok/s on an RTX 3080 Ti [1]."}]})
    out, summary = await check(
        llm, md, [src(1, "Qwen3.6 runs at 24 tokens/second on an RTX 3080 Ti.")],
        query="Best local model for an RX 6700 XT with 12GB?")
    assert summary.rewritten == 1 and "RTX 3080 Ti [1]" in out


async def test_keep_leaves_the_sentence():
    llm = FakeLLM({"citecheck": [{"action": "keep", "reason": "a unit conversion"}]})
    out, summary = await check(llm, WER, WER_SRC, query="q")
    assert out == WER and summary.kept == 1


def test_an_unknown_action_means_keep():
    assert CiteCheckOut.model_validate({"action": "delete it"}).action == "keep"
    assert CiteCheckOut.model_validate({"cite": ["[33]", "4"]}).cite == [33, 4]


async def test_a_failed_call_keeps_the_sentence():
    llm = FakeLLM({"citecheck": [RuntimeError("server went away")]})
    out, summary = await check(llm, WER, WER_SRC, query="q")
    assert out == WER and summary.failed == 1


async def test_the_check_never_costs_the_document(monkeypatch):
    def boom(*a, **k):
        raise ValueError("bug")
    monkeypatch.setattr(citecheck, "_units", boom)
    out, summary = await check(FakeLLM({}), WER, WER_SRC, query="q")
    assert out == WER


async def test_a_sentence_flagged_only_by_the_askers_setup_is_never_sent():
    # the model dropped the asker's own engine from such a sentence
    md = ("# T\n\nOn your 5600X it should be fine [1].\n\n"
          "Whisper Large V3 has a WER of ~3.2% on clean speech [1].\n")
    reply, seen = recorder({"action": "keep"})
    llm = FakeLLM({"citecheck": [reply]})
    _, summary = await check(llm, md, WER_SRC, query="I have a Ryzen 5600X.")
    assert summary.flagged == 1 and len(seen) == 1 and "3.2%" in seen[0]


async def test_the_ceiling_caps_what_is_sent(monkeypatch):
    monkeypatch.setattr(citecheck, "_MAX_CHECKS", 1)
    md = ("# T\n\nWhisper Large V3 has a WER of ~3.2% on clean speech [1].\n\n"
          "Whisper Large V3 needs 9GB of memory [1].\n")
    reply, seen = recorder({"action": "keep"})
    _, summary = await check(FakeLLM({"citecheck": [reply]}), md, WER_SRC, query="q")
    assert summary.flagged == 2 and summary.skipped == 1 and len(seen) == 1


async def test_open_questions_questions_and_unknowns_are_not_checked():
    llm = FakeLLM({})
    md = ("# T\n\n## Findings\n\nLong-tail latency (P99.9) remains less documented [1].\n\n"
          "Does Qwen3.8 keep its 2x speed-up under load [1]?\n\n"
          "## Open questions\n\n- Qwen3.8-27B with Q4_K_XL vs Q4_K_M for tool calling [1].\n")
    out, summary = await check(llm, md, [src(1, "P99 latency was 4 ms.")], query="q")
    assert out == md and summary.sentences == 0 and llm.total_calls == 0


def test_slash_compounds_are_read_as_their_parts():
    assert keys("Requires AVX2/FMA3 [17].") == {"avx2", "fma3"}


def test_short_names_are_not_claims():
    # "E2E encryption" was once dropped as if it were an unsourced figure
    assert keys("Full E2E encryption on an M4 with Whisper V3 at Q4 [2].") == set()


async def test_a_rounding_or_another_name_for_the_model_is_not_a_mismatch():
    llm = FakeLLM({})
    md = ("# T\n\nIt needs more than 125GB of RAM [1].\n\n"
          "Qwen3.8-Flash-Next reaches 40 tok/s on one card [1].\n")
    f = src(1, "Qwen 3.8 Flash ran at 40 tokens/second; peak RAM was 125.7GB.")
    out, summary = await check(llm, md, [f], query="q")
    assert summary.flagged == 0 and llm.total_calls == 0


def test_a_shorter_model_name_does_not_match_inside_a_longer_version():
    from app.research.citecheck import _index
    idx = _index("Qwen3.6 runs at 24 tokens/second on an RTX 3080 Ti.")
    (coder,) = claim_tokens("Qwen3-Coder")
    assert not idx.has(coder)                    # "qwen3" sits inside "qwen36"


async def test_recite_keeps_the_original_citation_when_it_still_supports_part():
    # the model's list alone left "15 of 19 benchmarks" to sources that never said it
    md = "# T\n\nIt has a 262K context and wins 15 of 19 benchmarks [5].\n"
    found = [src(5, "It wins 15 of 19 benchmarks."), src(3, "Its context is 262K.")]
    llm = FakeLLM({"citecheck": [{"action": "recite", "cite": [3]}]})
    out, summary = await check(llm, md, found, query="q")
    assert "wins 15 of 19 benchmarks [3][5]." in out and summary.recited == 1


async def test_a_correction_must_state_the_figure_with_its_unit():
    # "12-15GB" was accepted from a source that said "12B-15B parameters"
    md = "# T\n\nFull-GPU inference needs ~14-16GB at Q4 [1].\n"
    f = src(1, "Dense models up to 12B-15B parameters fit on 12GB cards.")
    llm = FakeLLM({"citecheck": [{"action": "rewrite",
                                  "revised": "Full-GPU inference needs ~12-15GB at Q4 [1]."}]})
    out, summary = await check(llm, md, [f], query="q")
    assert out == md and summary.rejected == 1


async def test_a_fix_that_changes_nothing_counts_as_kept():
    llm = FakeLLM({"citecheck": [{"action": "rewrite", "revised": WER.split("\n\n")[1].strip()},
                                 {"action": "recite", "cite": [1]}]})
    for _ in range(2):
        out, summary = await check(llm, WER, WER_SRC, query="q")
        assert out == WER and summary.kept == 1


# --- in the pipeline -------------------------------------------------------

NUMERIC_SYNTH = ("# Test Research\n\n## Detail\n\n"
                 "Solid claim of 42 widgets [1].\n\n## Open questions\n\n- More?\n")
FIX = {"action": "rewrite", "revised": "Solid claim [1].", "reason": "no source gives 42"}


def mock_one_source():
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://example-a.com/article", "Article A")])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))


@respx.mock
async def test_a_run_corrects_its_overview_and_records_it(data_dir):
    cfg = make_cfg(data_dir)
    mock_one_source()
    sc = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    sc["synth"] = [NUMERIC_SYNTH]
    sc["citecheck"] = [FIX]
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(sc))
    run_id = orch.enqueue(RunParams(query="widgets", depth=1, recency="all",
                                    origin="cli"))
    await orch.execute_now(run_id)
    row = repo.get_run(run_id)
    assert row["status"] == "completed"
    overview = (cfg.research_dir / row["dir"] / "overview.md").read_text()
    assert "Solid claim [1]." in overview and "42 widgets" not in overview
    stats = json.loads(row["stats_json"])
    assert stats["citation_check"]["rewritten"] == 1


@respx.mock
async def test_resynthesis_runs_the_check_and_the_setting_turns_it_off(data_dir):
    cfg = make_cfg(data_dir)
    mock_one_source()
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(
                            script([{"state_md": "s", "saturated": True,
                                     "next_queries": []}])))
    run_id = orch.enqueue(RunParams(query="widgets", depth=1, recency="all",
                                    origin="cli"))
    await orch.execute_now(run_id)
    run_dir = cfg.research_dir / repo.get_run(run_id)["dir"]

    llm = FakeLLM({"synth": [NUMERIC_SYNTH], "citecheck": [FIX],
                   "followups": [{"items": []}]})
    await Pipeline(cfg, repo, ProgressBus(), llm_factory=lambda: llm).resynthesize(run_id)
    assert "Solid claim [1]." in (run_dir / "overview.md").read_text()
    assert json.loads(repo.get_run(run_id)["stats_json"])["citation_check"]["rewritten"] == 1

    off_cfg = dataclasses.replace(cfg, citation_check=False)
    off = FakeLLM({"synth": [NUMERIC_SYNTH], "followups": [{"items": []}]})
    await Pipeline(off_cfg, repo, ProgressBus(),
                   llm_factory=lambda: off).resynthesize(run_id)
    assert "42 widgets" in (run_dir / "overview.md").read_text()
    assert off.calls["citecheck"] == 0
