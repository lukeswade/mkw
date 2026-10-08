"""Three outcomes that now say what happened (2026-10-07):

- a claim check's "unsupported" must rest on a verbatim contradicting quote
  from two independent sources, or it is downgraded;
- a page skipped by the notes step says why, instead of "unusable notes
  output" for every cause;
- a refused gap call ends the research, not the run: what was gathered
  still makes a report.
"""
from __future__ import annotations

import json

import httpx
import respx

from app.db import Repo, connect
from app.llm.client import LLMError, PromptExceedsFreeMemory, ServerOutOfMemory
from app.llm.json_utils import LLMJsonError
from app.models import RunParams, VerdictOut
from app.research.notes import skip_reason, take_notes
from app.research.orchestrator import Orchestrator
from app.research.progress import ProgressBus
from app.research.verify import Evidence, bound_unsupported, judge
from tests.fake_llm import FakeLLM
from tests.test_pipeline_e2e import SX, article, make_cfg, script, sx_payload, sx_result

_QUOTE = "The model has 137 million parameters."
_EV = [Evidence(n=1, label="a", url="https://alpha.example/page", text=_QUOTE),
       Evidence(n=2, label="b", url="https://beta.example/spec", text="It ships with 137M parameters."),
       Evidence(n=3, label="c", url="https://alpha.example/other", text="Another alpha page."),
       Evidence(n=4, label="your research — x", url="/runs/r1", text="A library chunk.")]


def _v(verdict: str, quote: str = "", sources=()) -> VerdictOut:
    return VerdictOut(verdict=verdict, confidence=8, reasoning="r", quote=quote,
                      sources=list(sources))


# ---- "unsupported" needs a contradiction, from two sources ------------------------

def test_unsupported_without_a_contradicting_quote_is_unverifiable():
    out = bound_unsupported(_v("unsupported"), _EV)
    assert out.verdict == "unverifiable" and "fails to confirm" in out.reasoning


def test_unsupported_on_one_source_is_contested():
    out = bound_unsupported(_v("unsupported", _QUOTE, [1]), _EV)
    assert out.verdict == "contested" and "Only one source" in out.reasoning


def test_two_pages_of_one_site_are_one_source():
    assert bound_unsupported(_v("unsupported", _QUOTE, [1, 3]), _EV).verdict == "contested"


def test_unsupported_on_two_independent_sources_stands():
    assert bound_unsupported(_v("unsupported", _QUOTE, [1, 2]), _EV).verdict == "unsupported"
    # an earlier run of your own counts as a source of its own
    assert bound_unsupported(_v("unsupported", _QUOTE, [1, 4]), _EV).verdict == "unsupported"


def test_other_verdicts_pass_through_untouched():
    for verdict in ("supported", "contested", "unverifiable"):
        assert bound_unsupported(_v(verdict), _EV).verdict == verdict


async def test_the_judge_applies_the_bound():
    """2026-10-07: a true 149M claim that no source gave a number for came
    back "unsupported (8/10)"."""
    llm = FakeLLM({"verify": [{"verdict": "unsupported", "confidence": 8,
                               "reasoning": "no count is given", "quote": "",
                               "sources": [1]}]})
    out = await judge(llm, "The model has about 149 million parameters.", _EV)
    assert out.verdict == "unverifiable"


# ---- a skipped page says why -------------------------------------------------------

def test_skip_reasons_say_what_happened():
    assert skip_reason(PromptExceedsFreeMemory("x")) == \
        "not enough free memory on the model server for this page"
    assert skip_reason(LLMJsonError("bad json")) == "unusable notes output"
    assert skip_reason(LLMError("LLM call 'notes' exceeded the 600s ceiling — the "
                                "server never stopped generating.")) == \
        "the notes call ran past its time limit"
    assert skip_reason(LLMError("LLM call 'notes' failed after retries: 429")).startswith(
        "the notes call failed: ")


async def test_take_notes_hands_back_the_reason():
    why: list[str] = []
    out = await take_notes(FakeLLM({"notes": [PromptExceedsFreeMemory("no room")]}),
                           brief="b", recency_desc="any", today="t", url="u", title="t",
                           detected_date=None, text="a short page about torque specs",
                           keywords=None, why=why)
    assert out is None
    assert why == ["not enough free memory on the model server for this page"]


def _round(cfg):
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result(f"https://example-{c}.com/article", f"Article {c.upper()}")
         for c in "abcde"])))
    for c in "abcde":
        respx.get(f"https://example-{c}.com/article").mock(
            return_value=httpx.Response(200, html=article(f"Article {c.upper()}")))


async def _execute(cfg, s):
    repo = Repo(connect(cfg.db_path))
    llm = FakeLLM(s)
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(), llm_factory=lambda: llm)
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)
    events = [json.loads(line) for line in
              (cfg.research_dir / run_id / "events.jsonl").read_text().splitlines()
              if line.strip()]
    return repo.get_run(run_id), llm, events


@respx.mock
async def test_a_page_skipped_for_memory_says_so_in_the_run(data_dir):
    cfg = make_cfg(data_dir)
    _round(cfg)
    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    ok = s["notes"][0]
    s["notes"] = [lambda m: PromptExceedsFreeMemory("no room")
                  if "example-c.com" in m[-1]["content"] else ok]
    row, _llm, events = await _execute(cfg, s)
    assert row["status"] == "completed"
    reasons = {e["url"]: e["reason"] for e in events if e["type"] == "source_skipped"}
    assert reasons["https://example-c.com/article"] == \
        "not enough free memory on the model server for this page"


# ---- a refused gap call ends the research, not the run ----------------------------

@respx.mock
async def test_a_refused_gap_call_still_ends_in_a_report(data_dir):
    cfg = make_cfg(data_dir)
    _round(cfg)
    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["gap"] = [PromptExceedsFreeMemory("not enough free memory for a prompt this long")]
    row, llm, events = await _execute(cfg, s)
    assert row["status"] == "completed"
    assert llm.calls["synth"] >= 1
    assert row["stop_reason"].startswith("gap analysis failed")
    assert any("writing the report from the 5 source(s)" in e.get("message", "")
               for e in events if e["type"] == "log")


@respx.mock
async def test_a_server_out_of_memory_at_the_gap_still_fails_the_run(data_dir):
    cfg = make_cfg(data_dir)
    _round(cfg)
    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["gap"] = [ServerOutOfMemory("LLM call 'gap' refused: the server is out of memory.")]
    row, llm, _events = await _execute(cfg, s)
    assert row["status"] == "failed" and "out of memory" in row["error"]
    assert llm.calls["synth"] == 0
