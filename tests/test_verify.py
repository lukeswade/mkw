"""Claim verification: extraction, the library-then-web order, and reporting."""
from __future__ import annotations

import httpx
import respx

from app.db import Repo, connect
from app.models import VERIFY_CLAIM_CAP, Claim, RunParams, VerdictOut
from app.research.orchestrator import Orchestrator
from app.research.progress import ProgressBus
from app.research.verify import (Checked, Evidence, clip_document, judge,
                                 library_evidence, render_report,
                                 select_claims, settled)
from tests.fake_llm import FakeLLM
from tests.test_pipeline_e2e import SX, article, make_cfg, sx_payload, sx_result

DOC = ("MLX decodes 30% faster than llama.cpp on an M4 Pro. "
       "Ollama added an MLX backend in version 0.19. "
       "I think MLX will win the Mac inference market. " * 3)


def _claim(text, importance=5, checkable=True):
    return Claim(text=text, importance=importance, checkable=checkable)


# ---- claim selection ------------------------------------------------------------

def test_selection_checks_the_most_load_bearing_and_reports_the_rest():
    claims = [_claim(f"c{i}", importance=i % 11) for i in range(25)]
    checked, skipped = select_claims(claims, 5)
    assert len(checked) == 5
    assert len(skipped) == 20
    assert min(c.importance for c in checked) >= max(c.importance for c in skipped)


def test_selection_sets_aside_opinions_rather_than_judging_them():
    claims = [_claim("MLX is 30% faster", 9),
              _claim("MLX will win the market", 9, checkable=False)]
    checked, skipped = select_claims(claims, VERIFY_CLAIM_CAP)
    assert [c.text for c in checked] == ["MLX is 30% faster"]
    assert skipped == []                      # the opinion is not "skipped", it is excluded


def test_selection_preserves_document_order_within_the_kept_set():
    claims = [_claim("first", 9), _claim("second", 9), _claim("third", 9)]
    checked, _ = select_claims(claims, 3)
    assert [c.text for c in checked] == ["first", "second", "third"]


def test_long_documents_are_clipped_and_flagged():
    short, clipped = clip_document("a paragraph")
    assert short == "a paragraph" and clipped is False
    _body, clipped = clip_document("word " * 200_000)
    assert clipped is True


# ---- the library-first rule -----------------------------------------------------

def test_a_confident_library_verdict_settles_a_claim():
    assert settled(VerdictOut(verdict="supported", confidence=8))
    assert not settled(VerdictOut(verdict="supported", confidence=4))
    # "unverifiable" never settles, however confident the model claims to be
    assert not settled(VerdictOut(verdict="unverifiable", confidence=10))


class _Rag:
    def __init__(self, hits):
        self.hits = hits
        self.queries: list[str] = []
        self.indexed: list[str] = []

    async def semantic_search(self, query, limit=20):
        self.queries.append(query)
        return self.hits

    async def index_run(self, repo, run_id):
        self.indexed.append(run_id)
        return 0


async def test_library_evidence_drops_weak_matches():
    rag = _Rag([{"run_id": "r1", "title": "MLX bench", "text": "41 tok/s", "score": 0.9},
                {"run_id": "r2", "title": "Unrelated", "text": "spark plugs", "score": 0.1}])
    ev = await library_evidence(rag, "MLX is fast")
    assert [e.text for e in ev] == ["41 tok/s"]
    assert ev[0].n == 1
    assert "MLX bench" in ev[0].label


async def test_library_failure_is_not_fatal():
    class Broken:
        async def semantic_search(self, *a, **k):
            raise RuntimeError("chroma down")
    assert await library_evidence(Broken(), "anything") == []
    assert await library_evidence(None, "anything") == []


async def test_no_evidence_yields_unverifiable_without_calling_the_model():
    llm = FakeLLM({})                       # any call would raise
    v = await judge(llm, "some claim", [])
    assert v.verdict == "unverifiable" and v.confidence == 0


async def test_an_adjudication_failure_degrades_to_unverifiable():
    class Boom:
        async def chat_json(self, *a, **k):
            raise RuntimeError("model down")
    v = await judge(Boom(), "c", [Evidence(1, "l", "u", "t")])
    assert v.verdict == "unverifiable"
    assert "Adjudication failed" in v.reasoning


# ---- report -----------------------------------------------------------------------

def _checked(text, verdict, conf=8, via="library", n_sources=1):
    ev = [Evidence(i + 1, f"src{i}", f"https://s/{i}", "t") for i in range(n_sources)]
    return Checked(claim=_claim(text), evidence=ev, via=via,
                   verdict=VerdictOut(verdict=verdict, confidence=conf,
                                      reasoning="because", quote="q",
                                      sources=[1] if n_sources else []))


def test_report_tabulates_verdicts_and_counts_them():
    md = render_report("Doc", [_checked("a", "supported"),
                               _checked("b", "unsupported")],
                       skipped=[], uncheckable=[], clipped=False)
    assert "2 claim(s) checked" in md
    assert "1 supported" in md and "1 unsupported" in md
    assert "✓ supported (8/10)" in md and "✗ unsupported (8/10)" in md


def test_report_lists_what_it_did_not_check():
    md = render_report("Doc", [_checked("a", "supported")],
                       skipped=[_claim("minor point", 2)],
                       uncheckable=[_claim("it will win", 9, checkable=False)],
                       clipped=True)
    assert "1 lower-importance claim(s) were not checked" in md
    assert "minor point" in md
    assert "1 statement(s) are opinions" in md
    assert "it will win" in md
    assert "longer than the extraction budget" in md


def test_report_escapes_pipes_so_the_table_survives():
    c = _checked("a | b", "supported")
    row = next(l for l in render_report("D", [c], skipped=[], uncheckable=[],
                                        clipped=False).splitlines()
               if l.startswith("| a "))
    import re
    assert len(re.findall(r"(?<!\\)\|", row)) == 5      # 4 columns


# ---- end to end -------------------------------------------------------------------

def _script(claims, verdict="supported", conf=9):
    return {"verify": [{"claims": claims}, {"verdict": verdict,
                                            "confidence": conf,
                                            "reasoning": "because",
                                            "quote": "q", "sources": [1]}]}


@respx.mock
async def test_a_settled_library_claim_never_reaches_the_web(data_dir):
    cfg = make_cfg(data_dir)
    sx = respx.get(f"{SX}/search").mock(
        return_value=httpx.Response(200, json=sx_payload([])))
    rag = _Rag([{"run_id": "r1", "title": "prior", "text": "evidence",
                 "score": 0.9}])
    llm = FakeLLM(_script([{"text": "MLX is faster", "importance": 9,
                            "checkable": True}]))
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(), rag=rag,
                        llm_factory=lambda: llm)
    rid = orch.enqueue(RunParams(query="Claim check", depth=0, recency="all",
                                 origin="cli", kind="verify", document=DOC))
    await orch.execute_now(rid)

    assert repo.get_run(rid)["status"] == "completed"
    assert not sx.called                       # the library settled it
    md = (cfg.research_dir / rid / "overview.md").read_text()
    assert "✓ supported" in md and "library" in md


@respx.mock
async def test_an_unsettled_claim_falls_through_to_the_web(data_dir):
    cfg = make_cfg(data_dir)
    sx = respx.get(f"{SX}/search").mock(return_value=httpx.Response(
        200, json=sx_payload([sx_result("https://e.com/a", "Evidence")])))
    respx.get("https://e.com/a").mock(
        return_value=httpx.Response(200, html=article("Evidence page")))
    # library returns nothing, so the first verdict cannot settle
    llm = FakeLLM(_script([{"text": "MLX is faster", "importance": 9,
                            "checkable": True}], verdict="contested", conf=6))
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(), rag=_Rag([]),
                        llm_factory=lambda: llm)
    rid = orch.enqueue(RunParams(query="Claim check", depth=0, recency="all",
                                 origin="cli", kind="verify", document=DOC))
    await orch.execute_now(rid)

    assert sx.called                           # it did go looking
    md = (cfg.research_dir / rid / "overview.md").read_text()
    assert "± contested" in md and "web" in md


@respx.mock
async def test_an_empty_document_fails_with_a_clear_reason(data_dir):
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload([])))
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM({}))
    rid = orch.enqueue(RunParams(query="Claim check", depth=0, recency="all",
                                 origin="cli", kind="verify", document=""))
    await orch.execute_now(rid)
    row = repo.get_run(rid)
    assert row["status"] == "failed"
    assert "empty" in (row["error"] or "")


def test_an_unsupported_verdict_never_settles_from_the_library_alone():
    """The errors are not symmetric. A wrong 'supported' echoes the reader's
    own research; a wrong 'unsupported' tells them something true is false,
    which is the failure that matters when you are hunting for errors.

    Observed live: retrieval returned six passages all on one side of a
    genuinely disputed point and the adjudicator said unsupported at 10/10,
    while the library's own comparison matrix recorded the opposite."""
    assert settled(VerdictOut(verdict="supported", confidence=7))
    assert settled(VerdictOut(verdict="contested", confidence=7))
    assert not settled(VerdictOut(verdict="unsupported", confidence=10))
    assert not settled(VerdictOut(verdict="unverifiable", confidence=10))


async def test_library_evidence_is_spread_across_sources():
    """Top-N retrieval gave all six slots to one document — the same source
    three times — and produced a confident verdict on one side of a point the
    library actually disputes."""
    loud = [{"run_id": "r1", "title": "One Loud Article",
             "text": f"passage {i}", "score": 0.9 - i * 0.01} for i in range(10)]
    others = [{"run_id": "r2", "title": "Second Source", "text": "other view",
               "score": 0.7},
              {"run_id": "r3", "title": "Third Source", "text": "third view",
               "score": 0.6}]
    ev = await library_evidence(_Rag(loud + others), "a disputed claim")
    labels = [e.label for e in ev]
    assert sum("One Loud Article" in l for l in labels) == 2   # capped
    assert any("Second Source" in l for l in labels)
    assert any("Third Source" in l for l in labels)
    assert [e.n for e in ev] == list(range(1, len(ev) + 1))    # renumbered


def test_pasted_assistant_answers_get_a_useful_title():
    """An assistant answer usually opens with a compliment, and that opener
    was becoming the run's name in the library."""
    from app.web.routes_verify import _title_for
    answer = ("You're absolutely right to ask about this! Here's a breakdown.\n\n"
              "## Local LLM inference on Apple Silicon\n\n"
              "MLX outperforms llama.cpp.\n")
    assert _title_for(answer) == \
        "Claim check: Local LLM inference on Apple Silicon"

    no_heading = ("Great question! Let me explain.\n\n"
                  "The M4 Pro supports up to 128GB of unified memory.\n")
    assert _title_for(no_heading) == \
        "Claim check: The M4 Pro supports up to 128GB of unified memory."

    assert _title_for("short") == "Claim check: pasted text"


def test_the_report_contains_no_html():
    """Markdown is rendered with html=False as an XSS guard, so any tag in
    generated markdown shows up as literal characters in the run page — a
    <br> in the Why column did exactly that."""
    import re
    md = render_report("Doc", [_checked("a", "supported")],
                       skipped=[_claim("x", 1)],
                       uncheckable=[_claim("y", 1, checkable=False)],
                       clipped=True)
    # <https://…> is a CommonMark autolink, not a tag — the bibliography
    # renderer uses that form deliberately.
    without_autolinks = re.sub(r"<[a-z][a-z0-9+.-]*://[^>]*>", "", md)
    assert not re.search(r"</?[a-zA-Z][^>]*>", without_autolinks), \
        "generated markdown contains HTML, which renders as literal text"


def test_every_citation_and_evidence_entry_is_a_link():
    """Library evidence rendered as plain text while web evidence came out
    clickable: an internal /runs/… path is not a valid CommonMark autolink,
    because it has no scheme."""
    c = Checked(claim=_claim("a", 9),
                evidence=[Evidence(1, "your research — Prior", "/runs/abc", "t"),
                          Evidence(2, "example.com — Page",
                                   "https://example.com/p", "t")],
                via="library+web",
                verdict=VerdictOut(verdict="supported", confidence=9,
                                   reasoning="r", quote="q", sources=[1, 2]))
    md = render_report("T", [c], skipped=[], uncheckable=[], clipped=False)
    assert "[[1]](/runs/abc)" in md                      # internal, now linked
    assert "[[2]](https://example.com/p)" in md
    assert "1. [your research — Prior](/runs/abc)" in md
    assert "<" not in md.split("## Evidence")[1]         # no bare autolinks left


# ---- the pages it read become the run's sources ---------------------------------

@respx.mock
async def test_web_evidence_is_recorded_as_sources(data_dir):
    """A claim check used to keep nothing: empty Sources tab, no bibliography
    in any export, and none of its reading reached the library."""
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(
        200, json=sx_payload([sx_result("https://e.com/a", "Evidence page")])))
    respx.get("https://e.com/a").mock(
        return_value=httpx.Response(200, html=article("Evidence page")))
    llm = FakeLLM(_script([{"text": "MLX is faster", "importance": 9,
                            "checkable": True}], verdict="contested", conf=6))
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(), rag=_Rag([]),
                        llm_factory=lambda: llm)
    rid = orch.enqueue(RunParams(query="Claim check", depth=0, recency="all",
                                 origin="cli", kind="verify", document=DOC))
    await orch.execute_now(rid)

    rows = repo.findings_for_run(rid)
    assert [r["url"] for r in rows] == ["https://e.com/a"]
    assert rows[0]["domain"] == "e.com"
    # this verdict cited the page, so the summary says so
    assert rows[0]["summary"].startswith("Cited by 1 claim(s)")
    assert rows[0]["relevance"] == 6            # the verdict's own confidence
    sources = (cfg.research_dir / rid / "sources.md").read_text()
    assert "e.com" in sources and "No sources were kept" not in sources
    assert (cfg.research_dir / rid / rows[0]["path"]).exists()


@respx.mock
async def test_library_passages_are_not_recorded_as_new_sources(data_dir):
    """They already belong to the run they came from; recording them again
    would duplicate that research under a new id."""
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload([])))
    rag = _Rag([{"run_id": "earlier", "title": "prior", "text": "evidence",
                 "score": 0.9}])
    llm = FakeLLM(_script([{"text": "MLX is faster", "importance": 9,
                            "checkable": True}]))
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(), rag=rag,
                        llm_factory=lambda: llm)
    rid = orch.enqueue(RunParams(query="Claim check", depth=0, recency="all",
                                 origin="cli", kind="verify", document=DOC))
    await orch.execute_now(rid)
    assert repo.findings_for_run(rid) == []          # settled from the library


def test_a_source_used_for_several_claims_appears_once():
    """Deduped by URL, and scored by how decisive it proved."""
    from app.research.pipeline import Pipeline
    shared = Evidence(1, "e.com — Page", "https://e.com/a", "text")
    results = [
        Checked(claim=_claim("first", 9), evidence=[shared], via="web",
                verdict=VerdictOut(verdict="supported", confidence=5,
                                   reasoning="r", sources=[1])),
        Checked(claim=_claim("second", 9), evidence=[shared], via="web",
                verdict=VerdictOut(verdict="supported", confidence=9,
                                   reasoning="r", sources=[1])),
    ]
    best: dict = {}
    for r in results:                      # mirrors the grouping in the method
        for e in r.evidence:
            entry = best.setdefault(e.url, {"claims": [], "score": 0})
            entry["claims"].append(r.claim.text)
            if e.n in set(r.verdict.sources):
                entry["score"] = max(entry["score"], r.verdict.confidence)
    assert list(best) == ["https://e.com/a"]
    assert best["https://e.com/a"]["score"] == 9      # the decisive use wins
    assert len(best["https://e.com/a"]["claims"]) == 2


def test_an_uncited_page_is_scored_as_consulted_not_worthless():
    """Library passages are numbered first, so a verdict often cites only
    those and every web page scored 0 — which reads as 'judged worthless'
    rather than 'did not settle it'. Observed on a live run: nine sources,
    all zero."""
    from app.research.pipeline import _CONSULTED
    assert 0 < _CONSULTED < 5


@respx.mock
async def test_a_page_no_verdict_cited_is_still_kept_and_marked_consulted(data_dir):
    """The common case: library passages are numbered first, so a verdict
    often cites only those and the web pages it also read go uncited."""
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(
        200, json=sx_payload([sx_result("https://e.com/a", "Evidence page")])))
    respx.get("https://e.com/a").mock(
        return_value=httpx.Response(200, html=article("Evidence page")))
    # the verdict cites nothing at all
    llm = FakeLLM({"verify": [
        {"claims": [{"text": "MLX is faster", "importance": 9,
                     "checkable": True}]},
        {"verdict": "contested", "confidence": 6, "reasoning": "r",
         "quote": "", "sources": []}]})
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(), rag=_Rag([]),
                        llm_factory=lambda: llm)
    rid = orch.enqueue(RunParams(query="Claim check", depth=0, recency="all",
                                 origin="cli", kind="verify", document=DOC))
    await orch.execute_now(rid)

    rows = repo.findings_for_run(rid)
    assert len(rows) == 1
    assert rows[0]["summary"].startswith("Consulted while checking")
    assert rows[0]["relevance"] == 3            # consulted, not worthless


async def test_one_run_cannot_fill_every_evidence_slot():
    """Source titles share a parent run, so a per-title cap alone still let
    one run supply all six passages — observed live, with every citation
    pointing back at the same research run."""
    from app.research.verify import library_evidence
    hits = [{"run_id": "runA", "title": f"Source {i}", "text": f"a{i}",
             "score": 0.9 - i * 0.01} for i in range(8)]
    hits += [{"run_id": "runB", "title": "Other view", "text": "b", "score": 0.6},
             {"run_id": "runC", "title": "Third view", "text": "c", "score": 0.5}]
    ev = await library_evidence(_Rag(hits), "a claim")
    labels = " ".join(e.text for e in ev)
    assert sum(1 for e in ev if e.text.startswith("a")) == 3   # runA capped
    assert "b" in labels and "c" in labels                     # others got in


def test_library_evidence_is_trimmed_before_the_web_is_appended():
    """The model reaches for low numbers. If the library could not settle the
    claim, it should not also own every low slot."""
    from app.research.verify import renumber, trim_for_fallthrough
    library = [Evidence(i, f"lib{i}", f"/runs/{i}", "t") for i in range(1, 7)]
    kept = trim_for_fallthrough(library)
    assert [e.n for e in kept] == [1, 2, 3]
    web = [Evidence(0, f"web{i}", f"https://e.com/{i}", "t") for i in range(3)]
    combined = renumber(kept + web)
    assert [e.n for e in combined] == [1, 2, 3, 4, 5, 6]
    # web evidence now sits inside the range a verdict actually cites
    assert [e.n for e in combined if e.url.startswith("http")] == [4, 5, 6]
