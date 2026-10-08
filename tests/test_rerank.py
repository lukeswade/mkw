"""The reranker: one client, three uses, each off until switched on.

Claim checks are the measured use (74% -> 85% shown the right source on this
library, 2026-10-07). Search ordering and passage picking are on trial.
"""
from __future__ import annotations

import json

import httpx
import respx

from app.config import Settings
from app.rag.rerank import Reranker, make_reranker
from app.research.verify import library_evidence

_URL = "http://localhost:8000/v1/rerank"


def _scored(scores: list[float]) -> dict:
    """A /rerank response: results best first, each carrying its input index."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return {"results": [{"index": i, "relevance_score": scores[i]} for i in order]}


# ---- the client -------------------------------------------------------------------

@respx.mock
async def test_scores_come_back_in_the_documents_order():
    route = respx.post(_URL).mock(return_value=httpx.Response(
        200, json=_scored([0.1, 0.9, 0.5])))
    r = Reranker("http://localhost:8000/v1", "Qwen3-Reranker-0.6B-mxfp8", "k")
    assert await r.scores("q", ["a", "b", "c"]) == [0.1, 0.9, 0.5]
    assert await r.order("q", ["a", "b", "c"]) == [1, 2, 0]
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "Qwen3-Reranker-0.6B-mxfp8"
    assert sent["query"] == "q" and sent["documents"] == ["a", "b", "c"]
    assert route.calls[0].request.headers["authorization"] == "Bearer k"


@respx.mock
async def test_long_lists_go_in_batches():
    def answer(request):
        docs = json.loads(request.content)["documents"]
        return httpx.Response(200, json=_scored([float(len(d)) for d in docs]))
    route = respx.post(_URL).mock(side_effect=answer)
    docs = ["x" * (i + 1) for i in range(150)]
    assert await Reranker("http://localhost:8000/v1", "m").scores("q", docs) == [
        float(i + 1) for i in range(150)]
    assert route.call_count == 3                      # 64 + 64 + 22


@respx.mock
async def test_a_server_that_cannot_rerank_gives_no_scores_not_an_error():
    respx.post(_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "x"}}))
    assert await Reranker("http://localhost:8000/v1", "m").scores("q", ["a"]) is None


@respx.mock
async def test_partial_scores_count_as_none():
    respx.post(_URL).mock(return_value=httpx.Response(
        200, json={"results": [{"index": 0, "relevance_score": 0.5}]}))
    assert await Reranker("http://localhost:8000/v1", "m").order("q", ["a", "b"]) is None


def test_each_use_is_off_until_a_model_is_named_and_its_switch_is_on():
    base = dict(llm_provider="local", llm_model="m",
                llm_base_url="http://llm:8000/v1", llm_api_key="k")
    assert make_reranker(Settings(**base), "claims") is None
    cfg = Settings(**base, rerank_model="Qwen3-Reranker-0.6B-mxfp8")
    assert make_reranker(cfg, "claims") is not None            # on once a model is named
    assert make_reranker(cfg, "search") is None                # on trial: off by default
    assert make_reranker(cfg, "passages") is None
    assert make_reranker(cfg, "claims").base_url == "http://llm:8000/v1"
    emb = Settings(**base, rerank_model="m", embedding_base_url="http://emb:9000/v1")
    assert make_reranker(emb, "claims").base_url == "http://emb:9000/v1"


# ---- claim checks: the library's evidence ----------------------------------------

class _Rag:
    def __init__(self, hits):
        self.hits = hits
        self.asked: dict = {}

    async def semantic_search(self, query, limit=20, text_chars=400):
        self.asked = {"limit": limit, "text_chars": text_chars}
        return self.hits[:limit]


class _Order:
    """A reranker double: `pick(docs)` gives the order, None means no scores."""
    def __init__(self, pick):
        self.pick = pick
        self.seen: list[str] = []

    async def order(self, query, docs):
        self.seen = list(docs)
        return None if self.pick is None else self.pick(docs)


def _hit(i: int, run: str, title: str, score: float = 0.9) -> dict:
    return {"run_id": run, "title": title, "text": f"chunk {i}", "score": score}


async def test_claim_evidence_is_reranked_from_whole_chunks_before_the_caps():
    rag = _Rag([_hit(i, f"r{i}", f"t{i}", 0.9 - i * 0.005) for i in range(40)])
    reverse = _Order(lambda docs: list(range(len(docs) - 1, -1, -1)))
    ev = await library_evidence(rag, "a claim", reranker=reverse)
    assert rag.asked == {"limit": 50, "text_chars": None}
    assert [e.text for e in ev] == [f"chunk {i}" for i in (39, 38, 37, 36, 35, 34)]


async def test_the_caps_still_hold_after_reranking():
    hits = [_hit(i, "same-run", f"t{i % 2}") for i in range(10)] + [_hit(10, "r2", "other")]
    ev = await library_evidence(_Rag(hits), "claim",
                                reranker=_Order(lambda d: list(range(len(d)))))
    # three from one run at most, then the next run gets its turn
    assert [e.text for e in ev] == ["chunk 0", "chunk 1", "chunk 2", "chunk 10"]


async def test_the_floor_applies_before_reranking():
    spy = _Order(lambda d: list(range(len(d))))
    ev = await library_evidence(
        _Rag([_hit(0, "r0", "weak", 0.2), _hit(1, "r1", "strong", 0.8)]), "claim",
        reranker=spy)
    assert spy.seen == ["chunk 1"] and [e.text for e in ev] == ["chunk 1"]


async def test_without_scores_the_embedding_order_and_pool_stand():
    rag = _Rag([_hit(i, f"r{i}", f"t{i}", 0.9 - i * 0.005) for i in range(40)])
    ev = await library_evidence(rag, "claim", reranker=_Order(None))
    assert [e.text for e in ev] == [f"chunk {i}" for i in range(6)]


async def test_no_reranker_keeps_todays_pool_but_reads_whole_chunks():
    rag = _Rag([_hit(i, f"r{i}", f"t{i}") for i in range(40)])
    await library_evidence(rag, "claim")
    assert rag.asked == {"limit": 24, "text_chars": None}


# ---- search ordering (on trial) -------------------------------------------------

def _run_with(cfg, s):
    from app.db import Repo, connect
    from app.models import RunParams
    from app.research.orchestrator import Orchestrator
    from app.research.progress import ProgressBus
    from tests.fake_llm import FakeLLM
    repo = Repo(connect(cfg.db_path))
    llm = FakeLLM(s)
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(), llm_factory=lambda: llm)
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="cli"))
    return repo, orch, run_id


def _mock_round():
    from tests.test_pipeline_e2e import SX, article, sx_payload, sx_result
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result(f"https://example-{c}.com/article", f"Article {c.upper()}")
         for c in "abcde"])))
    for c in "abcde":
        respx.get(f"https://example-{c}.com/article").mock(
            return_value=httpx.Response(200, html=article(f"Article {c.upper()}")))


def _triage_order(prompt: str) -> list[str]:
    import re
    return re.findall(r"Article ([A-E])", prompt)


@respx.mock
async def test_search_results_reach_triage_in_the_rerankers_order(data_dir):
    from tests.test_pipeline_e2e import make_cfg, script
    cfg = make_cfg(data_dir)
    cfg.rerank_model, cfg.rerank_search = "m", True
    cfg.embedding_base_url = "http://rerank.test/v1"
    _mock_round()

    def prefer_late_letters(request):
        docs = json.loads(request.content)["documents"]
        return httpx.Response(200, json=_scored([float(ord(d.split()[1][0])) for d in docs]))
    respx.post("http://rerank.test/v1/rerank").mock(side_effect=prefer_late_letters)
    seen: list[str] = []

    def keep_all(messages):
        seen.append(messages[-1]["content"])
        return {"drop": []}
    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["triage"] = [keep_all]
    repo, orch, run_id = _run_with(cfg, s)
    await orch.execute_now(run_id)
    order = _triage_order(seen[0])
    assert order[:5] == ["E", "D", "C", "B", "A"]


@respx.mock
async def test_search_order_is_unchanged_when_the_reranker_cannot_answer(data_dir):
    from tests.test_pipeline_e2e import make_cfg, script
    cfg = make_cfg(data_dir)
    cfg.rerank_model, cfg.rerank_search = "m", True
    cfg.embedding_base_url = "http://rerank.test/v1"
    _mock_round()
    respx.post("http://rerank.test/v1/rerank").mock(return_value=httpx.Response(503))
    seen: list[str] = []

    def keep_all(messages):
        seen.append(messages[-1]["content"])
        return {"drop": []}
    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["triage"] = [keep_all]
    repo, orch, run_id = _run_with(cfg, s)
    await orch.execute_now(run_id)
    assert repo.get_run(run_id)["status"] == "completed"
    assert _triage_order(seen[0])[:5] == ["A", "B", "C", "D", "E"]


# ---- passage picking (on trial) -------------------------------------------------

from app.llm.client import PromptExceedsFreeMemory  # noqa: E402
from app.research.notes import _passages, _rerank_cut, take_notes  # noqa: E402
from tests.fake_llm import FakeLLM  # noqa: E402

_NOTES = {"relevance": 8, "summary": "s", "notes_md": "n", "key_facts": [],
          "published_date": None}


class _Prefers:
    """A reranker double that scores pieces containing `word` first."""
    def __init__(self, word: str | None):
        self.word = word

    async def order(self, query, docs):
        if self.word is None:
            return None
        return sorted(range(len(docs)), key=lambda i: (self.word not in docs[i], i))


def test_a_page_is_cut_into_pieces_at_sentence_ends():
    page = "One short sentence here. " * 400
    pieces = _passages(page, size=2_000)
    assert all(len(p) <= 2_000 for p in pieces)
    assert all(p.endswith(".") for p in pieces[:-1])
    assert "".join(pieces).replace(" ", "") == page.replace(" ", "")


async def test_the_best_pieces_are_kept_in_page_order():
    page = ("filler words without meaning. " * 70 + "\n\n") * 6
    page = page.replace("filler words without meaning. ", "the torque spec is 15 ft-lb. ", 1)
    cut = await _rerank_cut(page, "torque", 4_500, _Prefers("torque"))
    assert cut.startswith("the torque spec is 15 ft-lb.")
    assert len(cut) <= 4_600
    assert await _rerank_cut(page, "torque", 4_500, _Prefers(None)) is None


_BIG = ("padding sentence with nothing in it. " * 1600
        + "The torque spec is 15 ft-lb on the 2UZ. "
        + "more padding after the fact. " * 600)          # ~77K: over the 56K budget


class _Spy(_Prefers):
    """A _Prefers that counts how often it is asked."""
    def __init__(self, word):
        super().__init__(word)
        self.asked = 0

    async def order(self, query, docs):
        self.asked += 1
        return await super().order(query, docs)


async def test_an_oversized_page_keeps_the_keyword_cut_without_the_reranker():
    """2026-10-08: about 24 s per 64K page beside the notes model, for no gain
    in the trial; the reranker is kept for cuts a memory refusal forces."""
    seen: list[str] = []

    def capture(messages):
        seen.append(messages[-1]["content"])
        return dict(_NOTES)
    spy = _Spy("torque")
    await take_notes(FakeLLM({"notes": [capture]}), brief="torque specs",
                     recency_desc="any", today="t", url="u", title="t",
                     detected_date=None, text=_BIG, keywords=None, reranker=spy)
    assert spy.asked == 0
    assert "[... document truncated ...]" in seen[0]   # clip_text's head and tail


async def test_a_memory_shrink_without_scores_keeps_the_keyword_cut():
    seen: list[str] = []

    def refuse_then_read(messages):
        seen.append(messages[-1]["content"])
        if len(seen) == 1:
            return PromptExceedsFreeMemory("not enough free memory", 0.5)
        return dict(_NOTES)
    page = ("padding sentence with nothing in it. " * 900
            + "The torque spec is 15 ft-lb on the 2UZ. " + "more padding. " * 300)
    out = await take_notes(FakeLLM({"notes": [refuse_then_read]}), brief="torque",
                           recency_desc="any", today="t", url="u", title="t",
                           detected_date=None, text=page, keywords=["torque"],
                           reranker=_Prefers(None))
    assert out is not None and "torque spec is 15 ft-lb" in seen[1]   # keyword window
    assert len(seen[1]) < len(seen[0])


async def test_a_memory_shrink_keeps_the_rerankers_pieces():
    seen: list[str] = []

    def refuse_then_read(messages):
        seen.append(messages[-1]["content"])
        if len(seen) == 1:
            return PromptExceedsFreeMemory("not enough free memory", 0.5)
        return dict(_NOTES)
    page = ("padding sentence with nothing in it. " * 900
            + "The torque spec is 15 ft-lb on the 2UZ. " + "more padding. " * 300)
    out = await take_notes(FakeLLM({"notes": [refuse_then_read]}), brief="torque",
                           recency_desc="any", today="t", url="u", title="t",
                           detected_date=None, text=page, keywords=None,
                           reranker=_Prefers("torque"))
    assert out is not None and "torque spec is 15 ft-lb" in seen[1]
    assert len(seen[1]) < len(seen[0]) * 0.5


@respx.mock
async def test_the_search_reranker_reads_a_title_and_the_opening_of_its_snippet(data_dir):
    """Science engines return whole abstracts as snippets: 299 results came to
    133K tokens and 49.8 s of reranking on 2026-10-07."""
    from tests.test_pipeline_e2e import SX, article, make_cfg, script, sx_payload
    cfg = make_cfg(data_dir)
    cfg.rerank_model, cfg.rerank_search = "m", True
    cfg.embedding_base_url = "http://rerank.test/v1"
    abstract = "Matryoshka embeddings nest coarse-to-fine information. " * 60
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [{"url": f"https://example-{c}.com/article", "title": f"Article {c.upper()}",
          "content": abstract, "engine": "test", "publishedDate": None}
         for c in "abcde"])))
    for c in "abcde":
        respx.get(f"https://example-{c}.com/article").mock(
            return_value=httpx.Response(200, html=article(f"Article {c.upper()}")))
    sent: list[str] = []

    def answer(request):
        docs = json.loads(request.content)["documents"]
        sent.extend(docs)
        return httpx.Response(200, json=_scored([1.0] * len(docs)))
    respx.post("http://rerank.test/v1/rerank").mock(side_effect=answer)
    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    _repo, orch, run_id = _run_with(cfg, s)
    await orch.execute_now(run_id)
    assert len(abstract) > 3_000 and sent
    assert all(d.startswith("Article ") and len(d) <= len("Article A\n") + 300
               for d in sent)
