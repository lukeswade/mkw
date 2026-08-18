"""Authority-site targeting: the planner and gap analysis are offered a
curated list of primary-document sites for site:-scoped queries, and the
per-round candidate budget that carries the wider net."""
from __future__ import annotations

import httpx
import respx

from app.db import Repo, connect
from app.models import RunParams
from app.research.orchestrator import Orchestrator
from app.research.pipeline import breadth_for_depth, candidates_per_round
from app.research.progress import ProgressBus
from tests.fake_llm import FakeLLM
from tests.test_next_level import _script
from tests.test_pipeline_e2e import SX, article, make_cfg, sx_payload, sx_result


def test_candidate_budget_grew_with_every_depth():
    for depth, expected in ((1, 14), (2, 14), (3, 18), (6, 22), (10, 30)):
        assert candidates_per_round(breadth_for_depth(depth)) == expected
    budgets = [candidates_per_round(breadth_for_depth(d))
               for d in range(1, 11)]
    assert budgets == sorted(budgets)  # monotone in depth


@respx.mock
async def test_planner_and_gap_are_offered_authority_sites(data_dir):
    cfg = make_cfg(data_dir)
    cfg.authority_sites = "charm.li — factory service manuals for cars"
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://example-a.com/article", "Article A")])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))

    prompts_seen: dict[str, str] = {}
    script = _script()

    def planner_capture(messages):
        prompts_seen["planner"] = messages[-1]["content"]
        return {"title": "T", "brief": "Investigate.", "subqueries": ["q1"]}

    def gap_capture(messages):
        prompts_seen["gap"] = messages[-1]["content"]
        return {"state_md": "s", "saturated": True, "next_queries": []}

    script["planner"] = [planner_capture]
    script["gap"] = [gap_capture]

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(script))
    run_id = orch.enqueue(RunParams(query="gx470 spark plugs", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)

    for kind in ("planner", "gap"):
        assert "charm.li — factory service manuals" in prompts_seen[kind]
        assert "site: operator" in prompts_seen[kind]


@respx.mock
async def test_no_authority_block_when_list_is_empty(data_dir):
    cfg = make_cfg(data_dir)
    cfg.authority_sites = ""
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://example-a.com/article", "Article A")])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))

    captured: list[str] = []
    script = _script()

    def planner_capture(messages):
        captured.append(messages[-1]["content"])
        return {"title": "T", "brief": "Investigate.", "subqueries": ["q1"]}

    script["planner"] = [planner_capture]
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(script))
    run_id = orch.enqueue(RunParams(query="gx470 spark plugs", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)
    assert captured and "site: operator" not in captured[0]


@respx.mock
async def test_site_scoped_query_passes_through_to_searxng(data_dir):
    cfg = make_cfg(data_dir)
    seen_queries: list[str] = []

    def handler(req):
        seen_queries.append(req.url.params["q"])
        return httpx.Response(200, json=sx_payload(
            [sx_result("https://charm.li/Lexus/2007/plug-spec", "Spark Plug Specs")]))

    respx.get(f"{SX}/search").mock(side_effect=handler)
    respx.get("https://charm.li/Lexus/2007/plug-spec").mock(
        return_value=httpx.Response(200, html=article("Spark Plug Specs")))

    script = _script()
    script["planner"] = [{"title": "T", "brief": "Investigate.",
                          "subqueries": ["site:charm.li 2007 GX470 spark plug"]}]
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(script))
    run_id = orch.enqueue(RunParams(query="gx470 spark plugs", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)

    assert any(q.startswith("site:charm.li") for q in seen_queries)
    assert {f["domain"] for f in repo.findings_for_run(run_id)} == {"charm.li"}


@respx.mock
async def test_site_scope_is_enforced_against_disobedient_engines(data_dir):
    """bing quietly drops site: and returns keyword junk — it must be filtered."""
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload([
        sx_result("https://charm.li/Lexus/2007/plug-spec", "Spark Plug Specs"),
        sx_result("https://www.opera.com/gx", "Opera GX browser"),
        sx_result("https://en.wikipedia.org/wiki/2007", "2007"),
    ])))

    from app.research.searcher import Searcher
    async with httpx.AsyncClient() as http:
        s = Searcher(SX, http, categories="general", max_concurrent=1)
        res = await s.search("site:charm.li 2007 GX470 spark plug", "all")
    assert [r.url for r in res] == ["https://charm.li/Lexus/2007/plug-spec"]

    # and without a site: operator nothing is filtered
    async with httpx.AsyncClient() as http:
        s = Searcher(SX, http, categories="general", max_concurrent=1)
        res = await s.search("2007 GX470 spark plug", "all")
    assert len(res) == 3


@respx.mock
async def test_empty_site_query_retries_with_first_five_words(data_dir):
    """Site-restricted indexes are thin — a long query that finds nothing is
    retried once, trimmed to its first five words."""
    seen: list[str] = []

    def handler(req):
        q = req.url.params["q"]
        seen.append(q)
        if q == "site:charm.li Lexus GX470 2007 spark plug":
            return httpx.Response(200, json=sx_payload(
                [sx_result("https://charm.li/Lexus/2007/plug-spec", "Specs")]))
        return httpx.Response(200, json=sx_payload([]))

    respx.get(f"{SX}/search").mock(side_effect=handler)
    from app.research.searcher import Searcher
    async with httpx.AsyncClient() as http:
        s = Searcher(SX, http, categories="general", max_concurrent=1)
        res = await s.search(
            "site:charm.li Lexus GX470 2007 spark plug replacement procedure "
            "torque specs", "all")
    assert len(seen) == 2 and seen[1] == "site:charm.li Lexus GX470 2007 spark plug"
    assert [r.url for r in res] == ["https://charm.li/Lexus/2007/plug-spec"]


@respx.mock
async def test_short_empty_site_query_is_not_retried(data_dir):
    seen: list[str] = []

    def handler(req):
        seen.append(req.url.params["q"])
        return httpx.Response(200, json=sx_payload([]))

    respx.get(f"{SX}/search").mock(side_effect=handler)
    from app.research.searcher import Searcher
    async with httpx.AsyncClient() as http:
        s = Searcher(SX, http, categories="general", max_concurrent=1)
        res = await s.search("site:charm.li GX470 plugs", "all")
    assert res == [] and len(seen) == 1
