from app.research.pipeline import (
    breadth_for_depth,
    candidates_per_round,
    max_docs_for_depth,
    max_llm_calls_for_depth,
    rounds_for_depth,
    saturation_patience,
)


def test_half_step_scale_mapping():
    # UI depth n ≈ old depth n/2: 2 = one research unit, 6 = three, 10 = five
    assert rounds_for_depth(1) == 1
    assert rounds_for_depth(2) == 1
    assert rounds_for_depth(3) == 2
    assert rounds_for_depth(6) == 3
    assert rounds_for_depth(10) == 5


def test_breadth_scale():
    assert breadth_for_depth(1) == 3
    assert breadth_for_depth(2) == 3
    assert breadth_for_depth(4) == 4
    assert breadth_for_depth(6) == 5
    assert breadth_for_depth(10) == 7


def test_caps_scale_with_depth():
    # The top of the scale is deep-research territory: source budgets in the
    # dozens, with a floor so even a quick look can cite a handful.
    assert max_docs_for_depth(1) == 8
    assert max_docs_for_depth(2) == 13
    assert max_docs_for_depth(3) == 20
    assert max_docs_for_depth(6) == 45
    assert max_docs_for_depth(10) == 85
    for depth in range(1, 11):
        # the notes-call ceiling must never strangle the source cap
        assert max_llm_calls_for_depth(depth) > max_docs_for_depth(depth) * 1.5


def test_candidate_budget_covers_the_source_cap():
    # rounds × candidates must make the source cap actually reachable
    for depth in range(1, 11):
        per_round = candidates_per_round(breadth_for_depth(depth))
        assert per_round * rounds_for_depth(depth) >= max_docs_for_depth(depth)


def test_saturation_needs_a_second_opinion_at_depth():
    assert saturation_patience(2) == 1
    assert saturation_patience(6) == 1
    assert saturation_patience(7) == 2
    assert saturation_patience(10) == 2


# ---- saturation patience, end to end -------------------------------------------

import httpx
import respx

from app.db import Repo, connect
from app.models import RunParams
from app.research.orchestrator import Orchestrator
from app.research.progress import ProgressBus
from tests.fake_llm import FakeLLM
from tests.test_pipeline_e2e import SX, article, make_cfg, script, sx_payload, sx_result


@respx.mock
async def test_deep_run_needs_two_saturated_verdicts(data_dir):
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://example-a.com/article", "Article A")])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))

    s = script([
        {"state_md": "s", "saturated": True, "next_queries": ["q-round-two"]},
        {"state_md": "s", "saturated": True, "next_queries": []},
    ])
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(s))
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=8,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)

    row = repo.get_run(run_id)
    assert row["status"] == "completed"
    # round 1's lone "saturated" verdict must NOT stop a deep run;
    # the second consecutive verdict (round 2) does.
    assert row["stop_reason"].startswith("saturated")
    import json
    assert json.loads(row["stats_json"])["rounds"] == 2


# ---- depth 0: snippet-grounded AI summary ---------------------------------------

@respx.mock
async def test_depth_zero_serves_a_cited_ai_summary(data_dir):
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload([
        sx_result("https://example-a.com/article", "Article A"),
        sx_result("https://example-b.org/report", "Report B"),
    ])))
    # NOTE: no page routes — depth 0 must never fetch a page

    captured: list[str] = []

    def chat_capture(messages):
        captured.append(messages[-1]["content"])
        return "Direct answer from the snippets [1]. Bogus [7] goes.\n"

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM({"chat": [chat_capture]}))
    run_id = orch.enqueue(RunParams(query="what is a spark plug gap", depth=0,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)

    assert repo.get_run(run_id)["status"] == "completed"
    assert "Article A" in captured[0]          # snippets reached the model
    overview = (cfg.research_dir / run_id / "overview.md").read_text()
    assert "Direct answer from the snippets [1]" in overview
    assert "[7]" not in overview               # invalid citation stripped
    assert "## Sources" in overview
    assert "https://example-b.org/report" in overview


@respx.mock
async def test_depth_zero_falls_back_to_plain_chat_when_search_is_empty(data_dir):
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(
        200, json=sx_payload([])))
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(
                            {"chat": ["Just the model's answer."]}))
    run_id = orch.enqueue(RunParams(query="what is a spark plug gap", depth=0,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)
    overview = (cfg.research_dir / run_id / "overview.md").read_text()
    assert "Just the model's answer." in overview
    assert "## Sources" not in overview
