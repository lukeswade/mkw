from app.research.pipeline import (
    breadth_for_depth,
    candidates_per_round,
    max_docs_for_depth,
    max_llm_calls_for_depth,
    saturation_patience,
)


def test_breadth_scale():
    assert breadth_for_depth(1) == 3
    assert breadth_for_depth(4) == 6
    assert breadth_for_depth(8) == 10
    assert breadth_for_depth(10) == 10  # capped


def test_caps_scale_with_depth():
    # High depths are deep-research territory: the source budget has to
    # allow Perplexity/OpenAI-DR-scale runs (50-250 sources), not a dozen.
    assert max_docs_for_depth(1) == 13
    assert max_docs_for_depth(3) == 45
    assert max_docs_for_depth(5) == 85
    assert max_docs_for_depth(10) == 220
    for depth in range(1, 11):
        # the notes-call ceiling must never strangle the source cap
        assert max_llm_calls_for_depth(depth) > max_docs_for_depth(depth) * 1.5


def test_candidate_budget_covers_the_source_cap():
    # rounds × candidates must make the source cap actually reachable
    for depth in range(1, 11):
        per_round = candidates_per_round(breadth_for_depth(depth))
        assert per_round * depth >= max_docs_for_depth(depth)


def test_saturation_needs_a_second_opinion_at_depth():
    assert saturation_patience(1) == 1
    assert saturation_patience(3) == 1
    assert saturation_patience(4) == 2
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
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=4,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)

    row = repo.get_run(run_id)
    assert row["status"] == "completed"
    # round 1's lone "saturated" verdict must NOT stop a depth-4 run;
    # the second consecutive verdict (round 2) does.
    assert row["stop_reason"].startswith("saturated")
    import json
    assert json.loads(row["stats_json"])["rounds"] == 2
