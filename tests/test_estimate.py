"""The pre-flight estimate has to be honest: monotonic in depth, and clearly
uncalibrated until there is real history to learn from."""
import json

from app.db import Repo, connect, utcnow
from app.research.estimate import DEFAULT_SECONDS_PER_SOURCE, estimate_run


def _repo(data_dir):
    return Repo(connect(data_dir / "app.sqlite3"))


def _completed_run(repo, run_id, *, kept, seconds, cost=None):
    repo.create_run(run_id=run_id, query="q", depth=3, recency="all",
                    dir=run_id, origin="web")
    llm = {"calls": 10, "prompt_tokens": 1, "completion_tokens": 1}
    if cost is not None:
        llm["est_cost_usd"] = cost
    repo.update_run(
        run_id, status="completed",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at=f"2026-01-01T00:{seconds // 60:02d}:{seconds % 60:02d}+00:00",
        stats_json=json.dumps({"sources_kept": kept, "llm": llm}))


def test_depth_zero_is_quick_chat(data_dir):
    est = estimate_run(_repo(data_dir), 0)
    assert est.sources_high == 0
    assert est.sources_label == "no web search"
    assert est.duration_label == "seconds"


def test_estimate_grows_with_depth(data_dir):
    repo = _repo(data_dir)
    seq = [estimate_run(repo, d) for d in range(1, 11)]
    for a, b in zip(seq, seq[1:]):
        assert b.sources_high >= a.sources_high
        assert b.llm_calls >= a.llm_calls
        assert b.seconds >= a.seconds


def test_uncalibrated_until_enough_history(data_dir):
    repo = _repo(data_dir)
    est = estimate_run(repo, 3)
    assert not est.calibrated
    assert est.samples == 0
    midpoint = (est.sources_low + est.sources_high) / 2
    assert est.seconds == midpoint * DEFAULT_SECONDS_PER_SOURCE
    assert est.cost_usd is None


def test_calibrates_from_completed_runs(data_dir):
    repo = _repo(data_dir)
    # 10 sources in 300s => 30s per source, three times slower than the default
    for i in range(3):
        _completed_run(repo, f"r{i}", kept=10, seconds=300, cost=0.20)
    est = estimate_run(repo, 3)
    assert est.calibrated and est.samples == 3
    midpoint = (est.sources_low + est.sources_high) / 2
    assert est.seconds == midpoint * 30.0
    assert est.cost_usd == round(midpoint * 0.02, 3)


def test_runs_without_usable_timing_are_ignored(data_dir):
    repo = _repo(data_dir)
    repo.create_run(run_id="nostats", query="q", depth=3, recency="all",
                    dir="nostats", origin="web")
    repo.update_run("nostats", status="completed", finished_at=utcnow())
    assert not estimate_run(repo, 3).calibrated


def test_duration_label_reads_naturally(data_dir):
    repo = _repo(data_dir)
    for i in range(3):
        _completed_run(repo, f"s{i}", kept=10, seconds=60)
    assert "min" in estimate_run(repo, 5).duration_label
