"""The pre-flight estimate has to be honest: monotonic in depth, and clearly
uncalibrated until there is real history to learn from."""
import json

from app.db import Repo, connect, utcnow
from app.research.estimate import DEFAULT_SECONDS_PER_SOURCE, estimate_run


def _repo(data_dir):
    return Repo(connect(data_dir / "app.sqlite3"))


def _completed_run(repo, run_id, *, kept, seconds, cost=None, skipped=None,
                   depth=3):
    repo.create_run(run_id=run_id, query="q", depth=depth, recency="all",
                    dir=run_id, origin="web")
    llm = {"calls": 10, "prompt_tokens": 1, "completion_tokens": 1}
    if cost is not None:
        llm["est_cost_usd"] = cost
    stats = {"sources_kept": kept, "llm": llm}
    if skipped is not None:
        stats["sources_skipped"] = skipped
    repo.update_run(
        run_id, status="completed",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at=f"2026-01-01T00:{seconds // 60:02d}:{seconds % 60:02d}+00:00",
        stats_json=json.dumps(stats))


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


def test_the_keep_rate_is_measured_once_there_is_history(data_dir):
    """KEEP_RATE=0.55 was the whole estimate; real runs kept 2%-58% of what
    they fetched. With history the measured rate replaces the guess."""
    from app.research.estimate import KEEP_RATE
    empty = estimate_run(_repo(data_dir), 8)
    repo = _repo(data_dir)
    for i in range(3):                       # 10% keep rate, at another depth
        _completed_run(repo, f"k{i}", kept=2, skipped=18, seconds=60, depth=5)
    est = estimate_run(repo, 8)              # depth 8 has no history of its own
    assert est.keep_rate == 0.1 and empty.keep_rate == KEEP_RATE
    assert est.sources_high < empty.sources_high
    assert not est.depth_calibrated


def test_the_band_is_what_runs_at_that_depth_actually_kept(data_dir):
    """Depth 10 promised 23-46 sources against real outcomes of 4-58 with a
    median of 14. The middle half of this depth's own history is the honest
    band, and a wide one is the truth rather than a failure."""
    repo = _repo(data_dir)
    for i, kept in enumerate((4, 8, 19, 40)):
        _completed_run(repo, f"d{i}", kept=kept, seconds=600, depth=10)
    est = estimate_run(repo, 10)
    assert est.depth_calibrated and est.depth_samples == 4
    assert (est.sources_low, est.sources_high) == (8, 19)
    assert est.sources_label == "8–19 sources"
    # another depth is untouched by depth-10 history
    assert not estimate_run(repo, 3).depth_calibrated


def test_identical_history_reads_as_one_figure(data_dir):
    repo = _repo(data_dir)
    for i in range(2):
        _completed_run(repo, f"e{i}", kept=10, seconds=60)
    assert estimate_run(repo, 3).sources_label == "~10 sources"


def test_briefs_and_quick_answers_do_not_pollute_research_history(data_dir):
    repo = _repo(data_dir)
    repo.create_run(run_id="b", query="Brief: x", depth=4, recency="week",
                    dir="b", origin="web", kind="brief")
    repo.update_run("b", status="completed",
                    started_at="2026-01-01T00:00:00+00:00",
                    finished_at="2026-01-01T00:01:00+00:00",
                    stats_json=json.dumps({"sources_kept": 50, "sources_skipped": 0,
                                           "llm": {"calls": 1}}))
    est = estimate_run(repo, 4)
    assert not est.calibrated and not est.depth_calibrated
