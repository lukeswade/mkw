"""Evergreen refresh: the flag stays on the original run, and a refresh is due
only when no child is in flight and none was created inside the window."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.db import Repo, connect
from app.refresh_worker import REFRESH_INTERVAL_HOURS, refresh_due_runs
from tests.test_web import make_app, seed_completed_run


def _ts(hours_ago: float) -> str:
    return (datetime.now(timezone.utc)
            - timedelta(hours=hours_ago)).isoformat(timespec="seconds")


class FakeOrch:
    def __init__(self, repo):
        self.repo = repo
        self.enqueued = []

    def enqueue(self, params):
        run_id = f"child-{len(self.enqueued)}"
        self.repo.create_run(run_id=run_id, query=params.query,
                             depth=params.depth, recency=params.recency,
                             dir=run_id, origin=params.origin,
                             parent_run_id=params.parent_run_id)
        self.enqueued.append(params)
        return run_id


@pytest.fixture
def repo(data_dir):
    return Repo(connect(data_dir / "app.sqlite3"))


def _evergreen_run(repo, run_id="parent", *, finished_hours_ago=48):
    repo.create_run(run_id=run_id, query="battery news", depth=2,
                    recency="all", dir=run_id, origin="web")
    repo.update_run(run_id, status="completed", evergreen=True,
                    finished_at=_ts(finished_hours_ago))
    return run_id


async def test_due_run_is_refreshed_and_keeps_its_flag(repo):
    parent = _evergreen_run(repo)
    orch = FakeOrch(repo)

    assert await refresh_due_runs(orch, repo) == 1
    params = orch.enqueued[0]
    assert params.parent_run_id == parent
    assert params.recency == "month"      # refreshes look for what's new
    assert params.query == "battery news"
    # the flag stays put, so a failed child can't break the chain
    assert repo.get_run(parent)["evergreen"] == 1


async def test_not_due_while_a_child_is_in_flight(repo):
    _evergreen_run(repo)
    orch = FakeOrch(repo)
    await refresh_due_runs(orch, repo)          # creates a queued child
    assert await refresh_due_runs(orch, repo) == 0


async def test_not_due_again_inside_the_window(repo):
    parent = _evergreen_run(repo)
    orch = FakeOrch(repo)
    await refresh_due_runs(orch, repo)
    repo.update_run("child-0", status="completed")
    assert await refresh_due_runs(orch, repo) == 0


async def test_due_again_once_the_window_passes(repo):
    _evergreen_run(repo)
    orch = FakeOrch(repo)
    await refresh_due_runs(orch, repo)
    repo.conn.execute(
        "UPDATE runs SET status='completed', created_at=? WHERE id='child-0'",
        (_ts(REFRESH_INTERVAL_HOURS + 2),))
    repo.conn.commit()
    assert await refresh_due_runs(orch, repo) == 1


async def test_freshly_finished_run_is_not_immediately_due(repo):
    _evergreen_run(repo, finished_hours_ago=1)
    assert await refresh_due_runs(FakeOrch(repo), repo) == 0


async def test_non_evergreen_runs_are_ignored(repo):
    repo.create_run(run_id="plain", query="q", depth=1, recency="all",
                    dir="plain", origin="web")
    repo.update_run("plain", status="completed", finished_at=_ts(72))
    assert await refresh_due_runs(FakeOrch(repo), repo) == 0


def test_toggle_endpoint_flips_the_flag(data_dir, monkeypatch):
    app, cfg = make_app(data_dir, monkeypatch)
    run_id = seed_completed_run(cfg)
    repo = Repo(connect(cfg.db_path))
    with TestClient(app) as client:
        r = client.post(f"/runs/{run_id}/evergreen",
                        headers={"HX-Request": "true"})
        assert r.status_code == 200
        assert "Evergreen on" in r.text          # header partial swapped back
        assert repo.get_run(run_id)["evergreen"] == 1

        client.post(f"/runs/{run_id}/evergreen", headers={"HX-Request": "true"})
        assert repo.get_run(run_id)["evergreen"] == 0

        assert client.post("/runs/missing/evergreen").status_code == 404


async def test_refresh_inherits_the_owner(repo):
    parent = _evergreen_run(repo)
    repo.update_run(parent, created_by="matt.wade")
    orch = FakeOrch(repo)
    await refresh_due_runs(orch, repo)
    assert orch.enqueued[0].created_by == "matt.wade"
