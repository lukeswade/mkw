import asyncio

from app.research.progress import ProgressBus
from app.research.storage import RunStore


async def test_publish_replay_late_subscriber(research_dir):
    bus = ProgressBus()
    st = RunStore.create(research_dir, "bus test")
    bus.attach(st)

    bus.publish(st.run_id, "status", status="running")
    bus.publish(st.run_id, "round_start", round=1)

    replay, q = bus.subscribe(st.run_id, st)
    assert [e["seq"] for e in replay] == [1, 2]
    assert q is not None

    bus.publish(st.run_id, "finding", idx=1)
    live = await asyncio.wait_for(q.get(), 1)
    assert live["seq"] == 3 and live["type"] == "finding"
    bus.unsubscribe(st.run_id, q)


async def test_subscribe_inactive_run_replays_from_file(research_dir):
    bus = ProgressBus()
    st = RunStore.create(research_dir, "inactive")
    bus.attach(st)
    bus.publish(st.run_id, "status", status="running")
    bus.publish(st.run_id, "done")
    bus.detach(st.run_id)

    replay, q = bus.subscribe(st.run_id, st)
    assert q is None
    assert [e["type"] for e in replay] == ["status", "done"]


async def test_seq_continues_after_reattach(research_dir):
    bus = ProgressBus()
    st = RunStore.create(research_dir, "reattach")
    bus.attach(st)
    bus.publish(st.run_id, "status", status="running")
    bus.detach(st.run_id)

    bus.attach(st)  # e.g. process restart mid-run → retry writes to same log? (new run in practice, but seq must not collide)
    e = bus.publish(st.run_id, "status", status="resumed")
    assert e["seq"] == 2

    replay, _ = bus.subscribe(st.run_id, st, after_seq=0)
    assert [ev["seq"] for ev in replay] == [1, 2]


async def test_stream_events_are_live_only(research_dir):
    """Persisting a file write per generated token would be pathological."""
    bus = ProgressBus()
    st = RunStore.create(research_dir, "stream test")
    bus.attach(st)
    _replay, q = bus.subscribe(st.run_id, st)

    bus.publish(st.run_id, "phase", phase="chatting")
    for chunk in ("Hello", " ", "world"):
        bus.publish(st.run_id, "stream", chunk=chunk)
    bus.publish(st.run_id, "done", status="completed")

    live = [q.get_nowait()["type"] for _ in range(q.qsize())]
    assert live.count("stream") == 3          # subscribers still see them

    persisted = [e["type"] for e in st.read_events()]
    assert "stream" not in persisted          # but nothing hit the disk
    assert persisted == ["phase", "done"]
