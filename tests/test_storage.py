from app.research.storage import (
    SERVABLE_RE,
    RunStore,
    atomic_write_text,
    slugify,
    validate_citations,
)


def test_slugify():
    assert slugify("Solid State Batteries!") == "solid-state-batteries"
    assert slugify("Ünïcödé Qùery") == "unicode-query"
    assert slugify("???") == "research"
    assert len(slugify("word " * 50)) <= 40
    assert not slugify("ends badly   ").endswith("-")


def test_run_store_create_and_collision(research_dir):
    a = RunStore.create(research_dir, "my query")
    b = RunStore.create(research_dir, "my query")  # same second → suffix
    assert a.dir != b.dir
    assert (a.dir / "findings").is_dir()
    assert (a.dir / "rounds").is_dir()
    assert b.dir.name.endswith("-2")


def test_atomic_write_no_tmp_left(tmp_path):
    p = tmp_path / "out.md"
    atomic_write_text(p, "hello")
    assert p.read_text() == "hello"
    assert not list(tmp_path.glob("*.tmp"))


def test_meta_roundtrip(research_dir):
    st = RunStore.create(research_dir, "q meta")
    st.write_meta({"status": "running"})
    st.update_meta(status="completed", stop_reason="saturated")
    meta = st.read_meta()
    assert meta["status"] == "completed"
    assert meta["stop_reason"] == "saturated"


def test_finding_and_round_paths(research_dir):
    st = RunStore.create(research_dir, "q files")
    rel = st.write_finding(1, "Some Great Article", "# notes")
    assert rel == "findings/001_some-great-article.md"
    assert (st.dir / rel).read_text() == "# notes"
    assert st.write_round(2, "round two") == "rounds/round-02.md"
    assert SERVABLE_RE.match(rel)
    assert SERVABLE_RE.match("rounds/round-02.md")


def test_servable_denies_traversal():
    for bad in ("../etc/passwd", "findings/../../x.md", "/etc/passwd",
                "findings/001_ok.md.tmp", "meta.json/../evil"):
        assert not SERVABLE_RE.match(bad)


def test_validate_citations():
    md, removed = validate_citations("Fact [1] and [2], bogus [9].", 2)
    assert removed == {9}
    assert "[9]" not in md
    assert "[1]" in md and "[2]" in md


def test_events_torn_line(research_dir):
    st = RunStore.create(research_dir, "q events")
    st.append_event({"seq": 1, "type": "status"})
    st.append_event({"seq": 2, "type": "round_start"})
    with st.events_path.open("a") as fh:
        fh.write('{"seq": 3, "type": "tru')  # simulated crash mid-write
    events = st.read_events()
    assert [e["seq"] for e in events] == [1, 2]
    assert [e["seq"] for e in st.read_events(after_seq=1)] == [2]
