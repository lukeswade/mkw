"""Server-rendered clocks show the configured zone, not the container's UTC."""
from app.timefmt import local_iso, local_time


def test_utc_stamps_render_in_central_time():
    assert local_time(0, "America/Chicago") == "18:00:00"                     # epoch is 18:00 CST the day before
    assert local_iso("2026-09-02T01:39:22+00:00", "America/Chicago") == "2026-09-01 20:39"
    assert local_iso("2026-09-02T01:39:22", "America/Chicago") == "2026-09-01 20:39"   # naive means UTC


def test_an_unknown_zone_falls_back_instead_of_breaking_the_page():
    assert local_iso("2026-09-02T01:39:22+00:00", "Mars/Olympus_Mons") == "2026-09-01 20:39"
    assert local_iso("", "America/Chicago") == ""


def test_the_run_log_line_carries_the_local_clock():
    from app.research.progress import format_event
    line = format_event({"type": "finding", "ts": 1788313162.0, "idx": 1, "title": "t",
                         "domain": "d", "relevance": 7}, tz="America/Chicago")
    assert line.startswith("[20:39:22]")


def test_skipped_sources_show_their_title_when_they_have_one():
    from app.research.progress import format_event
    line = format_event({"type": "source_skipped", "ts": 0, "url": "https://x/y",
                         "reason": "dropped at triage", "title": "FIX stock price"}, tz="UTC")
    assert line.endswith('(dropped at triage)  "FIX stock price"')


def test_a_kept_source_line_ends_with_its_score():
    from app.research.progress import format_event
    line = format_event({"type": "finding", "ts": 0, "idx": 3, "title": "Reel seat loose -- how to fix it in place?",
                         "domain": "stripersonline.com", "relevance": 8}, tz="UTC")
    assert line.endswith("Reel seat loose -- how to fix it in place? (stripersonline.com) · 8/10")
