"""Excerpt selection feeds every note-taking call, so a sloppy match here
quietly degrades the notes for every source in every run."""
from app.research.notes import HEADER_CHARS, select_excerpts

HEADER = "Battery Weekly — published 2026-03-02 by A. Reporter. " + "x" * 200
BODY_AI = "The company said it would maintain a plain chain of custody. " * 30
BODY_REAL = "Their AI system was retrained in 2026 on new telemetry data. " * 5


def test_short_keyword_does_not_match_inside_words():
    text = HEADER + BODY_AI
    out = select_excerpts(text, ["AI"])
    # "said", "maintain", "plain", "chain" must not count as hits for "AI"
    assert "retrained" not in out
    assert out == "" or "maintain a plain chain" not in out


def test_short_keyword_matches_the_real_occurrence():
    text = HEADER + BODY_AI + BODY_REAL
    out = select_excerpts(text, ["AI"])
    assert "AI system was retrained" in out


def test_header_is_always_kept_so_dates_survive():
    text = HEADER + BODY_AI + BODY_REAL
    out = select_excerpts(text, ["AI"])
    assert "published 2026-03-02" in out


def test_phrase_keyword_tolerates_whitespace():
    text = HEADER + "Research into solid\n   state batteries continues. " * 3
    out = select_excerpts(text, ["solid state"])
    assert "state batteries continues" in out


def test_punctuation_keyword_still_matches():
    text = HEADER + "Costs fell to $70/kWh last quarter. " * 3
    assert "$70/kWh" in select_excerpts(text, ["$70/kWh"])


def test_no_keywords_or_no_hits_returns_empty():
    assert select_excerpts("some text", []) == ""
    assert select_excerpts("some text", ["nonexistentterm"]) == ""


def test_respects_char_budget():
    text = ("keyword " + "filler " * 400) * 20
    out = select_excerpts(text, ["keyword"], max_chars=4000)
    assert len(out) <= 4000 + HEADER_CHARS + 200
