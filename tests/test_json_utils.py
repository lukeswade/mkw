import pytest

from app.llm.json_utils import LLMJsonError, extract_json


def test_clean_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced():
    assert extract_json('Here you go:\n```json\n{"a": 1}\n```\nDone.') == {"a": 1}


def test_prose_wrapped():
    text = 'Sure! The answer is {"a": {"b": [1, 2]}} — hope that helps.'
    assert extract_json(text) == {"a": {"b": [1, 2]}}


def test_think_prefix():
    text = '<think>hmm {"fake": true} reasoning</think>{"a": 1}'
    assert extract_json(text) == {"a": 1}


def test_unclosed_think():
    with pytest.raises(LLMJsonError):
        extract_json('<think>never stops thinking {"a": 1}')


def test_trailing_comma_repaired():
    assert extract_json('{"a": [1, 2,], "b": 3,}') == {"a": [1, 2], "b": 3}


def test_curly_quotes_repaired():
    assert extract_json('{“a”: 1}') == {"a": 1}


def test_braces_inside_strings():
    assert extract_json('{"a": "closing } inside", "b": 1} trailing')["b"] == 1


def test_list_root():
    assert extract_json('[1, 2, 3]') == [1, 2, 3]


def test_empty_and_garbage():
    with pytest.raises(LLMJsonError):
        extract_json("")
    with pytest.raises(LLMJsonError):
        extract_json("no json here at all")
    with pytest.raises(LLMJsonError):
        extract_json('{"never": "closes"')


# ---- truncation salvage -----------------------------------------------------
# A response that hit the output ceiling was unsalvageable: the brace scan
# returned the unbalanced remainder, cheap repairs never close a brace, and the
# caller then paid a repair round-trip that re-sends the prompt plus the
# truncated text at the same budget and truncates again. Three full model calls
# and the source discarded — and notes are 101 of 115 calls in a deep run.

import pytest
from app.llm.json_utils import LLMJsonError, close_open, close_truncated, extract_json


@pytest.mark.parametrize("raw, expect", [
    # the payload field cut mid-sentence: the sentence is kept, nothing invented
    ('{"relevance": 8, "notes_md": "The pitch should be 25',
     {"relevance": 8, "notes_md": "The pitch should be 25"}),
    # stopped on a bare key: the incomplete element goes, the source survives
    ('{"relevance": 5, "summary": "x", "notes_md"',
     {"relevance": 5, "summary": "x"}),
    ('{"relevance": 6, "summary": "x", ', {"relevance": 6, "summary": "x"}),
    ('{"relevance": 5, "part_relevance": 1', {"relevance": 5, "part_relevance": 1}),
    ('{"a": {"b": [1, 2, 3', {"a": {"b": [1, 2, 3]}}),
])
def test_a_truncated_response_is_salvaged(raw, expect):
    assert extract_json(raw, allow_truncated=True) == expect


def test_a_half_written_list_item_is_dropped_not_guessed():
    got = extract_json('{"key_facts": [{"claim": "A.", "confidence": 9}, {"claim": "B.", "conf',
                       allow_truncated=True)
    assert got["key_facts"][0] == {"claim": "A.", "confidence": 9}
    assert all("conf" in f or "claim" in f for f in got["key_facts"])
    assert len(got["key_facts"]) <= 2


def test_valid_json_is_returned_byte_identical():
    for s in ('{"a": [1,2], "b": {"c": 3}}', '[1, 2, 3]', '{"s": "has {braces} inside"}'):
        assert close_open(s) == s
        assert close_truncated(s) == s
        assert extract_json(s) == __import__("json").loads(s)


def test_a_string_containing_braces_does_not_confuse_the_scan():
    assert extract_json('{"notes_md": "he said {this} and [that]", "relevance": 3',
                        allow_truncated=True
                        ) == {"notes_md": "he said {this} and [that]", "relevance": 3}


def test_still_unparseable_input_still_raises():
    for junk in ("", "   ", "no json here at all", "<think>only thinking</think>"):
        with pytest.raises(LLMJsonError):
            extract_json(junk)


def test_salvage_is_off_unless_the_caller_saw_a_truncated_response():
    """A merely malformed response must still get its repair round-trip: a
    half-empty object accepted here would be kept as a source with no notes."""
    with pytest.raises(LLMJsonError):
        extract_json('{"never": "closes"')
    assert extract_json('{"never": "closes"', allow_truncated=True) == {"never": "closes"}
