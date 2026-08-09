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
