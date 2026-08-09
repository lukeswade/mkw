from app.research.pipeline import (
    breadth_for_depth,
    max_docs_for_depth,
    max_llm_calls_for_depth,
)


def test_breadth_scale():
    assert breadth_for_depth(1) == 3
    assert breadth_for_depth(4) == 6
    assert breadth_for_depth(6) == 8
    assert breadth_for_depth(10) == 8  # capped


def test_caps_scale_with_depth():
    assert max_docs_for_depth(1) == 12
    assert max_docs_for_depth(10) == 120
    assert max_llm_calls_for_depth(1) == 35
    assert max_llm_calls_for_depth(10) == 170
