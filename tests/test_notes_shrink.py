"""A page the server lacks the memory for is read shorter, not skipped.

2026-10-07: with the Mac's RAM tight, oMLX refused 17 of one round's 18 pages
because each long page needed ~850 MB it did not have, though it had room for
most of each. A notes call now answers that refusal with a shorter read of the
same page, keyword passages first, sized from the room the server reported,
and skips the page only when even a short read would not fit.
"""
from __future__ import annotations

from app.llm.client import LLMError, PromptExceedsFreeMemory
from app.research.notes import take_notes
from tests.fake_llm import FakeLLM

_PAYLOAD = {"relevance": 8, "summary": "s", "notes_md": "n",
            "key_facts": [], "published_date": None}
# ~42K chars: under the 56K budget, so the first read sends the whole page
_PAGE = ("padding sentence with nothing in it. " * 900
         + " The torque spec is 15 ft-lb on the 2UZ. "
         + "more padding after the fact. " * 300)


def _refusing(times: int, fits: float | None, prompts: list[str]):
    """Refuse the first `times` notes calls for memory, then answer."""
    def answer(messages):
        prompts.append(messages[-1]["content"])
        if len(prompts) <= times:
            return PromptExceedsFreeMemory(
                "not enough free memory for a prompt this long", fits)
        return dict(_PAYLOAD)
    return answer


async def _notes(llm, keywords: list[str] | None = ["torque"]):
    return await take_notes(llm, brief="b", recency_desc="any", today="t",
                            url="u", title="t", detected_date=None,
                            text=_PAGE, keywords=keywords)


async def test_a_page_refused_for_memory_is_read_shorter_not_skipped():
    prompts: list[str] = []
    out = await _notes(FakeLLM({"notes": [_refusing(1, 0.70, prompts)]}))
    assert out is not None and len(prompts) == 2
    # aimed under the room the server reported: 0.70 x 0.8 of the page
    assert len(prompts[1]) < len(prompts[0]) * 0.65


async def test_the_keyword_passage_survives_the_shrink():
    prompts: list[str] = []
    await _notes(FakeLLM({"notes": [_refusing(1, 0.3, prompts)]}))
    assert "torque spec is 15 ft-lb" in prompts[1]
    assert "padding sentence" in prompts[1]      # the page's opening fills the rest


async def test_without_numbers_the_page_is_halved():
    prompts: list[str] = []
    out = await _notes(FakeLLM({"notes": [_refusing(1, None, prompts)]}),
                       keywords=None)
    assert out is not None
    assert len(prompts[1]) < len(prompts[0]) * 0.6


async def test_a_page_that_never_fits_is_skipped_after_two_shorter_reads():
    prompts: list[str] = []
    out = await _notes(FakeLLM({"notes": [_refusing(99, 0.5, prompts)]}))
    assert out is None and len(prompts) == 3


async def test_no_shorter_read_when_too_little_would_fit():
    prompts: list[str] = []
    out = await _notes(FakeLLM({"notes": [_refusing(99, 0.05, prompts)]}))
    assert out is None and len(prompts) == 1


async def test_other_failures_are_not_retried_shorter():
    prompts: list[str] = []

    def fail(messages):
        prompts.append(messages[-1]["content"])
        return LLMError("exceeded the 600s ceiling")

    out = await _notes(FakeLLM({"notes": [fail]}))
    assert out is None and len(prompts) == 1
