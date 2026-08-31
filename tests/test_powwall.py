"""The hashcash wall some forums answer with instead of their pages."""
from __future__ import annotations

import hashlib

from app.research import powwall

CHALLENGE_PAGE = """<!DOCTYPE html><html><head><script>
    window.POW_CHALLENGE_DATA={
        challenge_nonce:'a31815bb7c51a15257d45e795c394d06',
        challenge_hmac:'6a6ec726a8a1f7481be65727',
        difficulty:'3',
        difficulty_char:'b',
        issued_at:'1788214438',
        cookie_duration:'3600',
        cookie_domain:'www.forum.test',
        referrer:'(null)',
        headless_check:'1'
    };
</script></head><body><noscript>JavaScript Required</noscript></body></html>"""


def test_the_challenge_is_read_out_of_the_page_that_carries_it():
    c = powwall.parse(CHALLENGE_PAGE)
    assert c is not None
    assert c.nonce == "a31815bb7c51a15257d45e795c394d06"
    assert c.hmac == "6a6ec726a8a1f7481be65727"
    assert (c.difficulty, c.difficulty_char) == (3, "b")
    assert c.cookie_domain == "www.forum.test"


def test_an_ordinary_page_carries_no_challenge():
    assert powwall.parse("<html><body>a normal page</body></html>") is None
    assert not powwall.looks_challenged(b"<html><body>a normal page</body>")
    assert powwall.looks_challenged(CHALLENGE_PAGE.encode())


def test_the_answer_is_what_the_page_asks_for():
    """nonce + issued_at + counter, hashed until the hex starts with the
    difficulty character repeated difficulty times."""
    c = powwall.parse(CHALLENGE_PAGE)
    cookie = powwall.solve(c)
    assert cookie is not None
    nonce, issued, counter, digest, mac = cookie.split("|")
    assert (nonce, issued, mac) == (c.nonce, c.issued_at, c.hmac)
    assert digest.startswith("bbb")
    # and it is genuinely that hash, not a plausible-looking string
    assert hashlib.sha256(
        (c.nonce + c.issued_at + counter).encode()).hexdigest() == digest


def test_work_priced_too_high_is_declined_rather_than_ground_out():
    """A bounded loop, so a site that raises the difficulty ends the fetch
    instead of pinning a core to it."""
    hard = powwall.Challenge(nonce="n", hmac="h", issued_at="1", difficulty=9,
                             difficulty_char="f", cookie_domain="x.test",
                             cookie_duration=3600)
    original = powwall.MAX_ITERATIONS
    try:
        powwall.MAX_ITERATIONS = 500      # keep the test fast
        assert powwall.solve(hard) is None
    finally:
        powwall.MAX_ITERATIONS = original


def test_a_malformed_challenge_is_not_a_crash():
    for bad in ("<script>window.POW_CHALLENGE_DATA={</script>",
                "<script>window.POW_CHALLENGE_DATA={difficulty:'x'}</script>",
                "<script>window.POW_CHALLENGE_DATA={difficulty:'3'}</script>"):
        assert powwall.parse(bad) is None
