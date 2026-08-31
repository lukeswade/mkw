"""Solve the hashcash proof-of-work some forums put in front of their pages.

A growing number of forum installs answer HTTP 200 with a ~2.6KB page that
carries no content, only a challenge. Eight fly-fishing forums in one run were
behind this one, and they held the best material the run found — the fetch
"succeeded", extraction found nothing, and the page was discarded.

The escalation ladder cannot reach it: nothing 4xx'd, so no rung fires, and
FlareSolverr was tested and returns the same challenge page rather than
waiting for the JavaScript to finish.

The challenge itself is small and fully specified in the page it arrives on:

    window.POW_CHALLENGE_DATA = {
        challenge_nonce: '…', challenge_hmac: '…',
        difficulty: '3', difficulty_char: 'b', issued_at: '…',
        cookie_duration: '3600', cookie_domain: '…', headless_check: '1'
    }

    i = 0
    while (i++ < 1e7) {
        c = sha256(nonce + issued_at + String(i))
        if (c.startsWith(difficulty_char.repeat(difficulty))) {
            cookie pow_bypass = nonce|issued_at|i|c|hmac[|signals]
            location.reload()
        }
    }

At difficulty 3 over hex that is 16**3 ≈ 4,096 hashes on average — measured at
1,275 iterations and under 10ms. The optional trailing field is the browser's
own bot self-report (`navigator.webdriver`, WebGL, touch); an ordinary browser
with none of those reports an empty string and appends nothing, which is what
this sends.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

MARKER = "POW_CHALLENGE_DATA"
# A ceiling, not a target. Difficulty 3 lands in ~4k hashes and 4 in ~65k; a
# site that raises it far beyond that has priced us out, and burning a core on
# it is worse than skipping the page. Bounded so a harder challenge ends
# quickly instead of hanging a fetch.
MAX_ITERATIONS = 2_000_000
_FIELD_RE = re.compile(r"(\w+)\s*:\s*'([^']*)'")


@dataclass(frozen=True)
class Challenge:
    nonce: str
    hmac: str
    issued_at: str
    difficulty: int
    difficulty_char: str
    cookie_domain: str
    cookie_duration: int


def looks_challenged(body: bytes) -> bool:
    """Cheap check before decoding: is this a challenge rather than a page?"""
    return MARKER.encode() in body[:8000]


def parse(html: str) -> Challenge | None:
    """Pull the challenge out of the page that carries it, or None."""
    if MARKER not in html:
        return None
    try:
        blob = html.split(MARKER, 1)[1].split("=", 1)[1].split("}", 1)[0]
        f = dict(_FIELD_RE.findall(blob))
        difficulty = int(f["difficulty"])
        char = f["difficulty_char"]
    except (IndexError, KeyError, ValueError):
        log.debug("POW challenge present but unparseable")
        return None
    if not char or difficulty < 1 or len(char) != 1:
        return None
    return Challenge(
        nonce=f.get("challenge_nonce", ""),
        hmac=f.get("challenge_hmac", ""),
        issued_at=f.get("issued_at", ""),
        difficulty=difficulty,
        difficulty_char=char,
        cookie_domain=f.get("cookie_domain", ""),
        cookie_duration=int(f.get("cookie_duration") or 3600),
    )


def solve(challenge: Challenge) -> str | None:
    """The `pow_bypass` cookie value, or None if the work is priced too high."""
    if not challenge.nonce or not challenge.issued_at:
        return None
    target = challenge.difficulty_char * challenge.difficulty
    prefix = (challenge.nonce + challenge.issued_at).encode()
    for i in range(1, MAX_ITERATIONS):
        digest = hashlib.sha256(prefix + str(i).encode()).hexdigest()
        if digest.startswith(target):
            return (f"{challenge.nonce}|{challenge.issued_at}|{i}"
                    f"|{digest}|{challenge.hmac}")
    log.info("POW challenge unsolved within %d iterations (difficulty %d%s)",
             MAX_ITERATIONS, challenge.difficulty, challenge.difficulty_char)
    return None
