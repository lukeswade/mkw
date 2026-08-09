"""Server-side markdown rendering.

html=False is load-bearing: findings contain text lifted from fetched pages,
so raw HTML must never pass through (stored-XSS guard). Citation markers [n]
become same-page links to the bibliography anchors #src-n.
"""
from __future__ import annotations

import re

from markdown_it import MarkdownIt

_md = MarkdownIt("gfm-like", options_update={"html": False, "linkify": True})

# [3] → [\[3\]](#src-3), skipping [3](...) markdown links and [x][y] refs
_CITE_RE = re.compile(r"\[(\d{1,3})\](?!\(|\[)")


def render(md_text: str) -> str:
    return _md.render(md_text or "")


def render_overview(md_text: str, n_sources: int) -> str:
    def linkify_cite(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= n_sources:
            return f"[\\[{n}\\]](#src-{n})"
        return m.group(0)

    return render(_CITE_RE.sub(linkify_cite, md_text or ""))
