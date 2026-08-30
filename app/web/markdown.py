"""Server-side markdown rendering.

html=False is load-bearing: findings contain text lifted from fetched pages,
so raw HTML must never pass through (stored-XSS guard). Citation markers [n]
become same-page links to the bibliography anchors #src-n.
"""
from __future__ import annotations

import re

from markdown_it import MarkdownIt
from markupsafe import Markup, escape

from app.db import FTS_MARK_CLOSE, FTS_MARK_OPEN

_md = MarkdownIt("gfm-like", options_update={"html": False, "linkify": True})
# Only link real URLs, never bare domain-ish words: research text is full of
# them ("180 kgf.cm", "v1.2.3", file names), and fuzzy linkify turned every
# one into a hyperlink to a website that has nothing to do with the source.
_md.linkify.set({"fuzzy_link": False, "fuzzy_email": False})

# [3] → [\[3\]](#src-3), skipping [3](...) markdown links and [x][y] refs
# (?!\() keeps real markdown links [3](url) intact. No (?!\[) — that used to
# skip every citation in an adjacent run like [1][8][9] except the last one.
_CITE_RE = re.compile(r"\[(\d{1,3})\](?!\()")


# A document's own H1 at the very top, which the run page's header already shows.
_LEADING_H1_RE = re.compile(r"\A\s*#\s+[^\n]*\n+")


def strip_leading_h1(md_text: str) -> str:
    """Drop the document's own H1 for in-app display only.

    overview.md and matrix.md keep their heading on disk — an export is a
    standalone document and needs one — but inside the run page the header
    above the tabs already carries the title, so rendering it again showed
    every run's name twice.
    """
    return _LEADING_H1_RE.sub("", md_text or "", count=1)


# A claim check's verdict cell is exactly "✓ supported (9/10)" — written by
# render_report, never by a fetched page. Markdown is rendered with html=False
# so the generator cannot emit its own markup; this puts the colour back on
# afterwards. The pattern is anchored to the four known words and a one- or
# two-digit score, and the replacement is fixed markup built from those
# groups, so nothing from the document can reach the output as HTML.
_VERDICT_CELL_RE = re.compile(
    r"<td>([✓±✗?]) (supported|contested|unsupported|unverifiable) "
    r"\((\d{1,2})/10\)</td>")


def _colour_verdicts(html: str) -> str:
    def one(m: re.Match) -> str:
        mark, word, conf = m.groups()
        return (f'<td><span class="verdict verdict-{word}">{mark} {word}</span>'
                f' <span class="verdict-conf">({conf}/10)</span></td>')
    return _VERDICT_CELL_RE.sub(one, html)


def render(md_text: str) -> str:
    return _colour_verdicts(_md.render(md_text or ""))


def highlight_snippet(snippet: str) -> Markup:
    """Make an FTS snippet safe to render.

    The snippet is page text sqlite copied verbatim, so it is escaped first;
    only then are the control-char sentinels replaced with real <mark> tags.
    """
    safe = str(escape(snippet or ""))
    return Markup(safe.replace(FTS_MARK_OPEN, "<mark>")
                      .replace(FTS_MARK_CLOSE, "</mark>"))


def render_overview(md_text: str, n_sources: int) -> str:
    def linkify_cite(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= n_sources:
            return f"[\\[{n}\\]](#src-{n})"
        return m.group(0)

    return render(_CITE_RE.sub(linkify_cite, md_text or ""))
