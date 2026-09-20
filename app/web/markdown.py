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


def strip_repeated_h1(md_text: str, title: str) -> str:
    """For exports: drop the overview's H1 only when it repeats the title the
    export already printed above it. A heading that says something else is
    kept — the standalone document still needs one."""
    m = _LEADING_H1_RE.match(md_text or "")
    if not m:
        return md_text or ""
    heading = " ".join(m.group(0).lstrip("#").split()).lower()
    if heading == " ".join((title or "").split()).lower():
        return _LEADING_H1_RE.sub("", md_text, count=1)
    return md_text


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


# Synthesis writes candidate x criterion tables into the overview, and a
# table of five columns is wider than a phone. Every rendered table gets a
# scroll container so the TABLE scrolls, never the page (393px iPhone,
# 2026-09-11: an unwrapped 555px table pushed the whole page to 568).
_TABLE_OPEN_RE = re.compile(r"<table\b")
_TABLE_CLOSE_RE = re.compile(r"</table>")


def _scroll_tables(html: str) -> str:
    html = _TABLE_OPEN_RE.sub('<div class="table-scroll"><table', html)
    return _TABLE_CLOSE_RE.sub("</table></div>", html)


_H2_RE = re.compile(r"<h2>(.*?)</h2>", re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")[:60] or "section"


def anchor_sections(html: str) -> tuple[str, list[tuple[str, str]]]:
    """Give every H2 a stable id and return the outline as (id, text).

    Ids are slugs of the heading text so a link survives a re-render; a
    repeated heading gets a numeric suffix. Only H2s: the document's
    sections are what a reader navigates, not every sub-point."""
    seen: dict[str, int] = {}
    outline: list[tuple[str, str]] = []

    def one(m: re.Match) -> str:
        inner = m.group(1)
        text = _TAG_RE.sub("", inner).strip()
        base = _slug(text)
        n = seen.get(base, 0)
        seen[base] = n + 1
        hid = base if n == 0 else f"{base}-{n + 1}"
        outline.append((hid, text))
        return f'<h2 id="{hid}">{inner}</h2>'

    return _H2_RE.sub(one, html), outline


def render(md_text: str) -> str:
    return _scroll_tables(_colour_verdicts(_md.render(md_text or "")))


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
