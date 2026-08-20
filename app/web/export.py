"""Export a run as a single self-contained PDF.

Uses pymupdf's Story layout engine — already a dependency for reading PDFs,
so the export costs no new packages and works offline. The document carries
the full research record: the question as asked, the cited overview, the
bibliography, and every source's notes.
"""
from __future__ import annotations

import io
import logging

from markupsafe import escape

log = logging.getLogger(__name__)

_CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.5; color: #111; }
h1 { font-size: 19pt; margin: 0 0 4pt 0; }
h2 { font-size: 13pt; margin: 16pt 0 4pt 0; }
h3 { font-size: 11pt; margin: 12pt 0 3pt 0; }
h4 { font-size: 10pt; margin: 10pt 0 3pt 0; }
p, li { margin: 3pt 0; }
ul, ol { margin: 4pt 0 4pt 14pt; }
a { color: #0a6b52; }
blockquote { color: #444; margin: 4pt 0 4pt 12pt; }
code { font-family: monospace; font-size: 9pt; }
.meta { color: #555; font-size: 9pt; }
.src { color: #555; font-size: 8.5pt; }
hr { margin: 12pt 0; }
"""


class PdfExportError(RuntimeError):
    pass


def build_run_html(*, title: str, query: str, meta_line: str,
                   overview_html: str, findings: list, cards: list) -> str:
    parts = [
        f"<h1>{escape(title)}</h1>",
        f'<p class="meta">Research question: &ldquo;{escape(query)}&rdquo;'
        f"<br>{escape(meta_line)}</p>",
        "<hr>",
        overview_html or "<p><i>No overview was produced.</i></p>",
    ]
    if findings:
        parts.append("<h2>Sources</h2><ol>")
        for f in findings:
            date = f["published_date"] or "undated"
            parts.append(
                f'<li>{escape(f["title"] or f["url"])}'
                f'<br><span class="src">{escape(f["url"])} &middot; '
                f'{escape(f["domain"] or "")} &middot; {escape(date)} &middot; '
                f'relevance {int(f["relevance"] or 0)}/10</span></li>')
        parts.append("</ol>")
    if cards:
        parts.append("<h2>Appendix &mdash; source notes</h2>")
        for card in cards:
            f = card["row"]
            parts.append(f'<h3>[{f["idx"]}] {escape(f["title"] or f["url"])}</h3>')
            parts.append(f'<p class="src">{escape(f["url"])}</p>')
            parts.append(card["html"])
    return "".join(parts)


def render_pdf(html: str) -> bytes:
    import pymupdf

    try:
        story = pymupdf.Story(html=html, user_css=_CSS)
        buf = io.BytesIO()
        writer = pymupdf.DocumentWriter(buf)
        mediabox = pymupdf.paper_rect("a4")
        where = mediabox + (42, 42, -42, -56)
        while True:
            dev = writer.begin_page(mediabox)
            more, _ = story.place(where)
            story.draw(dev)
            writer.end_page()
            if not more:
                break
        writer.close()
        # Story writes duplicate font objects on every page — dedupe and
        # deflate cuts a ~4.6MB document to a fraction of that.
        doc = pymupdf.open(stream=buf.getvalue(), filetype="pdf")
        out = doc.tobytes(garbage=4, deflate=True)
        doc.close()
        return out
    except Exception as e:  # Story chokes on markup it doesn't know
        log.exception("PDF layout failed")
        raise PdfExportError(str(e)) from e


_PAGE_CSS = """
:root { color-scheme: light dark;
  --bg: #ffffff; --text: #1b1f23; --muted: #57606a; --border: #d8dee4;
  --accent: #0a6b52; --card: #f6f8fa; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #101418; --text: #e6edf3; --muted: #9da7b1;
          --border: #30363d; --accent: #56d4b1; --card: #161b22; }
}
* { box-sizing: border-box; }
body { margin: 0 auto; padding: 2.5rem 1.25rem 4rem; max-width: 46rem;
  background: var(--bg); color: var(--text);
  font: 16px/1.6 -apple-system, "Segoe UI", Roboto, sans-serif; }
h1 { font-size: 1.7rem; line-height: 1.25; margin: 0 0 .4rem; }
h2 { font-size: 1.25rem; margin: 2rem 0 .5rem; }
h3 { font-size: 1.05rem; margin: 1.4rem 0 .4rem; }
a { color: var(--accent); }
.meta { color: var(--muted); font-size: .85rem; }
.src { color: var(--muted); font-size: .8rem; word-break: break-all; }
blockquote { color: var(--muted); border-left: 3px solid var(--border);
  margin: .6rem 0; padding: 0 0 0 .9rem; }
code { font-family: ui-monospace, monospace; font-size: .85em;
  background: var(--card); padding: .1em .3em; border-radius: 4px; }
pre { overflow-x: auto; background: var(--card); padding: .8rem;
  border-radius: 8px; }
table { border-collapse: collapse; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--border); padding: .3rem .6rem; }
hr { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
details { border: 1px solid var(--border); border-radius: 8px;
  padding: .5rem .9rem; margin: .6rem 0; background: var(--card); }
details summary { cursor: pointer; font-weight: 600; }
details summary .src { font-weight: 400; }
footer { margin-top: 3rem; color: var(--muted); font-size: .8rem; }
img { max-width: 100%; }
"""


def standalone_html(*, title: str, query: str, meta_line: str,
                    overview_html: str, findings: list, cards: list) -> str:
    """The whole run as one self-contained web page: cited overview,
    bibliography, and every source's notes in collapsible sections. No
    external assets — it works from a file:// double-click, an email
    attachment, or any static host."""
    parts = [
        f"<h1>{escape(title)}</h1>",
        f'<p class="meta">Research question: &ldquo;{escape(query)}&rdquo;'
        f"<br>{escape(meta_line)}</p>",
        "<hr>",
        overview_html or "<p><i>No overview was produced.</i></p>",
    ]
    if findings:
        parts.append("<h2>Sources</h2><ol>")
        for f in findings:
            date = f["published_date"] or "undated"
            url = str(f["url"])
            parts.append(
                f'<li id="src-{int(f["idx"])}">'
                f'<a href="{escape(url)}">{escape(f["title"] or url)}</a>'
                f'<br><span class="src">{escape(f["domain"] or "")} &middot; '
                f'{escape(date)} &middot; '
                f'relevance {int(f["relevance"] or 0)}/10</span></li>')
        parts.append("</ol>")
    if cards:
        parts.append("<h2>Source notes</h2>")
        for card in cards:
            f = card["row"]
            parts.append(
                f'<details><summary>[{int(f["idx"])}] '
                f'{escape(f["title"] or f["url"])} '
                f'<span class="src">{escape(f["domain"] or "")}</span>'
                f"</summary>{card['html']}</details>")
    parts.append("<footer>Generated by Deep Research — a self-hosted "
                 "research agent.</footer>")
    body = "\n".join(parts)
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{escape(title)}</title><style>{_PAGE_CSS}</style></head>"
            f"<body>{body}</body></html>")
