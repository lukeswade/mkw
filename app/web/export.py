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
