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
            f'<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🔭</text></svg>">'
            f"<title>{escape(title)}</title><style>{_PAGE_CSS}</style></head>"
            f"<body>{body}</body></html>")


_APP_CSS = _PAGE_CSS + """
:root[data-theme="light"] { color-scheme: light;
  --bg: #ffffff; --text: #1b1f23; --muted: #57606a; --border: #d8dee4;
  --accent: #0a6b52; --card: #f6f8fa; }
:root[data-theme="dark"] { color-scheme: dark;
  --bg: #101418; --text: #e6edf3; --muted: #9da7b1; --border: #30363d;
  --accent: #56d4b1; --card: #161b22; }
body { max-width: 52rem; }
.topbar { display: flex; flex-wrap: wrap; gap: .6rem; align-items: center;
  margin: 1rem 0; }
.topbar input[type="search"] { flex: 1 1 14rem; padding: .45rem .7rem;
  border: 1px solid var(--border); border-radius: 8px;
  background: var(--card); color: var(--text); font: inherit; }
.topbar button { padding: .4rem .7rem; border: 1px solid var(--border);
  border-radius: 8px; background: var(--card); color: var(--text);
  cursor: pointer; font: inherit; font-size: .85rem; }
nav.tabs { display: flex; gap: .25rem; border-bottom: 1px solid var(--border);
  margin: 0 0 1.2rem; overflow-x: auto; }
nav.tabs button { border: none; background: none; color: var(--muted);
  padding: .5rem .9rem; cursor: pointer; font: inherit; font-size: .95rem;
  border-bottom: 2px solid transparent; white-space: nowrap; }
nav.tabs button.active { color: var(--text); border-bottom-color: var(--accent); }
main > section { display: none; }
main > section.active { display: block; }
.card[hidden] { display: none; }
.badge { display: inline-block; font-size: .75rem; color: var(--muted);
  border: 1px solid var(--border); border-radius: 10px; padding: 0 .5rem;
  margin-left: .4rem; vertical-align: middle; }
.flash { animation: flash 1.2s ease-out; }
@keyframes flash { from { background: color-mix(in srgb, var(--accent) 25%, transparent); } to { background: transparent; } }
pre.log { font-size: .78rem; line-height: 1.45; }
@media print {
  .topbar, nav.tabs { display: none; }
  main > section { display: block !important; }
  main > section::before { content: attr(data-title); font-weight: 700;
    font-size: 1.2rem; display: block; margin: 1.5rem 0 .5rem; }
  details { border: none; padding: 0; }
  details > *:not(summary) { display: block; }
}
"""

_APP_JS = """
(function () {
  var root = document.documentElement;
  try {
    var saved = localStorage.getItem("dr-theme");
    if (saved === "dark" || saved === "light") root.dataset.theme = saved;
  } catch (e) {}
  document.getElementById("theme-btn").addEventListener("click", function () {
    var dark = root.dataset.theme === "dark" ||
      (!root.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("dr-theme", root.dataset.theme); } catch (e) {}
  });

  var tabs = document.querySelectorAll("nav.tabs button");
  var sections = document.querySelectorAll("main > section");
  function activate(name) {
    tabs.forEach(function (b) { b.classList.toggle("active", b.dataset.tab === name); });
    sections.forEach(function (s) { s.classList.toggle("active", s.id === "tab-" + name); });
  }
  tabs.forEach(function (b) {
    b.addEventListener("click", function () { activate(b.dataset.tab); });
  });

  // citations jump to the bibliography entry, in its tab
  document.addEventListener("click", function (ev) {
    var a = ev.target.closest ? ev.target.closest('a[href^="#src-"]') : null;
    if (!a) return;
    ev.preventDefault();
    activate("sources");
    var el = document.querySelector(a.getAttribute("href"));
    if (el) {
      el.scrollIntoView({ block: "center" });
      el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
    }
  });

  // search filters the sources list and the note cards
  var searchables = [].slice.call(document.querySelectorAll("[data-search]"));
  searchables.forEach(function (el) {
    el.dataset.text = (el.textContent || "").toLowerCase();
  });
  var counts = { sources: 0, notes: 0 };
  searchables.forEach(function (el) { counts[el.dataset.search] += 1; });
  var totals = { sources: counts.sources, notes: counts.notes };
  function label(tab, shown) {
    var b = document.querySelector('nav.tabs button[data-tab="' + tab + '"]');
    if (!b) return;
    b.textContent = b.dataset.label +
      (shown === null ? " (" + totals[tab] + ")"
                      : " (" + shown + "/" + totals[tab] + ")");
  }
  label("sources", null); label("notes", null);
  document.getElementById("search-box").addEventListener("input", function () {
    var q = this.value.trim().toLowerCase();
    var shown = { sources: 0, notes: 0 };
    searchables.forEach(function (el) {
      var hit = q.length < 2 || el.dataset.text.indexOf(q) !== -1;
      el.hidden = !hit;
      if (hit) shown[el.dataset.search] += 1;
      if (hit && q.length >= 2 && el.tagName === "DETAILS") el.open = true;
    });
    label("sources", q.length < 2 ? null : shown.sources);
    label("notes", q.length < 2 ? null : shown.notes);
    if (q.length >= 2) activate("notes");
  });

  var openAll = function (open) {
    document.querySelectorAll("#tab-notes details").forEach(function (d) { d.open = open; });
  };
  document.getElementById("expand-btn").addEventListener("click", function () { openAll(true); });
  document.getElementById("collapse-btn").addEventListener("click", function () { openAll(false); });
})();
"""


def interactive_html(*, title: str, query: str, meta_line: str,
                     overview_html: str, findings: list, cards: list,
                     log_lines: list[str]) -> str:
    """The run as a portable mini-app in one file: tabs, client-side search
    over sources and notes, live citation jumps, theme toggle, print layout.
    Still zero external assets."""
    import re as _re
    from collections import Counter as _Counter
    cite_counts = _Counter(int(m) for m in
                           _re.findall(r'#src-(\d+)"', overview_html or ""))

    src_items = []
    for f in findings:
        date = f["published_date"] or "undated"
        url = str(f["url"])
        idx = int(f["idx"])
        cited = (f'<span class="badge">cited ×{cite_counts[idx]}</span>'
                 if cite_counts.get(idx) else "")
        src_items.append(
            f'<li id="src-{idx}" data-search="sources">'
            f'<a href="{escape(url)}">{escape(f["title"] or url)}</a>{cited}'
            f'<br><span class="src">{escape(f["domain"] or "")} &middot; '
            f'{escape(date)} &middot; '
            f'relevance {int(f["relevance"] or 0)}/10</span></li>')

    note_cards = []
    for card in cards:
        f = card["row"]
        idx = int(f["idx"])
        note_cards.append(
            f'<details class="card" data-search="notes"><summary>[{idx}] '
            f'{escape(f["title"] or f["url"])} '
            f'<span class="badge">{int(f["relevance"] or 0)}/10</span> '
            f'<span class="src">{escape(f["domain"] or "")}</span>'
            f"</summary>{card['html']}</details>")

    tabs = ['<button data-tab="overview" data-label="Overview" class="active">Overview</button>',
            '<button data-tab="sources" data-label="Sources">Sources</button>',
            '<button data-tab="notes" data-label="Notes">Notes</button>']
    log_section = ""
    if log_lines:
        tabs.append('<button data-tab="log" data-label="Log">Log</button>')
        log_section = (f'<section id="tab-log" data-title="Log">'
                       f'<pre class="log">{escape(chr(10).join(log_lines))}'
                       f"</pre></section>")

    body = (
        f"<h1>{escape(title)}</h1>"
        f'<p class="meta">Research question: &ldquo;{escape(query)}&rdquo;'
        f"<br>{escape(meta_line)}</p>"
        f'<div class="topbar">'
        f'<input id="search-box" type="search" '
        f'placeholder="Search sources and notes&hellip;">'
        f'<button id="expand-btn" type="button">Expand all</button>'
        f'<button id="collapse-btn" type="button">Collapse all</button>'
        f'<button id="theme-btn" type="button" title="Toggle theme">☀/☾</button>'
        f"</div>"
        f'<nav class="tabs">{"".join(tabs)}</nav>'
        f"<main>"
        f'<section id="tab-overview" class="active" data-title="Overview">'
        f'{overview_html or "<p><i>No overview was produced.</i></p>"}</section>'
        f'<section id="tab-sources" data-title="Sources"><ol>'
        f'{"".join(src_items) or "<p><i>No sources.</i></p>"}</ol></section>'
        f'<section id="tab-notes" data-title="Source notes">'
        f'{"".join(note_cards) or "<p><i>No notes.</i></p>"}</section>'
        f"{log_section}"
        f"</main>"
        f"<footer>Generated by Deep Research — a self-hosted research "
        f"agent.</footer>")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🔭</text></svg>">'
            f"<title>{escape(title)}</title><style>{_APP_CSS}</style></head>"
            f"<body>{body}<script>{_APP_JS}</script></body></html>")
