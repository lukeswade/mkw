# Session record: 2026-09-11 evening → 2026-09-12 early

Twenty-three commits after the synthesis-coverage handoff (`09ba82d`). Each
theme below names its commits, what was measured, and what a future reader
needs to know that the code cannot tell them. The two measured reviews have
their own documents:

- `synthesis-coverage-2026-09-11.md` — the afternoon's work plus round two
  (the candidate axis and the reconciliation pass; 33% → 83% cited).
- `scoring-triage-review-2026-09-11.md` — relevance scoring, triage, gap
  analysis and the per-round caps, with everything that shipped from them.

## 1. Synthesis: the second axis and the pass (`4304ee9`…`c7279d8`, doc)

Candidate extraction + `SYNTH_CANDIDATES_BLOCK`; a code-bounded reconciliation
pass; the honest citation metric (`cited_ids()` stops at the unused heading).
Matt's 66-source run: 22 → 55 cited in the body, words per cite flat at ~58.

## 2. The served context window (`916801e`, `00bc0cf`)

The model card says 262,144; the oMLX profile served 65,536 and **refused**
a longer prompt with a 400. `LLM_CONTEXT_TOKENS` setting (Settings page),
`single_call_budget()`, `PromptTooLong` learned from the refusal, digest
fallback instead of a failed run. Luke then set the profile and server
default to 262,144 and Hot Cache Size to `8GB`; verified with a 79,766-token
needle prompt and a prefix-cache probe (10.5s → 1.9s on the repeat).

## 3. Phone layout, three passes (`5a8fb6d`, `3b9c7f0`, `9495aa8`, `c87ec35`, `7781a15`, `6f14404`, `734d58e`)

- Pico styles every `<nav>` as `display:flex`; `.nav-inner` could not shrink
  and the whole page scrolled sideways at 393px. `.topnav{display:block}`.
- Synthesis tables had no scroll container; every rendered `<table>` is now
  wrapped in `.table-scroll` by `markdown.py`.
- Brand text removed (icon only), tabs spread across the width at ~12px, all
  seven visible at 393. "Check claims" → "Claims". Briefs left the tab bar;
  it is reached from a card on Settings and filtered in the Library.
- List rows under 560px are one grid: title + status badge on row one; kind
  badges left and the user pill + evergreen star as one right-aligned group
  on row two; the detail line full width on row three. The row is a `div`
  with the **title as the link** (a click elsewhere navigates via app.js) —
  a row that was one big `<a>` could not hold the star button.
- The evergreen star shows only on runs that are evergreen; it is the off
  switch. Turning on is the run page's job.
- No per-row ×. Swipe left on touch reveals Delete; **Select to delete** on
  both lists turns rows into checkboxes for a batch. One endpoint
  (`POST /runs/batch-delete`) hands the New tab its list back and reloads
  the Library with its filter — the old × had redirected the New tab to the
  Library on every delete. The New tab's poll pauses while a list is busy.
- Run page: a question over 160 characters folds to one line; the action
  row is Export PDF · Export HTML · Re-synthesize · **More** (Keep
  evergreen, Export Markdown, Build comparison, Delete last). Wide screens
  show the same order inline.

Measured with `document.documentElement.scrollWidth` vs `clientWidth` at 375,
393 and 1280 — never by eye. The Browser pane tab reports `document.hidden`,
so scroll events and rAF do not fire there; verify that logic inline.

## 4. Reading and reusing a long document (`9b59b55`, `d0d044d`, `f4d4ff4`)

Every H2 gets a slug id; a sticky **On this page** rail beside the prose from
1100px (the run page widens to 1120 so the prose keeps its 880 measure), a
collapsed list above it on phones. `export.md` (overview + bibliography),
`export.zip` (every servable file), Copy as Markdown. The interactive export
is removed at Luke's request.

## 5. Installable, compressed, cached (`1f5f97d`, `e49f036`, `f2ea22a`)

Manifest (`display: standalone`, SVG icon), theme-color, Apple meta per
Luke's Setup A; **no service worker on purpose** (live progress streams).
GZip for text responses; static URLs carrying their content hash are cached
a year. **Downloads are excluded from gzip** — a gzip-encoded `attachment`
with no Content-Length is what Safari saves as a damaged PDF, and Export PDF
broke on a MacBook for exactly that until `f2ea22a`.

**The home-screen icon needs a Cloudflare Access bypass.** iOS fetches
`apple-touch-icon` without the Access session cookie, got the login page,
and drew a letter tile. Fixed on Luke's side (2026-09-12): a second Access
application, destination `mattkwade.com` path `static/*`, one policy
**Bypass → Everyone**. Policies attach to an application, not a destination,
so this cannot live inside the main app without opening the whole site.
Verify with `curl -sI https://mattkwade.com/static/favicon.png` → 200 png.

## 6. Logs (`c05e3b1`)

One structured `triage` event per round (counts, top dropped domains,
dropped URLs kept in the event) instead of one `source_skipped` per dropped
page; both renderers hide the legacy per-page lines. Matt's run: 353 events →
213 rendered lines, 0 per-drop lines. The comparison page reads the new
event and falls back to the old ones.

## 7. Scoring and triage, measured then changed (`6126a20`, doc)

Economy rule removed from NOTES (+38% recovery at the keep boundary, no junk
inflation); bounded roundup spare in triage (4/round); `standard` demoted by
URL shape; `notes_recheck` on. Method that worked: re-fetch real dropped or
rejected pages and re-score them with the deployed prompt via
`take_notes(template=…)`.

## 8. Gap analysis and caps, measured then changed (`da30202`, doc part two)

Near-duplicate queries filtered (Jaccard ≥ 0.7 on `facets.content_words`;
they yielded 1.03 kept/query vs 1.48); short rounds refilled from the
thinnest facets (`Pipeline._fill_from_thin_facets`); `_ROUND_SCALE = 1.6`
(depth 10 → 8 rounds; depth, not saturation, ended 60 of 91 runs). Offline
gap A/B is possible from `rounds/round-NN.md`.

## Facts about this install worth keeping

- oMLX 0.6.4: `claude` profile on Qwen3.6-35B-A3B-4bit-DWQ, context 262,144,
  thinking off, temperature 0.2, repetition 1.05; Hot Cache Size `8GB`.
  KV is ~20 KB/token for this hybrid model; the full window is ~5 GB.
- `/v1/models` also exposes `Qwen3.6-35B-A3B-4bit-DWQ:thinking-off` as an
  addressable model id.
- Embedding model id is the short alias `nomicai-modernbert-embed-base-bf16`
  (both forms resolve).
- Library reference row for the synthesis work: `[A/B final]`
  (`20260911_211953_…`). The intermediate A/B rows were deleted.

## Still open

- Query length: 29% of gap queries exceed 8 words and yield 1.29 vs 1.67
  kept per query; a cap would need a refill.
- Under 62%→83% cited, the remaining strong-uncited sources are roundups
  the note-taker rated 6–9; relevance as a proxy for "belongs in this
  document" is the next question.
- Facet accounting exists for a handful of runs; re-measure targeting later.
- The estimate on the New page derives time from past rounds and will
  adjust to the new round count on its own.
- Desktop: the user pill now sits at the far right of the badge line rather
  than after the date — flagged to Luke, not yet confirmed as wanted.
