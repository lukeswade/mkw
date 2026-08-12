# Deep Research — code review & remediation plan

Review of commits `d1d0d7b..fbaef1e` (11 commits made via Antigravity) plus the
current working tree, on 2026-08-12. Everything below was verified against the
running system, the real `data/` directory, and the actual code — findings are
marked with the evidence that establishes them.

**Verdict:** the feature work is genuinely good — verbatim evidence quotes,
depth-0 quick chat, run deletion, a much better-looking UI. But none of it was
ever run or tested. The research engine is **broken right now**, a fresh install
**cannot boot**, the delete button **destroys data and then errors**, and the
test suite has been **red since the first of these commits**.

---

## P0 — Broken right now (must fix before anything else)

### 1. Every research run fails immediately (`app/llm/prompts.py:69-79`)

Commit `e5a969a` appended a literal JSON example to the `NOTES` template with
unescaped braces. `NOTES` is consumed by `str.format()`, so `{` opens a
replacement field:

```python
# notes.py:128
prompt = prompts.NOTES.format(brief=..., text=...)
# -> KeyError: '\n  "relevance"'
```

Verified inside the **running container**: `NOTES.format()` raises `KeyError`.
`take_notes` only catches `LLMJsonError`, so this escapes through
`asyncio.gather` → `_round` → `_run` and the run is marked `failed` on the very
first document. No findings, no synthesis. Applies to every run with depth ≥ 1.

**Why nobody noticed:** the last research run was 2026-08-11 11:48 CDT; the
breaking commit landed at 12:40 CDT. Zero runs have been attempted since.

*Fix:* escape the example braces (`{{` / `}}`), or move the example out of the
`.format()` template. Add a test that formats every prompt template.

### 2. A fresh database cannot start the server (`app/db.py:23-27` + `app/schema.sql:15`)

`schema.sql` gained the `evergreen` column *and* a redundant `ALTER TABLE`
migration was appended. On a new database `migrate()` runs both:

```
applying migration 0 (creates runs WITH evergreen)
applying migration 1 (ALTER TABLE runs ADD COLUMN evergreen)
  -> sqlite3.OperationalError: duplicate column name: evergreen
```

Your live DB is masked because it is already at `user_version=2`. **Any fresh
clone, new Docker volume, CI run, or the Ubuntu deploy will fail to boot.** This
is also the direct cause of 14 of the 16 test failures.

*Fix (validated against all three DB states — fresh, legacy v1, current v2):*
make migration 1 idempotent with a `PRAGMA table_info` guard before the `ALTER`.

### 3. Deleting a run destroys it, then returns HTTP 500 (`app/web/routes_runs.py:247`, also `:194`)

`Response` is never imported (only `FileResponse`, `RedirectResponse`,
`StreamingResponse`). The `NameError` fires on the **success** path — *after*
`repo.delete_run()` and `shutil.rmtree()`:

```python
241    request.app.state.repo.delete_run(run_id)   # DB row gone
245    shutil.rmtree(run_dir)                      # markdown gone
247    return Response(status_code=200, ...)       # NameError -> 500
```

htmx does not swap on 5xx, so the row stays on screen and `HX-Redirect` never
fires. To the user it looks like the delete failed — but the research is
permanently gone.

*Fix:* import `Response`. Then see item 8 for what else deletion must do.

### 4. Test suite is red — 16 failures

`docker compose ... run --rm app pytest` → **16 failed**. Fourteen are the
migration crash above. The other two are stale fixtures the commits never
updated:

- `tests/test_pipeline_e2e.py:71` feeds `key_facts` as plain strings, but
  `NotesOut.key_facts` is now `list[Fact]` → `ValidationError` (not
  `LLMJsonError`, so it isn't caught) → both e2e tests fail. That is the entire
  end-to-end coverage of the pipeline.
- `tests/test_web.py:200` tests `/api/graph`, a route deleted in `6d90c88`.

---

## P1 — Silent quality regressions (the engine works but reasons worse)

### 5. Python dict reprs are injected into the gap-analysis prompt (`app/research/gap.py:22`)

`pipeline.py:304` now stores facts as dicts (`[f.model_dump() for f in ...]`),
but `gap.py` was never updated and still formats each element as a string. The
LLM receives:

```
  - {'claim': 'X is true', 'evidence_quote': 'quoted bit', 'confidence': 9}
```

Gap analysis rewrites `state_md`, which is the pipeline's only memory and feeds
synthesis. This degrades saturation judgement, next-round query targeting, and
the final overview — on every run.

### 6. Literal `\n` in every finding (`app/research/notes.py:152,155`)

`"\\n"` in a non-raw string is backslash-plus-n, not a newline. Confirmed in the
served HTML: **134 literal `\n` on a single run page.** The whole "Key facts"
section collapses to one run-on line and the blockquote evidence never renders —
in the UI, in the `.md` on disk, and in downloads.

### 7. The evidence quotes never reach synthesis

`synthesizer.py::_note_block` uses only `citation_line`, `url`, and `notes_md`.
`key_facts` — and therefore every verbatim `evidence_quote` — is never passed to
the synthesis stage. The headline feature of `e43e356` does not influence the
output document it was built to improve.

### 8. Deletion leaves orphans and can race a live run

| Store | Cleaned? |
|---|---|
| DB row, `findings`, `run_entities`, `run_links` | yes (CASCADE, FKs are on) |
| FTS index | yes |
| Run directory | yes |
| **Chroma vectors** | **no** |

`rag/index.py:50` has `delete_run()`; the endpoint never calls it. Deleted runs
keep surfacing in Ask and semantic search, rendering `<a href="/runs/{deleted}">`
404 links, and their text is still fed to the LLM as context. Deleting for
privacy or bad-data reasons does not actually remove the content.

Also: no status guard (deleting a *running* run rmtrees the directory out from
under the pipeline, which keeps writing), no `orch.cancel()`, and no
`is_relative_to()` containment check before `shutil.rmtree` — the sibling
`run_file` endpoint 20 lines above does that check correctly.

### 9. Stored XSS in library keyword search (`app/web/templates/library.html:24`)

```jinja
{{ hit.snippet | safe if mode == 'keyword' else hit.snippet }}
```

`hit.snippet` comes from SQLite's `snippet()`, which does **not** escape HTML,
over a body populated from fetched web pages. Proven: `<img src=x
onerror=alert(document.cookie)>` passes through unescaped. This bypasses the
exact guard `markdown.py` exists to provide.

Pre-existing (not from these commits), but the new verbatim-excerpt extraction
copies raw source text into findings, which materially raises the odds of live
HTML reaching the index. Fix by escaping the body then re-injecting `<mark>`.

---

## P2 — Local-model reliability (your MLX setup specifically)

Measured across all 11 real runs in `data/research_data`:

```
sources KEPT:    60
sources SKIPPED: 112
    55  relevance too low
    15  fetch failed
    13  robots.txt
    13  outside recency window
     9  unusable notes output (JSON failure)   <-- 13.0% of usable sources lost
     7  no extractable text
```

**9 good sources were thrown away purely because the local model emitted
malformed JSON.** There is a proper fix, and I verified your server supports it:

- `POST /v1/chat/completions` with
  `response_format: {"type":"json_schema","json_schema":{...,"strict":true}}`
  returned **HTTP 200 and schema-perfect JSON** from
  `mlx-community/Qwen3.6-35B-A3B-4bit-DWQ`. Grammar-constrained decoding makes
  malformed JSON structurally impossible. Today the code only ever sends the
  weaker `{"type":"json_object"}` (`client.py:105`).
- `client.py` **never checks `finish_reason`**, so a response truncated at
  `max_tokens` is treated as complete and then fails to parse. The repair
  round-trip re-sends the prompt *plus* the truncated output, so it truncates
  again — which is why 9 of 12 repairs failed. Detect `finish_reason == "length"`
  and retry with a larger budget instead of a doomed repair.
- `Fact` is the only non-forgiving LLM model (`models.py:67`): `confidence` has
  no default and `claim` has `min_length=1`, so one malformed fact sinks the
  entire document. `FollowUpsOut` and `EntitiesOut` both drop bad items instead.
  Make `Fact` match that pattern.
- The local API key is hardcoded to `"sk-mlx-local"` (`client.py:53`). Make it a
  real setting (`LOCAL_LLM_API_KEY`) with a Settings field.

Also worth tuning: "relevance too low" discards 55 sources — by far the biggest
bucket. Worth checking whether the ≥5 threshold is right for a local model that
may score more harshly than DeepSeek.

### 10. Your Telegram bot token is written to disk 15,287 times

`data/app.log` is 3.1 MB and mostly httpx INFO lines of the form
`POST https://api.telegram.org/bot<TOKEN>/getUpdates`. The log is gitignored (so
not in GitHub), but the README tells you to back up with `tar czf backup.tgz
data/` — which puts the token in every backup.

*Fix:* set the `httpx` logger to WARNING, truncate the existing log, and rotate
the bot token since it has been on disk in cleartext.

---

## P3 — Dead weight, hygiene, and polish

**Orphaned knowledge-graph layer.** The graph page is gone but entity extraction
still runs on every completed run (`pipeline.py:350-358`) — an extra LLM call and
up to 1500 output tokens per run, ~30–60 s on a local 35B, feeding tables
(`entities`, `run_entities`) whose only reader (`db.py::graph_rows`) has zero
callers. 92 entities extracted for nothing. Decide: delete the layer, or rebuild
the graph UI. Related dead weight: `vendor/force-graph.min.js` (177 KB, shipped,
zero references), `app.js:71-139`, `app.css:831-853`.

**Assets.** `favicon.png` is **425 KB** (512×512, no alpha) and replaced a
zero-byte inline SVG — it is 16× the size of the entire stylesheet. `base.html`
now loads **Inter from Google Fonts over CDN**, which contradicts the vendored,
self-hosted design, leaks visitor IPs to Google, and breaks on a LAN/air-gapped
deploy. Ironic pairing: the dead graph library is vendored, the live font is not.

**CSS.** The rewrite looks good but has real defects: `.test-result.ok/.fail`
were deleted while `partials/test_result.html` still emits them, so the Settings
"Test LLM"/"Test SearXNG" buttons show success and failure in **identical
colors**; three overlapping `@media (max-width:640px)` blocks where the first is
entirely dead and the other two duplicate each other; an 11-`!important` block
appended *after* the media queries so nothing responsive can override it;
`html { font-size: 14px }` discards the user's browser font-size preference; and
~40 hardcoded colors bypassing the variable system.

**Cache busting.** The manual `?v=N` scheme already failed twice in 11 commits
(`cbff392` and `edff882` both changed `app.css` without bumping; `0c39447` exists
solely to clean up after the second). Replace with a content hash computed at
startup and exposed as a Jinja global.

**The old 3D/CAD project is still in the repo — and is a live hazard.** 34 files,
23 MB, **96 % of the repo by size** (`public/models/3DBenchy.stl` 11 MB, two
identical 6 MB icons). Nothing in the Python app references it. The hazard:
`package.json` defines `"deploy": "npm run build && wrangler deploy"` and
`wrangler.jsonc` claims `mattkwade.com` as a custom domain — the same hostname
your Cloudflare tunnel now serves the research app on. A stray `npm run deploy`,
or a Cloudflare Git integration that auto-detects `wrangler.jsonc`, would
republish the old 3D app over your research app. All 23 MB is also sent to the
Docker daemon as build context on every build (`.dockerignore` misses `src`,
`public`, `index.html`).

**Evergreen research is shipped but unreachable.** No template references it, and
`POST /runs` never passes the flag, so `list_evergreen_runs()` always returns
empty and `refresh_worker` is a permanent no-op. It also sleeps 24 h *before* its
first check (so a restart inside a day means it never runs), its startup errors
vanish silently into an un-awaited task, and `server.py:23` imports it at module
scope — so any future import error there takes down the whole server rather than
degrading like the RAG and Telegram layers do. Either wire up the UI or remove it.

**Smaller items.** `uv.lock` is untracked, empty, and locks nothing (pyproject
has no `[project]` table) — delete it and gitignore `.venv/`. `cloudflared` has no
compose profile, so `scripts/dev.sh` now starts it with an empty token and it
crash-loops under `restart: unless-stopped`. `CLOUDFLARE_TUNNEL_TOKEN` is missing
from `.env.example`. `WEB_PASSWORD` is empty — Cloudflare Access *is* protecting
`mattkwade.com` (verified: the public URL redirects to
`lukewade.cloudflareaccess.com`), so this is a missing second layer rather than
an open door, but it means anything that bypasses Access has no app-level gate.

**README drift** — and it is now user-visible, because the app serves it at
`/readme`. It still advertises the removed knowledge graph, still says depth is
"1–10", still tells you to "set `WEB_PASSWORD` if it's reachable by anyone but
you" when the default compose now publishes it to the internet, and documents
none of: depth-0 quick chat, run deletion, evergreen research, the `/readme`
page, or the tunnel. The backup/recovery section also now contradicts the delete
button, which rmtrees the markdown the README calls the source of truth.

---

## The plan

### Phase 0 — Unbreak (do this first; ~30–45 min)
1. Escape the braces in the `NOTES` template; add a test that `.format()`s every template.
2. Make the `evergreen` migration idempotent (guarded `ALTER`); add a test for fresh / v1 / v2.
3. Import `Response` in `routes_runs.py`.
4. Fix `test_pipeline_e2e.py` fixtures to emit `Fact` dicts; delete `test_graph_api_shape`.
5. Rebuild, confirm green suite, then run one real depth-2 run end to end.

**Exit:** tests pass, a real research run completes on your local model.

### Phase 1 — Correctness regressions
6. `gap.py:22` — render `fact["claim"]` (+ confidence) as prose, not a dict repr.
7. `notes.py:152,155` — `"\\n"` → `"\n"`.
8. Pass evidence quotes into synthesis so the feature actually affects the overview.
9. Delete endpoint: call `rag.index.delete_run()` and `orch.cancel()`, guard
   against deleting a running run, add the `is_relative_to()` check before `rmtree`.
10. Escape the FTS snippet before `| safe`.
11. Tests for each of the above.

### Phase 2 — Local-model reliability (biggest quality win for your setup)
12. Send `response_format: json_schema` (strict) derived from the pydantic model
    when the provider supports it; keep `json_object` + repair as fallback for DeepSeek.
13. Detect `finish_reason == "length"` and retry with a larger token budget
    instead of a repair round-trip; raise the notes budget above 1200.
14. Make `Fact` forgiving (drop bad items, default confidence).
15. Move the local API key into Settings.
16. Re-measure the skip breakdown; revisit the relevance-5 threshold.

**Exit:** JSON-failure skips at zero; source yield up ~13 %.

### Phase 3 — Decide the half-finished features
17. Evergreen: either add the UI toggle + first-check-on-boot + awaited shutdown
    + guarded import, or delete `refresh_worker.py` and the flag entirely.
18. Entities/graph: either rebuild the graph page (the data is already there and
    the force-graph lib is already vendored), or delete the extraction call,
    `entities.py`, `graph_rows`, the tables, the vendored JS, and the README line.
    Right now you pay for it every run and get nothing.

### Phase 4 — Hygiene & polish
19. Remove the old 3D/CAD project in one dedicated commit (recovery:
    `git checkout 8ef6a06 -- <path>`). **Confirm first that nothing still needs
    `wrangler.jsonc`** — deleting it removes the ability to redeploy the old site.
20. Resize the favicon to ~2 KB; vendor Inter as woff2 (or drop to system fonts).
21. CSS: restore `.test-result.ok/.fail`, merge the three mobile blocks, remove
    the `!important` block via specificity, `html { font-size: 100% }`, migrate
    stray hex values to variables.
22. Replace `?v=N` with a startup content hash.
23. `cloudflared` behind a compose profile; `CLOUDFLARE_TUNNEL_TOKEN` into
    `.env.example`; delete `uv.lock`; gitignore `.venv/`; add `src`/`public`/
    `index.html` to `.dockerignore`.
24. Silence httpx INFO logging, truncate `app.log`, **rotate the Telegram token**.
25. Set a `WEB_PASSWORD` as defense in depth behind Access.
26. Full README pass against current reality.

### Phase 5 — Stop this from recurring
27. A `scripts/check.sh` (or pre-push hook / tiny GitHub Action) that runs the
    suite. Every P0 above would have been caught by simply running `pytest` once.
28. Extend `smoke_live.sh` to cover depth-0 and deletion.
29. A startup self-check that formats all prompt templates and opens the DB, so
    a broken template or migration fails loudly at boot rather than mid-run.

---

## Optional enhancements (given your local model stack)

- You have `nomicai-modernbert-embed-base` loaded in MLX. Serving embeddings from
  there instead of in-container sentence-transformers would cut ~2.2 GB from the
  image and use the GPU. Worth doing only after Phase 2 — it changes the vector
  space, so it requires a full `cli reindex`.
- `parakeet-tdt` (STT) and `Kokoro` (TTS) would make a genuinely nice Telegram
  addition: send a voice note to start research, get the TL;DR read back.
