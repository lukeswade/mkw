# 🔭 Deep Research

[![CI](https://github.com/lukeswade/deep-research/actions/workflows/ci.yml/badge.svg)](https://github.com/lukeswade/deep-research/actions/workflows/ci.yml)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab.svg)](https://www.python.org/)
[![Self-hosted](https://img.shields.io/badge/runs-100%25%20self--hosted-6ee7b7.svg)](#install)

A research assistant that runs on your own machine and digs much deeper than
a chat answer. Ask it a question, choose how hard it should work, and it goes
and reads the web for you — dozens of pages, not a snippet — then writes a
cited overview you can check line by line.

Everything stays on your computer. The one thing that can leave it is the
call to the language model, and even that stays home if you point it at a
local model (LM Studio, Ollama, llama.cpp).

The same machinery does three jobs:

| | you give it | you get back |
| --- | --- | --- |
| **Research** | a question | a cited overview built from dozens of sources |
| **Brief** | a list of sites you follow | one summary of what is new, daily if you like |
| **Claim check** | pasted text (an article, a chatbot's answer) | a verdict on every factual claim, with the quote that decided it |

Every run is plain markdown on disk. Delete the database and it rebuilds
itself from the files.

![The research form, with a live estimate of what the chosen depth implies](docs/screenshots/home.png)

## What it looks like

A finished run: the cited overview, the sources behind every claim, notes on
each source with verbatim evidence, suggested follow-ups, and the full log.

![A completed run showing the cited overview, related runs and result tabs](docs/screenshots/run-overview.png)

The library remembers everything you have ever researched and searches it by
meaning, not just by keyword.

![Library search results ranked by semantic similarity](docs/screenshots/library.png)

Any OpenAI-compatible model works, and the many small note-taking calls can go
to a cheaper, faster model than the one doing the thinking.

![Settings page showing provider presets and the fast-model option](docs/screenshots/settings.png)

---

## Install

You need Docker (with the compose plugin) and a language model. Nothing else.

```bash
git clone https://github.com/lukeswade/deep-research.git
cd deep-research

cp .env.example .env
nano .env                 # add a model (next section) and a SEARXNG_SECRET

mkdir -p data && sudo chown -R 1000:1000 data   # the container runs as user 1000

docker compose build      # build on the machine that will run it
docker compose up -d
```

Open <http://localhost:8090>. The first build takes a few minutes and makes a
~2.5 GB image, because a small text-embedding model is baked into it.

### Choose a model

Deep Research talks to any **OpenAI-compatible** endpoint. Pick a preset on
the Settings page and the address is filled in for you.

| | Providers |
|---|---|
| **Cloud** | DeepSeek · OpenAI · OpenRouter · Groq · Together |
| **Local** | LM Studio · Ollama · llama.cpp / vLLM / MLX (choose "Other") |

The quickest cloud start is DeepSeek — cheap, and good at the structured
output the pipeline relies on:

```
DEEPSEEK_API_KEY=sk-...
```

The quickest local start is LM Studio or Ollama:

```
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:32b-instruct
LLM_CONCURRENCY=1        # one request at a time suits a single GPU
```

Two things that matter when running locally:

- **Pick a model that supports strict `json_schema` output** (LM Studio and
  recent llama.cpp do). The pipeline uses schema-constrained decoding so
  malformed JSON is impossible. Without it everything still works, but a
  weaker model will occasionally produce something unparseable and lose that
  source. A 30B-class instruct model is a good floor.
- **Set a fast model.** A run makes one planning call and one synthesis call
  but a dozen or more note-taking calls per round, so almost all the time
  goes into notes. Name a smaller model in the *Fast model* box and it takes
  the notes, the triage and the first-look screening, leaving the big model
  to think:

  ```
  LLM_MODEL=qwen2.5:32b-instruct   # planning + synthesis
  FAST_MODEL=qwen2.5:7b-instruct   # notes, triage
  ```

Everything here can be changed later on **Settings**, which also has *Test
LLM* and *Test SearXNG* buttons. Settings are stored in `data/settings.json`
(readable only by the app) and win over `.env`.

---

## Using it

### The New page

**Depth** is a 0–10 effort dial. Each pair of steps buys roughly one full
search round, so odd numbers are real half-steps.

| depth | what you get | sources |
|---|---|---|
| **0** | an instant answer: one search, a short cited summary from the result snippets. No pages are read. | — |
| **1–4** | a question answered: one or two rounds. The form defaults to 3. | ~8–28 |
| **5–7** | a deep dive: two to four rounds | up to ~54 |
| **8–10** | deep research: an hour or more on a local model | up to ~85 |

Runs stop early when the topic is saturated, after two rounds that find
nothing new, or at the depth's source and model-call caps. When the gap
analysis has nothing left to ask but the run still has budget, it goes one
results page deeper on the queries that worked instead of quitting.

**The estimate** under the form — searches, sources, model calls, time and
cost — is calibrated on *your* finished runs. After a few completed runs at a
depth it uses that depth's own history (the middle half of it, so one freak
run does not skew it); before that it falls back to defaults. The form also
warns you if something similar is already in the library.

**Search categories** pick which groups of search engines a run uses:
*general* (the web), *science* (Crossref, OpenAlex, Semantic Scholar, arXiv —
research-grade sources that never rate-limit), *it*, *news*, *videos*
(YouTube, read through caption transcripts), *social media* (Reddit threads,
post and comments) and *files*. The default is `general,science`; your last
choice is remembered in the browser, and retries, follow-ups and daily
refreshes inherit the run's own selection.

**Recency** maps to search-engine time filters plus a date check on the
pages themselves. Date metadata on the web is imperfect, so undated sources
are kept but flagged and the synthesis is told to prefer dated material.
Treat the windows as best effort.

**Build on earlier research** (on by default) lets the planner see what your
library already knows and aim at the gaps. Turn it off to run cold — worth
doing when an earlier conclusion was wrong and you do not want it anchoring
the new one.

### The run page

While a run is going you see its log stream live. When it is finished:

- **Overview** — the cited synthesis. Every `[n]` jumps to the bibliography.
  If this run builds on earlier research, a follow-up or a daily refresh, it
  leads with *what's new since the last look* rather than repeating itself.
- **Sources** — every kept source with its notes and evidence quotes — each
  quote checked against the page; one the page does not contain is repaired
  to the page's own sentence or removed — and a ☆ *Follow* button that turns
  that site into a brief source.
- **Comparison** — appears once you press *Build comparison* on a run that
  weighs things against each other (see below).
- **Further research** — suggested follow-ups, each a one-click run that
  inherits this run's settings.
- **Log** — the full record.

Below the tabs, **How this run went** folds out the whole story in one place:
how many rounds and searches, which engines refused, how many results were
filtered out before selection, how many pages were read and kept, how many
pages fought back (and how they were beaten), how many quotes were repaired
or removed, and what the model cost. A run that kept almost nothing also gets
a banner at the top saying *why* — engines refusing (fixable — see *When
search goes quiet* below) or a topic with little to find (not). If the same
question has other completed runs, the header offers each as a one-click
**side-by-side comparison**.

**Exports** — three ways to take the record with you: **PDF**, a
self-contained **HTML** page (dark-mode aware, opens from a double-click,
shares over anything), and **Markdown** — the overview and bibliography as
one `.md` file that drops straight into Obsidian, Joplin or an agent's
context. The Sources tab's *Files* list also offers every file of the run
as one `.zip`: overview, bibliography, and a note file per source.

**Re-synthesize** rewrites the overview from the stored sources without
searching again — for when the research succeeded but the final write-up did
not (a truncation, a model that emitted its reasoning instead of the
document). Normal runs detect that and retry once on their own. The rewrite
keeps everything the document admits about itself, from the run's own
coverage record; a run that finished before that record was kept says so at
its foot instead, rather than dropping those sections and reading as more
certain than the document it replaced.

**Build comparison** re-reads the stored findings of a finished run and,
when the run really does compare two or more things, writes a table: the
things across the top, the axes the sources actually cover down the side,
every cell cited, gaps marked as gaps, and disagreements flagged with †
rather than averaged into a number nobody wrote. One model call, no
searching, works on runs from weeks ago. Runs that are not comparisons say so
and build nothing.

**☆ Evergreen** re-researches the topic once a day against a recent window.
Each refresh is a linked child run whose overview leads with what changed.

**Retry** (on a failed or interrupted run) is the same run again — same
kind, categories, reading list, document and memory setting.

### Library

Everything you have researched, searchable by keyword or by meaning. Each
run carries a badge for what it is — research, brief, claim check — plus a
marker if it has a comparison table. Deleting a run removes it everywhere,
files included; take a backup first if you might want it later.

### Learned

What this install has worked out from its own runs — nothing on this page is
a setting, it is what the data says:

- **Sources that pay off** — domains ranked by how often a read became a kept
  source. On a new question, the sources that proved themselves on *related*
  earlier questions start round one with a larger share.
- **Your authority sites** and their yield, since those bypass triage.
- **Sources that never pay off** — read many times across several runs, kept
  nothing. They already rank last; a button blocks them for good.
- **Search engines, last 14 days** — how often each refused, and how often
  each engine's results turned into kept sources, which is what decides the
  order engines are drawn from.
- **Brave requests this month**, against your plan if you enter it.
- **Estimate calibration** — what the New page's estimate knows about each
  depth from your recent runs.

### Ask

Asks a question of everything you have researched so far and answers with
citations to the runs it drew on. Every answer offers *Research this deeper*,
a one-click handoff into a full run.

### Briefs

A brief is a saved reading list plus a standing interest. Name it, add
sources, run it: instead of searching the web it reads those sites, fetches
what is new, and writes one synthesis across all of it. Toggle **☆ daily**
and it runs on its own; anything an earlier brief already reported is
skipped, so day two is not day one again. Briefs are reached from the
**Settings** page — they left the tab bar so it fits a phone — and finished
briefs appear in the Library under the *Briefs* filter.

Add sources **by site address, not feed URL** — nobody knows where a site
keeps its feed. Type `simonwillison.net` and it finds the feed itself;
`owner/repo` becomes that GitHub project's releases feed. A source only
counts if its feed parses *and* has entries, so a page that pretends to exist
never gets subscribed. Or press ☆ *Follow* on any source in a finished run.

The **interest** is optional but does real work: one pass over the entry
titles narrows the week before anything is fetched, which is what keeps a
post about a video game out of a brief about local language models. Briefs
also score items differently from research — the question is *is this a
change I should know about*, so a three-line release note beats a long post
restating what you already assume.

### Check claims

Paste anything that makes factual assertions — an article, a report, an
answer from ChatGPT, Gemini or Claude. It pulls out each claim, checks your
own library first and the web second, and returns a table: **supported**,
**contested**, **unsupported** or **unverifiable**, each with a confidence, a
reason, and the quote that decided it. Opinions and predictions are listed
separately, not judged. An *unsupported* verdict is never accepted from your
library alone — a wrong "supported" merely echoes research you already have,
but a wrong "unsupported" tells you something true is false, which is the
failure that matters when you are hunting for errors. Checking is capped at
the 15 most load-bearing claims; the rest are listed, not dropped.

### Settings

Most fields explain themselves on the page. The ones people ask about:

- **Default categories** — recommended `general,science`. Leave `it` out: it
  answers repair and how-to questions with package registries.
- **Relevance threshold** — recommended **4**. Every source is scored 0–10
  against the question and discarded below this line. 4 keeps a page with
  real material on part of the question; 6 gives fewer, tighter sources and
  thinner runs.
- **Embedding API key** — recommended **blank**. Only needed when the
  embedding endpoint is a different service from the LLM. The built-in model
  needs none.
- **Authority sites** — a short list of "domain — what it holds" for
  goldmines search engines barely index (the shipped example is charm.li, the
  mirror of full factory service manuals). When a topic fits one, the planner
  dedicates a query to it, its pages skip triage, and it starts every round
  at the six-slot ceiling other sources have to earn. Add your own as you
  find them — and watch their yield on the Learned page.
- **Planner prompt / per-query scope / gap-analysis prompt** — three switches
  that were each decided by a paired A/B run; the recommended value is marked
  and the result is in each explainer. Leave them unless you are testing.
- **Display time zone** — every clock the server prints. Runs are stored in
  UTC either way.
- **Blocked domains** — sites you never want fetched.
- **Follow references** — when a source makes the cut, its best outbound
  links become candidates too. Off by default: over a fortnight on this
  install it read 73 chased pages and kept 3, and the rejects scored 0-2,
  not near-misses. When on it is capped at 4 references a round and 6 a run.

### Telegram (optional)

1. Message **@BotFather** → `/newbot` → put the token in `.env` as
   `TELEGRAM_BOT_TOKEN`.
2. `docker compose restart app`, message your bot `/id`, put the number in
   `TELEGRAM_ALLOWED_USER_IDS`, restart again.
3. Send it a question. It walks you through depth and recency and delivers
   the overview when the run finishes. `/help` lists every command.

Only allowlisted user IDs get answers. Without a token the app runs web-only.

### Command line

Everything the web form can say, the CLI can say:

```bash
docker compose exec app python -m app.cli run "your question" --depth 3 --recency month
docker compose exec app python -m app.cli run "what changed" --categories general,news --no-prior
docker compose exec app python -m app.cli run --kind verify --document ./article.md "check this"
docker compose exec app python -m app.cli run --kind brief --brief-id 2 "weekly read"
docker compose exec app python -m app.cli runs
docker compose exec app python -m app.cli ask "a question over everything so far"
docker compose exec app python -m app.cli reindex    # rebuild indexes from the .md files
```

**A/B testing a change** — one question, two arms back to back under
different settings, scored side by side:

```bash
docker compose exec app python -m app.cli ab "your question" --depth 4 --env-a "GAP_VARIANT=default" --env-b "GAP_VARIANT=anchored"
```

The table shows kept sources, quality, waste, time and cost for each arm and
links to the same comparison in the app. Every tuning decision in this
project was made this way; one run per arm is directional, not proof.

---

## How it reads the web

Search returns far more than is worth reading, and much of the web does not
want to be read.

First the question is broken into **parts** — every separate thing you asked
for gets its own name, and the run is scored against those names from then
on. Rounds go to whichever part has the least so far, no part may take more
than a third of a round while another has none, and a part that never finds
a source is reported to you rather than quietly filled in.

The planner also picks out up to two **assumptions the question takes for
granted** that a published standard could settle. "The fields we play on are
way too small" is checkable, because governing bodies publish field
dimensions; "we are getting destroyed" is not. Each gets one search, added
after the parts have been allocated so checking costs them nothing. If an
assumption turns out to be wrong the whole answer changes, so the document
settles it before it answers anything else.

What happens between a query and a source:

1. **Search** goes through your own [SearXNG](https://github.com/searxng/searxng)
   instance, so no search engine sees an API key or a profile. The planner
   names each query's *scope* — web, video, code, papers, forums — so a
   balloon question never asks PubMed. Small independent indexes
   (searchmysite, wiby) and video engines match short keyword strings, so long
   queries also go to them shortened to their key terms. Engines are drawn in
   the order their results have actually turned into kept sources for you,
   and the engine list itself is pruned from this install's own record: an
   engine that refuses every run and never contributes a page that is read
   is switched off (declared, so it can come back). Paper results that point
   at a paywalled DOI page are redirected to the open-access copy when one
   exists.
2. **Selection and triage** — results that share no word with the question
   are never fetched; storefronts and unreadable hosts are filtered; each
   source starts with two slots a round and earns more by keeping what it is
   given (sources that paid off on related earlier questions start higher);
   then a fast-model pass over titles and snippets drops the obvious junk
   before anything is fetched, with a floor so one bad verdict cannot empty a
   round. Scraped SEO clones of the same article collapse into one source.
3. **Fetch**, with an escalation ladder for pages that refuse: a retry
   presenting a real Chrome TLS fingerprint (recovers most CDN blocks —
   Britannica, Merriam-Webster — without a browser); the hashcash
   proof-of-work wall a growing number of forums hide behind, solved in
   milliseconds; and, if you opt in, a headless browser for JavaScript
   challenges. Reddit threads are read through Reddit's JSON API, and rebuilt
   from public archives when Reddit refuses (the archive with the comments
   wins). YouTube is read through caption transcripts, or through the video's
   own description when captions are thin or missing — a demonstration video
   is judged by what it shows, not by how much it narrates. Some walls defeat
   everything — those fail with an honest reason in the log.
4. **Notes** — the note-taker sees the whole extracted page, not a snippet,
   scores it against the question, and writes notes with evidence quotes.
   Every quote is checked against the page: one the page does not contain is
   repaired to the page's own sentence or removed. Sources below the
   relevance threshold are dropped; every read is remembered, which is how
   the install learns which domains pay off.
5. **Gap analysis** decides what to search next, or that the topic is
   saturated.
6. **Synthesis** writes the overview, with every claim cited and every
   citation checked against a real source. Notes are grouped by the part of
   the question they answer, so a thinly-sourced part cannot be compressed
   away by a crowded one on the way in, and within each part the sources are
   ordered by what kind of page they are: a standard or specification from a
   named body first, then research, then practitioners, then roundups. A
   roundup never silently overrules a standard. The document is as long as
   the research it carries — a target of 50 words per kept source, clamped —
   with a section per part of the question. When the question is choosing
   among named things, the candidates the sources cover are named too, each
   assessed by name, with a comparison table. Then a second pass hands the
   draft the strong sources it left uncited and asks for each to be placed
   where it adds something specific; the revision is kept only if it lost
   nothing the draft had.

**What the document admits about itself.** A confident report about nothing
reads exactly like a confident report about something, so the overview is
made to say where it is thin:

- **Checking what the question assumes** — at the top, when the question
  rested on something a published standard could settle. It says what the
  sources actually establish and whether the assumption holds, or says
  plainly that nothing settled it.
- **Not researched** — parts of your question no source answered. Synthesis
  is separately forbidden to write a section on them out of its own
  knowledge, and the claim is re-checked against the finished document before
  it is printed, so a part that *was* answered is not reported as a gap.
- **Researched but not used** — sources the run kept that the overview
  does not cite, with their numbers so you can read them yourself, and —
  where the synthesis named the cited source that covers the same ground —
  that reason. Parts of the question whose sources all went uncited are
  listed here too.

A run where nothing cleared the relevance bar opens with a **Thin result**
banner instead: the overview is built from the best partial matches
available, and says so rather than presenting them as findings.

The optional headless browser:

```bash
docker compose --profile browser up -d
```

then `BROWSER_SOLVER_URL=http://flaresolverr:8191` in `.env` (and
`COMPOSE_PROFILES=browser` so plain `docker compose up -d` keeps managing
it), or paste the URL in Settings.

---

## When search goes quiet

Google, Brave, DuckDuckGo and Startpage all throttle or CAPTCHA home
connections under research-volume traffic — and the stock SearXNG `general`
category is made up of exactly those. A throttled engine is invisible from a
chat window: results just quietly get worse. Here it is visible: a thin run
says which engines refused, and *How this run went* lists them on every run.

What helps, in order of effort:

1. **See who is answering.** `./scripts/check_engines.sh` asks SearXNG
   directly and prints your public address and per-engine health.
2. **Wait, or get a new address.** Rate limits are tied to your network
   address, not to this tool. A new address usually clears them; a
   power-cycle of modem and router only works if you leave them off long
   enough for the lease to lapse (15+ minutes, sometimes overnight).
3. **Add a keyed engine.** A [Brave Search API](https://brave.com/search/api/)
   key in `.env` as `BRAVE_API_KEY` turns on the API engine and turns off
   the scraper it replaces (1,000 requests a month free; the Learned page
   counts them against your billing cycle). A Marginalia key
   (`MARGINALIA_API_KEY`) adds a small independent index that specialises in
   the non-commercial web — on one measured run, every result it returned
   was a page no other engine had found. A GitHub fine-grained token (`GITHUB_CODE_TOKEN`,
   public repositories, read-only, no other permission) turns on GitHub
   code search for the code scope, which refuses unauthenticated calls. A
   free [CORE](https://core.ac.uk/services/api) key (`CORE_API_KEY`) adds
   its open-access paper index to the science category. The app enables
   each engine when its key is present and never writes a key into a
   committed file; `.env.example` has the click-path for each.
4. **Lean on what never blocks.** The `science` category (on by default)
   reaches Crossref, OpenAlex, Semantic Scholar and arXiv. The small
   independent indexes (mwmbl, searchmysite, wiby) answer when the majors
   will not.
5. **Clear the bench.** After an error SearXNG sits an engine out — half an
   hour for a CAPTCHA, a day for a Cloudflare or reCAPTCHA challenge — and
   says nothing. The bench lives in memory, so
   `docker compose restart searxng` clears it at once; do this before
   assuming an engine is gone for good. Blocks tied to your address are
   different: SearXNG's retry every three minutes keeps them fresh. The app
   keeps a bench of its own for those — an engine that has refused every
   search for half an hour (too many requests, a CAPTCHA, an access denial)
   is left out of queries for six hours, then tried once; an answer clears
   it, another refusal doubles the sit-out, up to two days. The run log
   says when an engine is benched and when it returns.

The `general` set here is not the stock one. DuckDuckGo, Startpage, Qwant
and Mojeek refused every one of the last 38 runs from a home address and
were costing a slice of each search for nothing, so they are off, along with
engines that cannot answer a research question at all (torrent trackers,
translators, currency converters). What remains answers: Brave (keyed),
Bing, mwmbl, searchmysite, wiby and Google CSE. The Learned page's engine
table is where to look before turning anything back on.

---

## Remote access (optional)

The app listens on port 8090 with no TLS and, by default, no password. Fine
on your own machine; **not** fine on the open internet.

To reach it from your phone, the intended path is a Cloudflare Tunnel with an
Access policy — no inbound ports, TLS handled for you, an identity check
before any request reaches the app:

Once it is reachable, **Add to Home Screen** installs it: the app ships a web
manifest, so it opens in its own window without browser chrome, with the
run's actions folded into a *More* menu and long overviews carrying an *On
this page* list. There is deliberately no service worker — pages are
rendered live and a run's progress streams — so nothing is cached stale.

If the site sits behind Cloudflare Access, the home-screen icon needs one
more thing: iOS fetches `apple-touch-icon` without your Access session, gets
the login page, and draws a letter tile instead. Add a second Access
application for the same hostname with path `static/*` and a single
**Bypass → Everyone** policy — policies attach to an application, not a path,
so it cannot live inside the main one — then re-add the app to the home
screen. What that exposes is CSS, JS, icons and the manifest; nothing under
`/static/` is run data.

```bash
# after creating a tunnel in the Cloudflare dashboard and adding an Access policy
echo 'CLOUDFLARE_TUNNEL_TOKEN=...' >> .env
docker compose --profile tunnel up -d
```

Set `WEB_PASSWORD` too. Access is the lock on the door; the password is the
lock on the room. Runs are tagged with whoever started them — the Cloudflare
identity, the `LAN_USER_LABEL`, or the Telegram sender — so a shared instance
shows who researched what.

---

## Operations

**Changing the engine list.** Engines live in `searxng/settings.yml`; the
keyed ones are appended from `.env` at container start, so never write a key
into the file. Validate before restarting, then restart only SearXNG (the
app keeps running):

```bash
docker compose exec -T searxng /usr/local/searxng/.venv/bin/python -c "import sys,yaml; yaml.safe_load(sys.stdin)" < searxng/settings.yml && docker compose restart searxng
```

Then `./scripts/check_engines.sh` shows who is answering.


- **Backup**: `tar czf backup.tgz data/` — that is the entire state.
- **Rebuild indexes**: `docker compose exec app python -m app.cli reindex`
  rebuilds the database, keyword search and vectors from the markdown on
  disk. The markdown is the source of truth.
- **Interrupted runs** (a restart mid-run) keep everything gathered so far
  and offer *Retry*. Queued runs survive restarts.
- **Logs**: `docker compose logs -f app`, plus a rotating `data/app.log`.
- **One process, one worker.** The job queue and live-progress bus live in
  memory in one process. Never add `--workers`, and run one app container per
  data directory.
- **Don't open `data/app.sqlite3` with a host `sqlite3` while the app is
  running** — it leaves stale sidecar files that break the container with
  "disk I/O error". Use the CLI instead. If it happens: stop the app, delete
  `data/app.sqlite3-shm` and `-wal`, start again.
- **Upgrading**: `git pull && docker compose build && docker compose up -d`,
  built on the machine that runs it — the torch stack is slow and flaky under
  emulation. Dev and prod images have separate tags, so building one never
  clobbers the other.

## Development

```bash
./scripts/dev.sh     # hot-reload stack; SearXNG exposed on 127.0.0.1:8081
./scripts/test.sh    # full test suite in the dev image
./scripts/check.sh   # what CI runs — tests plus a startup self-check
```

## Security notes

- Secrets live in `.env` (never committed) and `data/settings.json` (mode
  600). Neither is baked into the image.
- Fetched pages are rendered with raw HTML disabled everywhere and framed as
  untrusted data in every prompt — page content cannot inject markup into
  your UI or instructions into the model.
- The fetcher refuses private, loopback and link-local addresses
  (`ALLOW_PRIVATE_FETCH=true` if you genuinely need intranet sources), limits
  itself to one request per second per site, and sends a standard browser
  user agent because many CDNs reject unknown clients (`USER_AGENT` to
  identify yourself instead). robots.txt is **not** honoured by default —
  this reads the same handful of pages you would open by hand, it does not
  crawl. `RESPECT_ROBOTS=true` to enforce it.
- Run files are served through a filename allowlist plus a containment
  check, so a run id cannot read outside its own directory.

## Licence

MIT — see [LICENSE](LICENSE).
