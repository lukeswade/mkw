# 🔭 Deep Research

A self-hosted deep-research agent — a personal Perplexity that digs much deeper.
Give it a question, a depth (1–10), and a recency window; it unpacks the
question into targeted web searches, reads the results, writes per-source
notes, lets the gaps steer further search rounds, and ends with a cited
overview plus follow-up suggestions. Everything runs on your own machine
except the LLM API calls (and even those can go to a local llama.cpp).

- **Web UI** with live progress, a research library (keyword + semantic
  search), a knowledge graph connecting runs and entities, and an "Ask"
  page that answers from your accumulated research with citations.
- **Telegram bot**: start runs, watch progress, get the overview delivered.
- **Everything is plain markdown on disk** (`data/research_data/<run>/`) —
  SQLite/Chroma are derived indexes you can rebuild any time with
  `reindex`.

## Quick start (Ubuntu server or any Docker host)

```bash
# 1. prerequisites: Docker Engine + compose plugin
#    https://docs.docker.com/engine/install/ubuntu/

# 2. get the code
git clone <your-remote-or-copy> deep-research && cd deep-research

# 3. configure secrets
cp .env.example .env
nano .env                     # add DEEPSEEK_API_KEY, SEARXNG_SECRET
mkdir -p data && sudo chown -R 1000:1000 data   # container runs as uid 1000

# 4. build & start (build natively on the server — do not cross-build)
docker compose build
docker compose up -d

# 5. open http://<server-ip>:8090
```

Everything in `.env` can also be set later on the **Settings** page
(stored in `data/settings.json`, chmod 600; settings-page values override
`.env`). If you expose the UI beyond localhost, set `WEB_PASSWORD`.

### DeepSeek API key

Create one at <https://platform.deepseek.com> → *API keys*. Put it in `.env`
(`DEEPSEEK_API_KEY=…`) or paste it into Settings → LLM. A depth-3 run
typically costs a few cents; per-run token usage and estimated cost show in
the run's stats footer.

### Telegram bot (optional)

1. Message **@BotFather** → `/newbot` → copy the token into `.env`
   (`TELEGRAM_BOT_TOKEN=…`) or Settings → Telegram.
2. Restart: `docker compose restart app`.
3. Message your bot `/id`, put the number in `TELEGRAM_ALLOWED_USER_IDS`,
   restart again.
4. Send it a question — it walks you through depth & recency, then delivers
   the overview when the run finishes. `/help` lists all commands.

### Local LLM instead of DeepSeek (optional)

Run any OpenAI-compatible server, e.g. llama.cpp:

```bash
llama-server -m your-model.gguf --host 0.0.0.0 --port 8080 -c 32768
```

Then Settings → LLM → provider **Local**, base URL
`http://host.docker.internal:8080/v1` (or the machine's LAN address on
Linux — add `extra_hosts: ["host.docker.internal:host-gateway"]` to the app
service if you use that hostname). A commented llama.cpp service example is
in `docker-compose.yml`. Use a model that handles JSON output well (Qwen
2.5 32B+, Llama 3.3 70B, …) — the pipeline repairs malformed JSON but can't
fix a model that can't reason.

## Using it

- **Depth** = maximum search rounds. Each round runs several targeted
  queries, reads the useful results, then a gap analysis decides what to
  search next. Runs stop early when saturated (nothing material left to
  find), after two dry rounds, or at per-depth source/LLM-call caps.
- **Recency** maps to search-engine time filters plus date checks on the
  documents themselves. Engine date metadata is imperfect: undated sources
  are kept but tagged, and the synthesis is told to prefer dated, in-window
  material. Treat 3-months/6-months/3-years as best effort.
- Every run directory contains `overview.md` (cited synthesis),
  `further-research.md`, `sources.md`, per-source `findings/*.md`, a
  `rounds/` log, `meta.json`, and `events.jsonl`. The "Further research" tab
  has one-click **Run this** buttons.
- **Ask** (web or `/ask` in Telegram) answers questions from everything
  you've researched so far, citing the runs. New runs automatically build
  on related earlier research and link to it.

### CLI

```bash
docker compose exec app python -m app.cli run "your question" --depth 3 --recency month
docker compose exec app python -m app.cli runs
docker compose exec app python -m app.cli ask "question over the corpus"
docker compose exec app python -m app.cli reindex   # rebuild indexes from the .md files
```

## Development (macOS or Linux)

```bash
./scripts/dev.sh        # hot-reload stack; SearXNG exposed on 127.0.0.1:8081
./scripts/test.sh       # run the test suite in the container
./scripts/smoke_live.sh # real end-to-end depth-1 run (needs DEEPSEEK_API_KEY)
```

Build natively on each architecture (arm64 Mac for dev, amd64 server for
prod). Don't cross-build with QEMU — the torch stack is slow and flaky under
emulation; on the server just `git pull && docker compose build`.

## Operations

- **Backup**: `tar czf backup.tgz data/` — that's the whole state
  (settings, SQLite, Chroma, research markdown).
- **Recover indexes**: `docker compose exec app python -m app.cli reindex`
  rebuilds SQLite rows, keyword search, and vectors from the markdown on
  disk (the markdown is the source of truth).
- **Interrupted runs** (restart mid-run) keep everything already gathered;
  the run page offers *Retry with same parameters*. Queued runs survive
  restarts.
- **Logs**: `docker compose logs -f app`, plus rotating `data/app.log`.
- **Never open `data/app.sqlite3` with a host `sqlite3` while the app is
  running** — WAL mode over a bind mount leaves stale `-shm`/`-wal` sidecars
  that break the containerized app with "disk I/O error". Inspect via
  `docker compose exec app python -m app.cli runs` instead. If it ever
  happens: stop the app, delete `data/app.sqlite3-shm` and `-wal` (safe when
  `-wal` is 0 bytes), start again.
- **One process, one worker**: the job queue, SSE bus, and run registry are
  in-process. Never add `--workers` to uvicorn; run exactly one app
  container per data directory (a second instance would also fight for the
  Telegram token — you'd see 409 Conflict in the logs).
- SearXNG engines occasionally rate-limit (`unresponsive_engines` in logs);
  other engines fill in and the pipeline rephrases on empty rounds. The
  SearXNG image is pinned by digest in `docker-compose.yml`; upgrade it
  deliberately.

## Security notes

- Secrets live in `.env` (never committed) and `data/settings.json`
  (mode 600). Neither is baked into the image.
- The web UI binds to port 8090; set `WEB_PASSWORD` if it's reachable by
  anyone but you. The Telegram bot only obeys allowlisted user ids.
- Fetched pages are rendered with raw HTML disabled everywhere (stored-XSS
  guard) and framed as untrusted data in prompts. The fetcher refuses
  private/loopback addresses (SSRF guard; `ALLOW_PRIVATE_FETCH=true` if you
  genuinely need intranet sources) and honors robots.txt by default.
