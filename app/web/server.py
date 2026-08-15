"""FastAPI app factory + lifespan wiring (db → recovery → worker → bot).

Single-process by design: the SSE bus, run registry, and job queue live in
this process, so uvicorn MUST run with exactly one worker.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import zlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import selfcheck
from app.config import Settings, load_settings
from app.llm import providers
from app.db import Repo, connect
from app.models import RECENCY_CHOICES, RECENCY_LABELS
from app.research.orchestrator import Orchestrator
from app.research.progress import ProgressBus
from app.web.auth import build_login_router, install_auth, load_signer
from app.refresh_worker import refresh_loop

log = logging.getLogger(__name__)

_HERE = Path(__file__).parent


def _build_rag(cfg: Settings):
    """Knowledge layer (Chroma + embeddings). Optional so the web app still
    boots if the ML deps are missing."""
    try:
        from app.rag.service import RagService
        return RagService(cfg)
    except ImportError as e:
        log.warning("knowledge layer disabled: %s", e)
        return None


def _asset_url(name: str) -> str:
    """Cache-bust by content hash.

    The previous scheme was a hand-incremented ?v=N in base.html, which was
    forgotten twice in eleven commits and shipped stale CSS to returning users
    both times. A hash cannot be forgotten.
    """
    path = _HERE / "static" / name
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
    except OSError:
        return f"/static/{name}"
    return f"/static/{name}?v={digest}"


def _build_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=str(_HERE / "templates"))
    templates.env.globals.update(
        RECENCY_CHOICES=RECENCY_CHOICES,
        RECENCY_LABELS=RECENCY_LABELS,
        PROVIDERS=providers.PROVIDERS,
        asset=_asset_url,
    )
    templates.env.filters["fromjson"] = lambda s: json.loads(s) if s else {}
    # Stable colour per initiator name. crc32, not hash(): python salts hash()
    # per process, which would recolour everyone on every restart.
    templates.env.filters["user_hue"] = (
        lambda name: zlib.crc32(str(name).lower().encode()) % 360)
    return templates


def _setup_logging(cfg: Settings) -> None:
    root = logging.getLogger()
    if any(getattr(h, "_dr_file", False) for h in root.handlers):
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx logs every request at INFO. Telegram long-polls every 10s and the
    # bot token is embedded in that URL, so leaving this on writes the token to
    # disk thousands of times a day and buries everything else.
    for noisy in ("httpx", "httpcore", "telegram.ext.Updater"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(cfg.data_path / "app.log",
                                 maxBytes=5_000_000, backupCount=3)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"))
        fh._dr_file = True  # marker so reloads don't stack handlers
        root.addHandler(fh)
    except OSError:
        pass  # unwritable data dir surfaces elsewhere with a better message


def create_app(cfg: Settings | None = None, enable_worker: bool = True,
               enable_bot: bool = True) -> FastAPI:
    fixed_cfg = cfg
    cfg_loader = (lambda: fixed_cfg) if fixed_cfg is not None else load_settings
    base_cfg = cfg_loader()
    base_cfg.ensure_dirs()
    _setup_logging(base_cfg)
    # Fail at boot rather than on the first document of the first run.
    selfcheck.run_all(base_cfg)

    templates = _build_templates()
    signer = load_signer(base_cfg.data_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        conn = connect(base_cfg.db_path)
        repo = Repo(conn)
        bus = ProgressBus()
        rag = _build_rag(base_cfg)
        orch = Orchestrator(cfg_loader, repo, bus, rag=rag)
        app.state.repo = repo
        app.state.bus = bus
        app.state.orch = orch
        app.state.rag = rag
        app.state.cfg_loader = cfg_loader
        app.state.templates = templates

        orch.recover()
        refresh_task = None
        if enable_worker:
            orch.start()
            refresh_task = asyncio.create_task(
                refresh_loop(orch, repo), name="evergreen-refresh")

        bot = None
        if enable_bot:
            try:
                from app.telegram.bot import start_bot
                bot = await start_bot(cfg_loader, orch, repo, bus)
            except ImportError:
                log.info("telegram module unavailable; web-only mode")
            except Exception:
                log.exception("telegram bot failed to start; web-only mode")
        app.state.bot = bot

        try:
            yield
        finally:
            if bot is not None:
                from app.telegram.bot import stop_bot
                await stop_bot(bot)
            if refresh_task is not None:
                refresh_task.cancel()
                # await it, or the task is garbage-collected mid-cancel and
                # python logs "Task was destroyed but it is pending"
                with contextlib.suppress(asyncio.CancelledError):
                    await refresh_task
            if enable_worker:
                await orch.stop()
            conn.close()

    app = FastAPI(title="Deep Research", docs_url=None, redoc_url=None,
                  lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    install_auth(app, cfg_loader, signer)
    app.include_router(build_login_router(templates, cfg_loader, signer))

    from app.web.routes_runs import router as runs_router
    from app.web.routes_library import router as library_router
    from app.web.routes_settings import router as settings_router
    from app.web.routes_ask import router as ask_router
    from app.web.routes_readme import router as readme_router
    app.include_router(runs_router)
    app.include_router(library_router)
    app.include_router(settings_router)
    app.include_router(ask_router)
    app.include_router(readme_router)

    app.mount("/static", StaticFiles(directory=str(_HERE / "static")),
              name="static")
    return app
