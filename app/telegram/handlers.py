"""Telegram command handlers: guided /new flow, status, ask, cancel.

All messages are plain text (no MarkdownV2 escaping minefield). One status
message per run is edited in place, throttled to ≥3s.
"""
from __future__ import annotations

import asyncio
import logging
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import (ApplicationHandlerStop, CallbackQueryHandler,
                          CommandHandler, ContextTypes, ConversationHandler,
                          MessageHandler, filters)

from app.models import RECENCY_CHOICES, RECENCY_LABELS, RunParams
from app.research.storage import RunStore

log = logging.getLogger(__name__)

QUERY, DEPTH, RECENCY, CONFIRM = range(4)
TG_LIMIT = 4000  # a little under Telegram's 4096 hard cap

HELP = (
    "🔭 Deep Research bot\n\n"
    "Send me a research question (or /new) and I'll walk you through "
    "depth and recency, then run the research and send you the results.\n\n"
    "/new — start a research run\n"
    "/status — active and queued runs\n"
    "/list — recent runs\n"
    "/ask <question> — answer from your accumulated research\n"
    "/cancel_run — cancel the active run\n"
    "/id — show your Telegram user id"
)


# ---- helpers ---------------------------------------------------------------

def split_message(text: str, limit: int = TG_LIMIT) -> list[str]:
    """Split on paragraph (then line, then hard) boundaries under the limit."""
    text = (text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(para) <= limit:
            current = para
            continue
        for line in para.split("\n"):
            candidate = f"{current}\n{line}" if current else line
            if len(candidate) <= limit:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = ""
            while len(line) > limit:
                chunks.append(line[:limit])
                line = line[limit:]
            current = line
    if current:
        chunks.append(current)
    return chunks


def extract_tldr(overview_md: str, fallback_chars: int = 900) -> str:
    """Pull the TL;DR section out of an overview, else the head of it."""
    lines = (overview_md or "").splitlines()
    out: list[str] = []
    in_tldr = False
    for line in lines:
        if line.lower().startswith("## tl;dr"):
            in_tldr = True
            continue
        if in_tldr and line.startswith("#"):
            break
        if in_tldr:
            out.append(line)
    text = "\n".join(out).strip()
    if not text:
        text = (overview_md or "").strip()[:fallback_chars]
    return text[:2000]


def status_line(counters: dict) -> str:
    parts = []
    status = counters.get("status", "running")
    icon = {"queued": "⏳", "running": "🔎", "completed": "✅",
            "failed": "❌", "cancelled": "🚫", "interrupted": "⚠️"}.get(status, "🔎")
    parts.append(f"{icon} {status}")
    if counters.get("round"):
        parts.append(f"round {counters['round']}/{counters['depth']}")
    if counters.get("findings"):
        parts.append(f"{counters['findings']} sources kept")
    if counters.get("phase"):
        parts.append(counters["phase"])
    return " · ".join(parts)


def _deps(context: ContextTypes.DEFAULT_TYPE):
    bd = context.application.bot_data
    return bd["orch"], bd["repo"], bd["bus"], bd["cfg_loader"]


def _authorized(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user = update.effective_user
    return bool(user) and user.id in context.application.bot_data["allowed_ids"]


# ---- run progress watcher ------------------------------------------------------

async def watch_run(application, chat_id: int, run_id: str) -> None:
    bd = application.bot_data
    repo, bus, cfg_loader = bd["repo"], bd["bus"], bd["cfg_loader"]
    bot = application.bot
    row = repo.get_run(run_id)
    if row is None:
        return
    store = RunStore(cfg_loader().research_dir / row["dir"])
    replay, queue = bus.subscribe(run_id, store)

    msg = await bot.send_message(chat_id, f"⏳ queued: {row['query'][:200]}")
    counters: dict = {"status": row["status"], "depth": row["depth"]}
    last_edit = 0.0

    async def maybe_edit(force: bool = False) -> None:
        nonlocal last_edit
        if not force and time.monotonic() - last_edit < 3.0:
            return
        try:
            await bot.edit_message_text(status_line(counters), chat_id,
                                        msg.message_id)
            last_edit = time.monotonic()
        except BadRequest as e:
            if "not modified" not in str(e).lower():
                log.debug("edit failed: %s", e)

    def ingest(e: dict) -> bool:
        t = e.get("type")
        if t == "status":
            counters["status"] = e.get("status")
        elif t == "round_start":
            counters["round"] = e.get("round")
            counters["depth"] = e.get("depth")
            counters["phase"] = None
        elif t == "finding":
            counters["findings"] = counters.get("findings", 0) + 1
        elif t == "phase":
            counters["phase"] = e.get("phase")
        return t == "done"

    done = False
    for e in replay:
        done = ingest(e) or done
    await maybe_edit(force=True)
    while not done and queue is not None:
        try:
            e = await asyncio.wait_for(queue.get(), timeout=600)
        except asyncio.TimeoutError:
            # safety net: if the run somehow died without a done event
            row = repo.get_run(run_id)
            if row is None or row["status"] not in ("queued", "running"):
                break
            continue
        done = ingest(e)
        await maybe_edit(force=done)

    # final result
    row = repo.get_run(run_id)
    counters["status"] = row["status"]
    await maybe_edit(force=True)
    if row["status"] == "completed":
        title = row["title"] or row["query"]
        overview = (store.overview_path.read_text(encoding="utf-8")
                    if store.overview_path.exists() else "")
        for part in split_message(f"✅ {title}\n\n{extract_tldr(overview)}"):
            await bot.send_message(chat_id, part)
        for path, name in ((store.overview_path, "overview.md"),
                           (store.sources_path, "sources.md")):
            if path.exists():
                await bot.send_document(chat_id, document=path.read_bytes(),
                                        filename=f"{run_id}_{name}")
        meta = store.read_meta()
        if meta.get("followups"):
            lines = ["Suggested follow-ups (start with /new):"]
            for fu in meta["followups"][:5]:
                lines.append(f"• {fu['query']}")
            await bot.send_message(chat_id, "\n".join(lines))
    else:
        detail = row["error"] or row["stop_reason"] or ""
        await bot.send_message(
            chat_id, f"Run ended: {row['status']}. {detail}".strip())


def spawn_watcher(application, chat_id: int, run_id: str) -> None:
    task = asyncio.create_task(watch_run(application, chat_id, run_id))
    watchers: set = application.bot_data["watchers"]
    watchers.add(task)
    task.add_done_callback(watchers.discard)


# ---- /new conversation ------------------------------------------------------------

def _depth_keyboard() -> InlineKeyboardMarkup:
    rows = [[2, 3, 4], [6, 8, 10]]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(str(d), callback_data=f"depth:{d}") for d in row]
        for row in rows])


def _recency_keyboard() -> InlineKeyboardMarkup:
    rows = [RECENCY_CHOICES[:3], RECENCY_CHOICES[3:6], RECENCY_CHOICES[6:]]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(RECENCY_LABELS[r], callback_data=f"recency:{r}")
         for r in row] for row in rows])


async def new_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("What should I research? (or /done to abort)")
    return QUERY


async def query_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = (update.message.text or "").strip()
    if len(query) < 3:
        await update.message.reply_text("A bit longer, please — what should I research?")
        return QUERY
    context.user_data["query"] = query[:2000]
    await update.message.reply_text(
        f"Research: {query[:300]}\n\nHow deep? (max search rounds)",
        reply_markup=_depth_keyboard())
    return DEPTH


async def depth_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    cq = update.callback_query
    await cq.answer()
    context.user_data["depth"] = int(cq.data.split(":")[1])
    await cq.edit_message_text(
        f"Depth {context.user_data['depth']}. How recent should sources be?")
    await cq.message.reply_text("Recency focus:", reply_markup=_recency_keyboard())
    return RECENCY


async def recency_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    cq = update.callback_query
    await cq.answer()
    recency = cq.data.split(":")[1]
    context.user_data["recency"] = recency
    q = context.user_data["query"]
    depth = context.user_data["depth"]
    await cq.edit_message_text(
        f"Ready:\n\n{q[:500]}\n\nDepth {depth} · {RECENCY_LABELS[recency]}",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("🚀 Start", callback_data="confirm:go"),
            InlineKeyboardButton("✖ Cancel", callback_data="confirm:no"),
        ]]))
    return CONFIRM


async def confirmed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    cq = update.callback_query
    await cq.answer()
    if cq.data != "confirm:go":
        await cq.edit_message_text("Cancelled.")
        return ConversationHandler.END
    orch, repo, _bus, _cfg = _deps(context)
    user = update.effective_user
    params = RunParams(
        query=context.user_data["query"],
        depth=context.user_data["depth"],
        recency=context.user_data["recency"],
        origin="telegram",
        origin_chat_id=update.effective_chat.id,
        created_by=(user.first_name or user.username or "telegram")[:120]
        if user else "telegram")
    run_id = orch.enqueue(params)
    await cq.edit_message_text("Research started.")
    spawn_watcher(context.application, update.effective_chat.id, run_id)
    return ConversationHandler.END


async def conv_abort(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Okay, aborted.")
    return ConversationHandler.END


# ---- simple commands ------------------------------------------------------------

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP)


async def id_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"Your Telegram user id: {update.effective_user.id}")


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _orch, repo, _bus, _cfg = _deps(context)
    rows = repo.runs_with_status("running", "queued")
    if not rows:
        await update.message.reply_text("Nothing running or queued.")
        return
    lines = [f"{'🔎' if r['status'] == 'running' else '⏳'} [{r['status']}] "
             f"{(r['title'] or r['query'])[:120]}" for r in rows]
    await update.message.reply_text("\n".join(lines))


async def list_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _orch, repo, _bus, _cfg = _deps(context)
    rows = repo.list_runs(limit=10)
    if not rows:
        await update.message.reply_text("No research runs yet.")
        return
    icon = {"completed": "✅", "running": "🔎", "queued": "⏳", "failed": "❌",
            "cancelled": "🚫", "interrupted": "⚠️"}
    lines = [f"{icon.get(r['status'], '·')} {(r['title'] or r['query'])[:110]}"
             f"  (d{r['depth']}, {r['recency']})" for r in rows]
    await update.message.reply_text("\n".join(lines))


async def cancel_run_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    orch, repo, _bus, _cfg = _deps(context)
    active = list(orch.active)
    if active and orch.cancel(active[0]):
        await update.message.reply_text("Cancelling the active run…")
        return
    queued = repo.runs_with_status("queued")
    if queued and orch.cancel(queued[-1]["id"]):
        await update.message.reply_text("Removed the most recent queued run.")
        return
    await update.message.reply_text("Nothing to cancel.")


async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    orch, repo, _bus, _cfg = _deps(context)
    question = " ".join(context.args or []).strip()
    if not question:
        await update.message.reply_text("Usage: /ask <question>")
        return
    if orch.rag is None:
        await update.message.reply_text("The knowledge layer is unavailable.")
        return
    await update.message.reply_text("Looking through your research…")
    try:
        result = await orch.rag.ask(question, repo)
    except Exception as e:
        log.exception("ask failed")
        await update.message.reply_text(f"Ask failed: {e}")
        return
    answer = result["answer"]
    if result["sources"]:
        answer += "\n\nDrawn from: " + "; ".join(
            s["title"] for s in result["sources"][:4])
    for part in split_message(answer):
        await update.message.reply_text(part)


async def unauthorized(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if _authorized(update, context):
        return  # authorized chatter that matched nothing: stay silent
    user = update.effective_user
    if update.effective_message and user:
        log.info("unauthorized telegram access from user id %s", user.id)
        await update.effective_message.reply_text(
            f"Not authorized. Your Telegram user id is {user.id} — add it to "
            "TELEGRAM_ALLOWED_USER_IDS (Settings page or .env) and restart.")
    raise ApplicationHandlerStop


# ---- registration ------------------------------------------------------------------

async def _on_error(update, context) -> None:
    log.error("telegram handler error", exc_info=context.error)


def register_handlers(app) -> None:
    allowed = app.bot_data["allowed_ids"]
    user_ok = filters.User(user_id=list(allowed)) if allowed else filters.User(user_id=[-1])

    conv = ConversationHandler(
        entry_points=[
            CommandHandler("new", new_cmd, filters=user_ok),
            MessageHandler(filters.TEXT & ~filters.COMMAND & user_ok,
                           query_received),
        ],
        states={
            QUERY: [MessageHandler(filters.TEXT & ~filters.COMMAND & user_ok,
                                   query_received)],
            DEPTH: [CallbackQueryHandler(depth_chosen, pattern=r"^depth:\d+$")],
            RECENCY: [CallbackQueryHandler(recency_chosen,
                                           pattern=r"^recency:\w+$")],
            CONFIRM: [CallbackQueryHandler(confirmed, pattern=r"^confirm:")],
        },
        fallbacks=[CommandHandler("done", conv_abort, filters=user_ok)],
    )

    app.add_handler(CommandHandler("id", id_cmd))  # open to anyone
    app.add_handler(CommandHandler(["start", "help"], start_cmd, filters=user_ok))
    app.add_handler(conv)
    app.add_handler(CommandHandler("status", status_cmd, filters=user_ok))
    app.add_handler(CommandHandler("list", list_cmd, filters=user_ok))
    app.add_handler(CommandHandler("cancel_run", cancel_run_cmd, filters=user_ok))
    app.add_handler(CommandHandler("ask", ask_cmd, filters=user_ok))
    app.add_handler(MessageHandler(filters.ALL, unauthorized))
    app.add_error_handler(_on_error)
