"""Optional single-password auth: signed session cookie, everything gated
except /login, /health, and /static. No password configured → no gate."""
from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, TimestampSigner

COOKIE_NAME = "dr_session"
MAX_AGE = 30 * 86400
_EXEMPT_PREFIXES = ("/static/",)
_EXEMPT_PATHS = ("/login", "/health")


def load_signer(data_dir: Path) -> TimestampSigner:
    """Signer secret is generated once and persisted (0600), independent of
    the password so changing the password doesn't break the signer."""
    secret_file = data_dir / "auth_secret"
    if not secret_file.exists():
        fd = os.open(secret_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(secrets.token_hex(32))
    return TimestampSigner(secret_file.read_text().strip())


def session_valid(request: Request, signer: TimestampSigner) -> bool:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return False
    try:
        signer.unsign(token, max_age=MAX_AGE)
        return True
    except (BadSignature, SignatureExpired):
        return False


def install_auth(app, cfg_loader, signer: TimestampSigner) -> None:
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        cfg = cfg_loader()
        if not cfg.web_password:
            return await call_next(request)
        path = request.url.path
        if path in _EXEMPT_PATHS or path.startswith(_EXEMPT_PREFIXES):
            return await call_next(request)
        if session_valid(request, signer):
            return await call_next(request)
        return RedirectResponse(f"/login?next={quote(path)}", status_code=303)


def build_login_router(templates, cfg_loader, signer: TimestampSigner) -> APIRouter:
    router = APIRouter()

    @router.get("/login")
    async def login_page(request: Request, next: str = "/"):
        return templates.TemplateResponse(
            request, "login.html", {"next": next, "error": None})

    @router.post("/login")
    async def login_submit(request: Request, password: str = Form(""),
                           next: str = Form("/")):
        cfg = cfg_loader()
        if cfg.web_password and hmac.compare_digest(password, cfg.web_password):
            target = next if next.startswith("/") and not next.startswith("//") else "/"
            resp = RedirectResponse(target, status_code=303)
            resp.set_cookie(
                COOKIE_NAME, signer.sign("ok").decode(),
                max_age=MAX_AGE, httponly=True, samesite="lax",
            )
            return resp
        return templates.TemplateResponse(
            request, "login.html",
            {"next": next, "error": "Wrong password."}, status_code=401)

    @router.get("/logout")
    async def logout():
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie(COOKIE_NAME)
        return resp

    return router
