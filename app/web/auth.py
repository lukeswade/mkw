"""Optional single-password auth: signed session cookie, everything gated
except /login, /health, and /static. No password configured → no gate."""
from __future__ import annotations

import hmac
import os
import secrets
import time
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


class LoginThrottle:
    """Slow down password guessing.

    A single shared password with no rate limit is trivially brute-forceable
    once the UI is reachable from anywhere. In-memory is enough here: there is
    exactly one process, and losing the counters on restart costs an attacker
    far more time than it saves them.
    """

    FREE_ATTEMPTS = 5
    LOCKOUT_SECONDS = 300
    MAX_TRACKED = 1024

    def __init__(self) -> None:
        self._failures: dict[str, tuple[int, float]] = {}

    def _key(self, request: Request) -> str:
        return (request.headers.get("cf-connecting-ip")
                or (request.client.host if request.client else "unknown"))

    def retry_after(self, request: Request) -> int:
        count, last = self._failures.get(self._key(request), (0, 0.0))
        if count < self.FREE_ATTEMPTS:
            return 0
        # back off 2^n seconds past the free attempts, capped
        delay = min(self.LOCKOUT_SECONDS, 2 ** (count - self.FREE_ATTEMPTS + 1))
        remaining = int(last + delay - time.monotonic())
        return max(0, remaining)

    def record_failure(self, request: Request) -> None:
        if len(self._failures) > self.MAX_TRACKED:
            self._failures.clear()  # crude, but unbounded growth is worse
        key = self._key(request)
        count, _ = self._failures.get(key, (0, 0.0))
        self._failures[key] = (count + 1, time.monotonic())

    def reset(self, request: Request) -> None:
        self._failures.pop(self._key(request), None)


def build_login_router(templates, cfg_loader, signer: TimestampSigner) -> APIRouter:
    router = APIRouter()
    throttle = LoginThrottle()

    @router.get("/login")
    async def login_page(request: Request, next: str = "/"):
        return templates.TemplateResponse(
            request, "login.html", {"next": next, "error": None})

    @router.post("/login")
    async def login_submit(request: Request, password: str = Form(""),
                           next: str = Form("/")):
        cfg = cfg_loader()
        wait = throttle.retry_after(request)
        if wait:
            return templates.TemplateResponse(
                request, "login.html",
                {"next": next,
                 "error": f"Too many attempts. Try again in {wait}s."},
                status_code=429, headers={"Retry-After": str(wait)})

        if cfg.web_password and hmac.compare_digest(password, cfg.web_password):
            throttle.reset(request)
            target = next if next.startswith("/") and not next.startswith("//") else "/"
            resp = RedirectResponse(target, status_code=303)
            resp.set_cookie(
                COOKIE_NAME, signer.sign("ok").decode(),
                max_age=MAX_AGE, httponly=True, samesite="lax",
            )
            return resp
        throttle.record_failure(request)
        return templates.TemplateResponse(
            request, "login.html",
            {"next": next, "error": "Wrong password."}, status_code=401)

    @router.get("/logout")
    async def logout():
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie(COOKIE_NAME)
        return resp

    return router
