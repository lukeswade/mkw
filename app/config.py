"""Application settings.

Precedence (lowest → highest):
    dataclass defaults  ←  environment variables  ←  <data_dir>/settings.json

settings.json holds values saved from the web Settings page and is written
atomically with mode 0600 (it can contain API keys).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

SETTINGS_FILENAME = "settings.json"

# dataclass field name → environment variable
ENV_MAP = {
    "llm_provider": "LLM_PROVIDER",
    "llm_base_url": "LLM_BASE_URL",
    "llm_model": "LLM_MODEL",
    "llm_api_key": "LLM_API_KEY",
    "fast_model": "FAST_MODEL",
    "deepseek_api_key": "DEEPSEEK_API_KEY",
    "deepseek_base_url": "DEEPSEEK_BASE_URL",
    "deepseek_model": "DEEPSEEK_MODEL",
    "local_llm_base_url": "LOCAL_LLM_BASE_URL",
    "local_llm_model": "LOCAL_LLM_MODEL",
    "local_llm_api_key": "LOCAL_LLM_API_KEY",
    "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
    "telegram_allowed_user_ids": "TELEGRAM_ALLOWED_USER_IDS",
    "web_password": "WEB_PASSWORD",
    "lan_user_label": "LAN_USER_LABEL",
    "searxng_url": "SEARXNG_URL",
    "data_dir": "DATA_DIR",
    "fetch_concurrency": "FETCH_CONCURRENCY",
    "llm_concurrency": "LLM_CONCURRENCY",
    "embedding_model": "EMBEDDING_MODEL",
    "embedding_base_url": "EMBEDDING_BASE_URL",
    "embedding_api_key": "EMBEDDING_API_KEY",
    "llm_timeout": "LLM_TIMEOUT",
    "llm_call_ceiling": "LLM_CALL_CEILING",
    "search_categories": "SEARCH_CATEGORIES",
    "search_concurrency": "SEARCH_CONCURRENCY",
    "relevance_threshold": "RELEVANCE_THRESHOLD",
    "reference_chasing": "REFERENCE_CHASING",
    "blocked_domains": "BLOCKED_DOMAINS",
    "authority_sites": "AUTHORITY_SITES",
    "feeds": "FEEDS",
    "respect_robots": "RESPECT_ROBOTS",
    "allow_private_fetch": "ALLOW_PRIVATE_FETCH",
    "user_agent": "USER_AGENT",
    "browser_impersonation": "BROWSER_IMPERSONATION",
    "browser_solver_url": "BROWSER_SOLVER_URL",
}

# Fields the web Settings page is allowed to persist into settings.json.
UI_EDITABLE = {
    "llm_provider",
    "llm_base_url",
    "llm_model",
    "llm_api_key",
    "fast_model",
    "deepseek_api_key",
    "deepseek_base_url",
    "deepseek_model",
    "local_llm_base_url",
    "local_llm_model",
    "local_llm_api_key",
    "telegram_bot_token",
    "telegram_allowed_user_ids",
    "web_password",
    "lan_user_label",
    "searxng_url",
    "search_categories",
    "relevance_threshold",
    "reference_chasing",
    "blocked_domains",
    "authority_sites",
    "feeds",
    "respect_robots",
    "llm_concurrency",
    "embedding_model",
    "embedding_base_url",
    "embedding_api_key",
    "browser_impersonation",
    "browser_solver_url",
}

SECRET_FIELDS = {"llm_api_key", "deepseek_api_key", "local_llm_api_key",
                 "telegram_bot_token", "web_password",
                 "embedding_api_key"}


@dataclass
class Settings:
    # Any OpenAI-compatible endpoint. `llm_provider` selects a preset from
    # app.llm.providers; the three fields below override the preset's defaults
    # and are what the client actually uses.
    llm_provider: str = "deepseek"
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    # Optional cheaper/faster model for the high-volume per-document note
    # calls, leaving the main model for planning and synthesis. Same endpoint.
    fast_model: str = ""

    # --- legacy fields, still read so existing .env files keep working ---
    deepseek_api_key: str = ""
    deepseek_base_url: str = ""
    deepseek_model: str = ""
    local_llm_base_url: str = ""
    local_llm_model: str = ""
    local_llm_api_key: str = ""

    telegram_bot_token: str = ""
    telegram_allowed_user_ids: str = ""
    web_password: str = ""
    # Shown on runs started from the LAN, where there is no Cloudflare
    # Access identity to read. Set it to your name.
    lan_user_label: str = "LAN"
    searxng_url: str = "http://searxng:8080"
    data_dir: str = "./data"
    fetch_concurrency: int = 8
    llm_concurrency: int = 4
    # Embeddings. Empty model = the bge-small baked into the image (fully
    # offline, no server needed). Naming a model routes embeddings to an
    # OpenAI-compatible /embeddings endpoint instead — e.g. a local oMLX or
    # LM Studio serving a stronger retrieval model. Blank URL/key reuse the
    # LLM endpoint's, which is the common case for a local server.
    embedding_model: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    llm_timeout: int = 180
    # Hard wall-clock ceiling on a single model call, total, not
    # idle. llm_timeout is the SDK's idle read timeout: it fires only
    # when the server sends NOTHING for that long. A local thinking
    # model can trickle reasoning tokens for an unbounded time, so the
    # idle clock keeps resetting and the call never returns — one such
    # call once hung a whole round for 65 minutes. This bounds the
    # total; raise it only if a legitimate single call on slow hardware
    # is being cut off (a thin run will say a source timed out).
    llm_call_ceiling: int = 600
    # SearXNG categories to query. general alone is four engines that all
    # rate-limit; science backfills with sources that do not.
    search_categories: str = "general,science"
    search_concurrency: int = 2
    relevance_threshold: int = 4
    # Fetch the most promising references cited by kept sources (one hop,
    # capped per round). The links a good source chooses are often better
    # than anything a search engine returns.
    reference_chasing: bool = True
    # Comma-separated domains never worth fetching for you (e.g. pinterest.com)
    blocked_domains: str = ""
    # Curated sites holding authoritative primary documents, offered to the
    # planner and gap analysis for site:-scoped queries when the topic fits
    # (search engines barely index their deep pages, so they rarely surface
    # on their own). One per line, "domain — what it holds".
    authority_sites: str = ("charm.li — full factory service manuals for most "
                            "cars and trucks, every make/model/year")
    # RSS/Atom reading list for brief runs, one URL per line. Empty means
    # briefs cannot run — the pipeline says so rather than producing nothing.
    feeds: str = ""
    # Off by default: this fetches a handful of pages a person could
    # open by hand, at one request per second per domain, with an
    # identifiable user agent. Turn it on if you want the crawler
    # convention enforced anyway.
    respect_robots: bool = False
    allow_private_fetch: bool = False
    # A standard browser UA, because ~a quarter of fetches were coming back
    # 403 — CDNs reject unknown clients outright. This tool reads pages a
    # person could open by hand, which is exactly what a browser UA claims.
    # Set USER_AGENT to override (e.g. back to an identifying string).
    user_agent: str = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/126.0.0.0 Safari/537.36")
    # When a fetch is refused with a bot-wall status (403/429/challenge), retry
    # it once presenting a real Chrome TLS fingerprint (curl_cffi). Most CDN
    # blocks key on the TLS handshake, not behavior, so this recovers them
    # without running a browser.
    browser_impersonation: bool = True
    # Optional last resort for pages that refuse even a real fingerprint:
    # a FlareSolverr endpoint (docker compose --profile browser up -d, then
    # http://flaresolverr:8191). It drives a real headless browser through
    # JavaScript challenges. Empty = escalation stops at impersonation.
    # Points at the compose service. Harmless when the "browser" profile
    # is not running: an unreachable solver refuses in ~40ms, so the only
    # cost is that, per challenged page — and anyone who does start the
    # profile gets the last rung with no settings change.
    browser_solver_url: str = "http://flaresolverr:8191"

    # --- resolved LLM endpoint -------------------------------------------
    # Precedence: explicit generic field → legacy provider-specific field →
    # the preset's default. The legacy step is what keeps a .env written
    # against the old DEEPSEEK_* / LOCAL_LLM_* names working untouched.
    def _legacy(self, field: str) -> str:
        if self.llm_provider == "deepseek":
            return {"base_url": self.deepseek_base_url,
                    "model": self.deepseek_model,
                    "api_key": self.deepseek_api_key}.get(field, "")
        return {"base_url": self.local_llm_base_url,
                "model": self.local_llm_model,
                "api_key": self.local_llm_api_key}.get(field, "")

    @property
    def provider(self):
        from app.llm import providers
        return providers.get(self.llm_provider)

    @property
    def resolved_base_url(self) -> str:
        return (self.llm_base_url or self._legacy("base_url")
                or self.provider.base_url)

    @property
    def resolved_model(self) -> str:
        return (self.llm_model or self._legacy("model")
                or self.provider.default_model)

    @property
    def resolved_api_key(self) -> str:
        return self.llm_api_key or self._legacy("api_key")

    @property
    def llm_is_configured(self) -> bool:
        """Enough to attempt a call: local servers need no key, clouds do."""
        from app.llm import providers
        if not self.resolved_model:
            return False
        if providers.is_local(self.llm_provider):
            return bool(self.resolved_base_url)
        return bool(self.resolved_api_key)

    # --- derived paths ---
    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def research_dir(self) -> Path:
        return self.data_path / "research_data"

    @property
    def db_path(self) -> Path:
        return self.data_path / "app.sqlite3"

    @property
    def chroma_dir(self) -> Path:
        return self.data_path / "chroma"

    @property
    def settings_path(self) -> Path:
        return self.data_path / SETTINGS_FILENAME

    def ensure_dirs(self) -> None:
        for p in (self.data_path, self.research_dir, self.chroma_dir):
            p.mkdir(parents=True, exist_ok=True)

    @property
    def allowed_telegram_ids(self) -> set[int]:
        out: set[int] = set()
        for part in self.telegram_allowed_user_ids.replace(";", ",").split(","):
            part = part.strip()
            if part.isdigit():
                out.add(int(part))
        return out


def _coerce(value: Any, like: Any) -> Any:
    if isinstance(like, bool):
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(like, int):
        try:
            return int(str(value).strip())
        except ValueError:
            return like
    return str(value)


def load_settings(data_dir: str | None = None) -> Settings:
    dd = data_dir or os.environ.get("DATA_DIR", "").strip() or "./data"
    file_values: dict[str, Any] = {}
    settings_file = Path(dd) / SETTINGS_FILENAME
    if settings_file.exists():
        try:
            file_values = json.loads(settings_file.read_text())
        except (json.JSONDecodeError, OSError):
            file_values = {}

    defaults = Settings()
    kwargs: dict[str, Any] = {"data_dir": dd}
    for f in fields(Settings):
        if f.name == "data_dir":
            continue
        default = getattr(defaults, f.name)
        value: Any = None
        if f.name in file_values and file_values[f.name] not in (None, ""):
            value = file_values[f.name]
        else:
            env_name = ENV_MAP.get(f.name, "")
            env_val = os.environ.get(env_name, "") if env_name else ""
            if env_val != "":
                value = env_val
        if value is not None:
            kwargs[f.name] = _coerce(value, default)
    return Settings(**kwargs)


def save_settings(settings_file: Path, updates: dict[str, Any]) -> dict[str, Any]:
    """Merge ``updates`` into settings.json atomically with mode 0600."""
    unknown = set(updates) - UI_EDITABLE
    if unknown:
        raise ValueError(f"not settable via UI: {sorted(unknown)}")
    current: dict[str, Any] = {}
    if settings_file.exists():
        try:
            current = json.loads(settings_file.read_text())
        except (json.JSONDecodeError, OSError):
            current = {}
    current.update(updates)

    settings_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = settings_file.with_name(settings_file.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(current, fh, indent=2)
    os.replace(tmp, settings_file)
    os.chmod(settings_file, 0o600)
    return current


def mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "•" * 8
    return "•" * 8 + value[-4:]
