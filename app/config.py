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
    "deepseek_api_key": "DEEPSEEK_API_KEY",
    "deepseek_base_url": "DEEPSEEK_BASE_URL",
    "deepseek_model": "DEEPSEEK_MODEL",
    "local_llm_base_url": "LOCAL_LLM_BASE_URL",
    "local_llm_model": "LOCAL_LLM_MODEL",
    "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
    "telegram_allowed_user_ids": "TELEGRAM_ALLOWED_USER_IDS",
    "web_password": "WEB_PASSWORD",
    "searxng_url": "SEARXNG_URL",
    "data_dir": "DATA_DIR",
    "fetch_concurrency": "FETCH_CONCURRENCY",
    "llm_concurrency": "LLM_CONCURRENCY",
    "llm_timeout": "LLM_TIMEOUT",
    "results_per_query": "RESULTS_PER_QUERY",
    "respect_robots": "RESPECT_ROBOTS",
    "allow_private_fetch": "ALLOW_PRIVATE_FETCH",
    "user_agent": "USER_AGENT",
}

# Fields the web Settings page is allowed to persist into settings.json.
UI_EDITABLE = {
    "llm_provider",
    "deepseek_api_key",
    "deepseek_base_url",
    "deepseek_model",
    "local_llm_base_url",
    "local_llm_model",
    "telegram_bot_token",
    "telegram_allowed_user_ids",
    "web_password",
    "searxng_url",
    "results_per_query",
    "respect_robots",
}

SECRET_FIELDS = {"deepseek_api_key", "telegram_bot_token", "web_password"}


@dataclass
class Settings:
    llm_provider: str = "deepseek"  # "deepseek" | "local"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    local_llm_base_url: str = "http://host.docker.internal:8080/v1"
    local_llm_model: str = "local"
    telegram_bot_token: str = ""
    telegram_allowed_user_ids: str = ""
    web_password: str = ""
    searxng_url: str = "http://searxng:8080"
    data_dir: str = "./data"
    fetch_concurrency: int = 8
    llm_concurrency: int = 4
    llm_timeout: int = 180
    results_per_query: int = 8
    respect_robots: bool = True
    allow_private_fetch: bool = False
    user_agent: str = "deep-research/0.1 (personal research agent)"

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
