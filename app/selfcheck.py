"""Fail-fast startup checks.

Both of the bugs that took this app down in production were things a single
cheap check at boot would have caught: a prompt template that raised KeyError
the first time a document was processed, and a migration that crashed on any
fresh database. Neither surfaced until minutes (or a deploy) later.
"""
from __future__ import annotations

import logging
import string

from app.config import Settings
from app.llm import prompts

log = logging.getLogger(__name__)


class SelfCheckError(RuntimeError):
    pass


def check_prompt_templates() -> None:
    """Every template must render with plain identifier placeholders.

    An unescaped '{' in an embedded JSON example turns into a replacement
    field, and str.format() then raises on the first real call.
    """
    for name, template in vars(prompts).items():
        if not (name.isupper() and isinstance(template, str)):
            continue
        fields = {f for _l, f, _s, _c in string.Formatter().parse(template)
                  if f is not None}
        bad = [f for f in fields if not f.isidentifier()]
        if bad:
            raise SelfCheckError(
                f"prompt template {name} has non-identifier placeholder(s) "
                f"{bad!r} — almost certainly an unescaped '{{' in a JSON "
                f"example. Double the braces."
            )
        template.format(**{f: "" for f in fields})


def check_database(cfg: Settings) -> None:
    """Opening the DB runs migrations; a fresh volume must survive that."""
    from app.db import connect
    conn = connect(cfg.db_path)
    conn.execute("SELECT COUNT(*) FROM runs").fetchone()
    conn.close()


def run_all(cfg: Settings) -> None:
    check_prompt_templates()
    check_database(cfg)
    log.info("self-check passed")
