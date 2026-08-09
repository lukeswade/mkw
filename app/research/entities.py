"""Entity extraction for the knowledge graph (LLM + SQLite only — works even
when the vector layer is unavailable)."""
from __future__ import annotations

import logging
import re
import unicodedata

from app.db import Repo
from app.llm import prompts
from app.llm.json_utils import LLMJsonError
from app.models import EntitiesOut

log = logging.getLogger(__name__)


def normalize_name(name: str) -> str:
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    n = re.sub(r"[^\w\s]", "", n).casefold()
    return re.sub(r"\s+", " ", n).strip()


async def extract_entities(llm, repo: Repo, run_id: str, overview: str) -> int:
    prompt = prompts.ENTITIES.format(overview=overview[:30_000])
    try:
        out = await llm.chat_json(
            "entities", [{"role": "user", "content": prompt}],
            EntitiesOut, max_tokens=1500, temperature=0.1)
    except LLMJsonError as e:
        log.warning("entity extraction skipped: %s", e)
        return 0
    stored = 0
    for ent in out.entities[:15]:
        norm = normalize_name(ent.name)
        if not norm or len(norm) < 2:
            continue
        entity_id = repo.upsert_entity(ent.name.strip(), norm, ent.type,
                                       ent.description)
        repo.set_run_entity(run_id, entity_id, ent.salience)
        stored += 1
    return stored
