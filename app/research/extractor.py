"""Fetched bytes → clean text + metadata (trafilatura for HTML, pymupdf for PDF)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import trafilatura

from app.research.fetcher import Fetched

log = logging.getLogger(__name__)

MAX_TEXT_CHARS = 80_000
MAX_PDF_PAGES = 40
MIN_TEXT_CHARS = 200

_DATE_RE = re.compile(r"^(\d{4})-?(\d{2})-?(\d{2})")


@dataclass
class Extracted:
    text: str
    title: str | None
    date: str | None  # YYYY-MM-DD


def _decode(fetched: Fetched) -> str:
    charset = "utf-8"
    return fetched.body.decode(charset, errors="replace")


def _clean_date(value: str | None) -> str | None:
    if not value:
        return None
    m = _DATE_RE.match(str(value).strip().removeprefix("D:"))
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def _extract_html(fetched: Fetched) -> Extracted | None:
    try:
        doc = trafilatura.bare_extraction(
            _decode(fetched), url=fetched.final_url, with_metadata=True,
            include_comments=False,
        )
    except Exception:  # trafilatura chokes on odd markup sometimes
        log.debug("trafilatura failed on %s", fetched.final_url, exc_info=True)
        return None
    if doc is None:
        return None
    text = (getattr(doc, "text", None) or "").strip()
    if len(text) < MIN_TEXT_CHARS:
        return None
    return Extracted(
        text=text[:MAX_TEXT_CHARS],
        title=(getattr(doc, "title", None) or "").strip() or None,
        date=_clean_date(getattr(doc, "date", None)),
    )


def _extract_pdf(fetched: Fetched) -> Extracted | None:
    try:
        import pymupdf
        with pymupdf.open(stream=fetched.body, filetype="pdf") as doc:
            pages = [page.get_text() for page in doc.pages(0, min(doc.page_count, MAX_PDF_PAGES))]
            meta = doc.metadata or {}
    except Exception:
        log.debug("pdf extraction failed on %s", fetched.final_url, exc_info=True)
        return None
    text = "\n".join(pages).strip()
    if len(text) < MIN_TEXT_CHARS:
        return None
    return Extracted(
        text=text[:MAX_TEXT_CHARS],
        title=(meta.get("title") or "").strip() or None,
        date=_clean_date(meta.get("creationDate")),
    )


def extract(fetched: Fetched) -> Extracted | None:
    if fetched.content_type == "application/pdf":
        return _extract_pdf(fetched)
    return _extract_html(fetched)
