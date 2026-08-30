"""Paste a document, get its claims checked.

The only feature with its own input: everything else starts from a question.
The result is an ordinary run, so it lands in the library and inherits
exports, Ask indexing and the progress stream.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse
from pydantic import ValidationError

from app.models import VERIFY_CLAIM_CAP, RunParams

log = logging.getLogger(__name__)
router = APIRouter()

_TITLE_CHARS = 90


def _tpl(request: Request):
    return request.app.state.templates


def _title_for(document: str) -> str:
    """A run needs a name; use the document's first real line."""
    for line in document.splitlines():
        line = line.strip().lstrip("#").strip()
        if len(line) > 12:
            return f"Claim check: {line[:_TITLE_CHARS]}"
    return "Claim check: pasted document"


@router.get("/verify")
async def verify_page(request: Request):
    return _tpl(request).TemplateResponse(
        request, "verify.html", {"nav": "verify", "cap": VERIFY_CLAIM_CAP})


@router.post("/verify")
async def start_verification(request: Request, document: str = Form("")):
    document = document.strip()
    ctx = {"nav": "verify", "cap": VERIFY_CLAIM_CAP, "prefill": document}
    if len(document) < 40:
        ctx["error"] = "Paste a document to check — at least a paragraph."
        return _tpl(request).TemplateResponse(request, "verify.html", ctx,
                                              status_code=422)
    try:
        params = RunParams(query=_title_for(document), depth=0, recency="all",
                           origin="web", kind="verify", document=document,
                           created_by=_initiator(request))
    except ValidationError as e:
        ctx["error"] = "; ".join(err["msg"] for err in e.errors())
        return _tpl(request).TemplateResponse(request, "verify.html", ctx,
                                              status_code=422)
    run_id = request.app.state.orch.enqueue(params)
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


def _initiator(request: Request) -> str:
    from app.web.routes_runs import _initiator as who
    return who(request)
