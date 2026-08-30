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


# Assistant answers open with a compliment far more often than with a topic,
# and that opener became the run's name in the library.
_FILLER = ("you're absolutely right", "you are absolutely right", "great question",
           "that's a great", "certainly", "of course", "sure thing", "happy to help",
           "here's a comprehensive", "here is a comprehensive", "absolutely")


def _title_for(document: str) -> str:
    """A run needs a name. Prefer a real heading over the opening line."""
    lines = [l.strip() for l in document.splitlines()[:40] if l.strip()]
    for line in lines:                       # a markdown heading names the topic
        if line.startswith("#"):
            head = line.lstrip("#").strip()
            if len(head) > 8:
                return f"Claim check: {head[:_TITLE_CHARS]}"
    for line in lines:                       # else the first line with substance
        low = line.lower()
        if len(line) > 12 and not any(low.startswith(f) for f in _FILLER):
            return f"Claim check: {line[:_TITLE_CHARS]}"
    return "Claim check: pasted text"


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
