"""Ask: RAG question-answering over the accumulated research corpus."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Form, Request

from app.web.markdown import render

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/ask")
async def ask_page(request: Request):
    return request.app.state.templates.TemplateResponse(
        request, "ask.html",
        {"nav": "ask", "rag_available": request.app.state.rag is not None})


@router.post("/ask")
async def ask_submit(request: Request, question: str = Form(...)):
    rag = request.app.state.rag
    templates = request.app.state.templates
    if rag is None:
        return templates.TemplateResponse(
            request, "partials/ask_answer.html",
            {"error": "Knowledge layer is not available in this build.",
             "answer_html": None, "sources": [], "question": question})
    try:
        result = await rag.ask(question, request.app.state.repo)
    except Exception as e:
        log.exception("ask failed")
        return templates.TemplateResponse(
            request, "partials/ask_answer.html",
            {"error": f"Ask failed: {e}", "answer_html": None, "sources": [],
             "question": question})
    return templates.TemplateResponse(
        request, "partials/ask_answer.html",
        {"error": None, "answer_html": render(result["answer"]),
         "sources": result["sources"], "question": question})
