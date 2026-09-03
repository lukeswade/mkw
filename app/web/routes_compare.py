"""Two runs side by side: the numbers, the queries, what each kept, the overviews."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.research.compare import side_by_side, summarize
from app.web.markdown import render_overview, strip_leading_h1

router = APIRouter()


@router.get("/compare")
async def compare(request: Request, a: str = "", b: str = ""):
    repo = request.app.state.repo
    cfg = request.app.state.cfg_loader()
    if not a or not b:
        raise HTTPException(400, "compare needs ?a=RUN_ID&b=RUN_ID")
    try:
        sa, sb = summarize(repo, cfg.research_dir, a), summarize(repo, cfg.research_dir, b)
    except KeyError as e:
        raise HTTPException(404, f"no such run: {e}") from e

    def overview(run_id: str) -> str:
        row = repo.get_run(run_id)
        path = cfg.research_dir / row["dir"] / "overview.md"
        md = path.read_text(encoding="utf-8") if path.is_file() else ""
        return render_overview(strip_leading_h1(md), len(repo.findings_for_run(run_id)))

    return request.app.state.templates.TemplateResponse(request, "compare.html", {
        "nav": "library", "a": sa, "b": sb, "rows": side_by_side(sa, sb),
        "overview_a": overview(a), "overview_b": overview(b),
    })
