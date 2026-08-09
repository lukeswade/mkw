from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app.web.markdown import render

router = APIRouter()
_ROOT = Path(__file__).parent.parent.parent

@router.get("/readme", response_class=HTMLResponse)
def get_readme(request: Request):
    readme_path = _ROOT / "README.md"
    content = ""
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
    
    html_content = render(content)
    
    return request.app.state.templates.TemplateResponse(
        request, "readme.html",
        {"nav": "readme", "content": html_content}
    )
