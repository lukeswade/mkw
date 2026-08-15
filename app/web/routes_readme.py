import re
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app.web.markdown import render

router = APIRouter()
_ROOT = Path(__file__).parent.parent.parent

# GitHub-facing decoration that breaks in-app: the CI/shields badges load from
# external hosts (dead offline or behind blockers), and the screenshots are
# relative docs/ paths that aren't shipped in the image at all — you're already
# looking at the app, it doesn't need pictures of itself. Strip linked images
# first so no orphan [](...) shell survives, then bare images.
_LINKED_IMAGE_RE = re.compile(r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# The screenshot gallery is pure captions-plus-images; with the images gone
# the captions narrate pictures that aren't there, so the section goes too.
_GALLERY_RE = re.compile(r"\n## What it looks like\n.*?\n---\n", re.DOTALL)


def _strip_images(md: str) -> str:
    md = _GALLERY_RE.sub("\n---\n", md)
    md = _LINKED_IMAGE_RE.sub("", md)
    md = _IMAGE_RE.sub("", md)
    return re.sub(r"\n{3,}", "\n\n", md)


@router.get("/readme", response_class=HTMLResponse)
def get_readme(request: Request):
    readme_path = _ROOT / "README.md"
    content = ""
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")

    html_content = render(_strip_images(content))

    return request.app.state.templates.TemplateResponse(
        request, "readme.html",
        {"nav": "readme", "content": html_content}
    )
