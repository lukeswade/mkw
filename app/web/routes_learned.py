"""What this install has learned from its own runs — and where it stands.

Dead and productive domains come from every page ever read; engine health
from the refusals every run records; the Brave count from the searches each
run made; the calibration from the estimate's own history. All of it existed
as data; none of it had a home.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Request

from app.billing_cycle import cycle_start
from app.config import load_settings
from app.research.estimate import estimate_run
from app.research.pipeline import authority_domains_from

router = APIRouter()


def _tpl(request: Request):
    return request.app.state.templates


@router.get("/learned")
async def learned(request: Request):
    repo = request.app.state.repo
    now = datetime.now(timezone.utc)
    since_14d = (now - timedelta(days=14)).isoformat()
    cfg = load_settings()
    cycle = cycle_start(getattr(cfg, "brave_cycle_day", 1), now)
    month_start = cycle.isoformat()

    recent = repo.stats_since(since_14d)
    refused: Counter = Counter()
    for d in recent:
        for engine in (d.get("blocked_engines") or {}):
            refused[engine] += 1
    yields = repo.engine_yields(since_14d, min_reads=1)
    names = set(refused) | set(yields)
    engines = sorted(
        ({"engine": e, "runs_refused": refused.get(e, 0), "share": refused.get(e, 0) / max(1, len(recent)),
          "yield": yields.get(e)} for e in names),
        key=lambda x: (-(x["yield"] or 0), -x["runs_refused"]))[:24]

    # One Brave request per search on any run whose categories include general
    # (or the default, which does). The braveapi engine sits in general.
    brave = repo.brave_requests_since(month_start)
    brave_quota = int(getattr(cfg, "brave_monthly_quota", 0) or 0)

    calibration = []
    for depth in range(1, 11):
        try:
            est = estimate_run(repo, depth)
        except Exception:
            continue
        calibration.append({
            "depth": depth,
            "samples": getattr(est, "depth_samples", 0),
            "calibrated": bool(getattr(est, "depth_calibrated", False)),
            "keep_rate": getattr(est, "keep_rate", None),
            "duration": getattr(est, "duration_label", ""),
            "sources": getattr(est, "sources_label", "") or getattr(est, "sources", ""),
        })

    authority = sorted(authority_domains_from(getattr(load_settings(), "authority_sites", "")))
    ay = {r["domain"]: r for r in repo.yield_for_domains(authority)}

    return _tpl(request).TemplateResponse(request, "learned.html", {
        "nav": "learned",
        "authority": [{"domain": d, "reads": ay[d]["reads"] if d in ay else 0,
                       "kept": ay[d]["kept"] if d in ay else 0, "runs": ay[d]["runs"] if d in ay else 0}
                      for d in authority],
        "dead": repo.dead_domains(),
        "productive": repo.productive_domains(),
        "engines": engines, "recent_runs": len(recent),
        "brave_requests": brave, "brave_quota": brave_quota,
        "month": f"cycle since {cycle.strftime('%b %-d')}",
        "calibration": calibration,
    })
