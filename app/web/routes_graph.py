"""Knowledge graph: runs ↔ entities force-graph, plus run↔run similarity."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/graph")
async def graph_page(request: Request):
    return request.app.state.templates.TemplateResponse(
        request, "graph.html",
        {"nav": "graph", "rag_available": request.app.state.rag is not None})


@router.get("/api/graph")
async def graph_data(request: Request):
    repo = request.app.state.repo
    runs, entities, links = repo.graph_rows()

    nodes = []
    node_ids = set()
    for r in runs:
        nid = f"run:{r['id']}"
        node_ids.add(nid)
        nodes.append({
            "id": nid, "label": r["title"] or r["query"], "group": "run",
            "val": max(3, min(14, 3 + (r["n_findings"] or 0) // 2)),
            "url": f"/runs/{r['id']}",
        })

    edges = []
    ent_meta: dict[int, dict] = {}
    for e in entities:  # one row per (entity, run) pair
        ent_id = f"ent:{e['id']}"
        if e["id"] not in ent_meta:
            ent_meta[e["id"]] = {"id": ent_id, "label": e["name"],
                                 "group": e["type"], "val": 2,
                                 "description": e["description"] or "",
                                 "salience": float(e["salience"] or 0)}
        else:
            ent_meta[e["id"]]["val"] += 1
            ent_meta[e["id"]]["salience"] = max(
                ent_meta[e["id"]]["salience"], float(e["salience"] or 0))
        if f"run:{e['run_id']}" in node_ids:
            edges.append({"source": f"run:{e['run_id']}", "target": ent_id,
                          "value": float(e["salience"] or 0.3), "kind": "mentions"})

    nodes.extend(ent_meta.values())
    node_ids.update(m["id"] for m in ent_meta.values())

    for l in links:
        a, b = f"run:{l['src_run_id']}", f"run:{l['dst_run_id']}"
        if a in node_ids and b in node_ids:
            edges.append({"source": a, "target": b,
                          "value": float(l["score"] or 0.5), "kind": l["kind"]})

    return {"nodes": nodes, "links": edges}
