/* Deep Research client JS: tabs, live SSE progress, knowledge graph. */
"use strict";

document.addEventListener("DOMContentLoaded", () => {
  // tab switching (run page)
  document.querySelectorAll(".tabs [data-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tabs [data-tab]").forEach((b) =>
        b.classList.toggle("active", b === btn));
      document.querySelectorAll(".tab-panel").forEach((p) =>
        p.classList.toggle("active", p.id === "tab-" + btn.dataset.tab));
    });
  });

  // clicking a citation link switches to the overview tab anchor smoothly
  const live = document.getElementById("live-log");
  if (live) initProgress(live);
});

function fmtEvent(e) {
  const t = new Date((e.ts || 0) * 1000).toLocaleTimeString([], { hour12: false });
  const list = (xs) => (xs || []).map((q) => `          · ${q}`).join("\n");
  switch (e.type) {
    case "status": return `[${t}] status: ${e.status}`;
    case "phase": return `[${t}] — ${e.phase} —`;
    case "plan": return `[${t}] plan: ${e.title}\n` + list(e.subqueries);
    case "round_start": return `[${t}] ROUND ${e.round}/${e.depth}\n` + list(e.queries);
    case "searched": return `[${t}]   ${e.results} results → ${e.candidates} new candidates`;
    case "source_skipped": return `[${t}]   ✗ ${e.url}  (${e.reason})`;
    case "finding": return `[${t}]   ✓ [${e.idx}] ${e.title} (${e.domain}, ${e.relevance}/10)`;
    case "gap": return `[${t}]   gap: saturated=${e.saturated}, next queries=${(e.next_queries || []).length}`;
    case "log": return `[${t}]   · ${e.message}`;
    case "error": return `[${t}] ERROR: ${e.message}`;
    case "done": return `[${t}] DONE: ${e.status} (${e.stop_reason || ""})`;
    default: return null;
  }
}

function initProgress(el) {
  const runId = el.dataset.run;
  const es = new EventSource(`/runs/${encodeURIComponent(runId)}/events`);
  es.onmessage = (msg) => {
    let e;
    try { e = JSON.parse(msg.data); } catch { return; }
    const line = fmtEvent(e);
    if (line) {
      el.textContent += line + "\n";
      el.scrollTop = el.scrollHeight;
    }
    if (e.type === "status") {
      const chip = document.getElementById("status-chip");
      if (chip) { chip.textContent = e.status; chip.className = "chip status-" + e.status; }
    }
    if (e.type === "done") {
      es.close();
      setTimeout(() => location.reload(), 700);
    }
  };
  // on error the browser auto-reconnects and replays via Last-Event-ID
}

/* ---- knowledge graph page ---- */
window.initGraph = async function initGraph() {
  const box = document.getElementById("graph");
  if (!box || typeof ForceGraph === "undefined") return;
  const data = await fetch("/api/graph").then((r) => r.json());
  const stats = document.getElementById("g-stats");
  if (stats) {
    const runs = data.nodes.filter((n) => n.group === "run").length;
    stats.textContent = `${runs} runs · ${data.nodes.length - runs} entities · ${data.links.length} links`;
  }

  const COLORS = {
    run: "#6ee7b7", person: "#79c0ff", org: "#ffa657", technology: "#d2a8ff",
    concept: "#f2cc60", place: "#7ee787", event: "#ff7b72", product: "#a5d6ff",
    other: "#8b949e",
  };

  const graph = new ForceGraph(box)
    .graphData(data)
    .width(box.clientWidth)
    .height(box.clientHeight)
    .backgroundColor("#090c10")
    .nodeId("id")
    .nodeVal("val")
    .nodeLabel((n) => n.group === "run" ? `research: ${n.label}`
                                        : `${n.group}: ${n.label}\n${n.description || ""}`)
    .nodeColor((n) => COLORS[n.group] || COLORS.other)
    .linkColor((l) => (l.kind === "similar" ? "#3fb95055" : l.kind === "followup" ? "#79c0ff55" : "#30363d"))
    .linkWidth((l) => (l.kind === "mentions" ? 0.6 : 1.6))
    .linkLineDash((l) => (l.kind === "similar" ? [3, 3] : null))
    .onNodeClick((n) => { if (n.url) window.location = n.url; })
    .nodeCanvasObjectMode(() => "after")
    .nodeCanvasObject((n, ctx, scale) => {
      if (n.group !== "run" && scale < 1.2) return; // label runs always, entities when zoomed
      const label = n.label.length > 34 ? n.label.slice(0, 32) + "…" : n.label;
      ctx.font = `${Math.max(10 / scale, 2.4)}px sans-serif`;
      ctx.textAlign = "center";
      ctx.fillStyle = "#c9d1d9";
      ctx.fillText(label, n.x, n.y + 7 + (n.val || 3) / 2);
    });

  const applyFilters = () => {
    const minSal = parseFloat(document.getElementById("g-salience").value || "0");
    const runsOnly = document.getElementById("g-runs-only").checked;
    const keepNode = (n) =>
      n.group === "run" || (!runsOnly && (n.salience === undefined || n.salience >= minSal));
    const nodes = data.nodes.filter(keepNode);
    const ids = new Set(nodes.map((n) => n.id));
    const links = data.links.filter((l) => {
      const s = typeof l.source === "object" ? l.source.id : l.source;
      const t = typeof l.target === "object" ? l.target.id : l.target;
      return ids.has(s) && ids.has(t);
    });
    graph.graphData({ nodes, links });
  };
  document.getElementById("g-salience").addEventListener("input", applyFilters);
  document.getElementById("g-runs-only").addEventListener("change", applyFilters);
};
