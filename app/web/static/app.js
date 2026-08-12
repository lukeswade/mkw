/* Deep Research client JS: tabs and live SSE progress. */
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
    
    if (e.type === "stream") {
      const chatOut = document.getElementById("chat-output");
      if (chatOut) {
        chatOut.textContent += e.chunk;
      }
      return;
    }
    
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
