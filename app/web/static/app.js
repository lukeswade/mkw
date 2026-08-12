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
        // reveal on the first token — for a research run the pane is hidden
        // until synthesis actually starts
        const pane = document.getElementById("stream-pane");
        if (pane && pane.hidden) pane.hidden = false;
        chatOut.textContent += e.chunk;
        chatOut.scrollTop = chatOut.scrollHeight;
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

/* ---- Settings: fill defaults when the provider preset changes ---- */
document.addEventListener("DOMContentLoaded", () => {
  const select = document.getElementById("llm-provider");
  if (!select) return;
  const baseUrl = document.getElementById("llm-base-url");
  const model = document.getElementById("llm-model");
  const apiKey = document.getElementById("llm-api-key");
  const hint = document.getElementById("provider-hint");
  // remember what the preset last filled in, so a value the user typed is
  // never silently overwritten
  let autofilled = { base: baseUrl.value, model: model.value };

  select.addEventListener("change", () => {
    const opt = select.selectedOptions[0];
    if (!baseUrl.value || baseUrl.value === autofilled.base) {
      baseUrl.value = opt.dataset.baseUrl || "";
    }
    if (!model.value || model.value === autofilled.model) {
      model.value = opt.dataset.model || "";
    }
    autofilled = { base: opt.dataset.baseUrl || "", model: opt.dataset.model || "" };
    if (hint) hint.textContent = opt.dataset.hint || "";
    const needsKey = opt.dataset.needsKey === "1";
    apiKey.placeholder = needsKey
      ? "required — blank keeps the current one"
      : "not needed for local servers";
  });
});
