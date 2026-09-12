/* Deep Research client JS: tabs and live SSE progress. */
"use strict";

document.addEventListener("DOMContentLoaded", () => {
  // tab switching (run page)
  // A citation on the Comparison tab points at #src-N, which lives in the
  // Overview panel — jumping there while that panel is display:none looks
  // like a dead link. Switch tabs first, then let the anchor resolve.
  document.addEventListener("click", (e) => {
    const link = e.target.closest('a[href^="#src-"]');
    if (!link) return;
    const panel = document.getElementById("tab-overview");
    if (!panel || panel.classList.contains("active")) return;
    const btn = document.querySelector('.tabs [data-tab="overview"]');
    if (btn) btn.click();
  });

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
    case "round_start": return `[${t}] ROUND ${e.round}/${e.depth}\n` + (e.queries || []).map((q, i) => `          · ${q}${(e.scopes || [])[i] ? `  [${e.scopes[i]}]` : ""}`).join("\n");
    case "searched": return `[${t}]   ${e.results} results → ${e.candidates} new candidates`;
    case "source_skipped": {
      // a page that was read and scored ends like a kept one: URL, title, score last
      const quoted = e.title ? `  "${String(e.title).slice(0, 70)}"` : "";
      const m = /^relevance (\d+)\/10$/.exec(e.reason || "");
      return m ? `[${t}]   ✗ ${e.url}${quoted} · ${m[1]}/10` : `[${t}]   ✗ ${e.url}  (${e.reason})${quoted}`;
    }
    case "finding": return `[${t}]   ✓ [${e.idx}] ${e.title} (${e.domain}) · ${e.relevance}/10`;
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

// ---- On this page: collapse on narrow screens, follow the reader on wide ----
(function () {
  var toc = document.querySelector('.toc');
  if (!toc) return;
  var wide = window.matchMedia('(min-width: 1100px)');
  if (!wide.matches) toc.removeAttribute('open');
  var links = Array.prototype.slice.call(toc.querySelectorAll('a[href^="#"]'));
  var byId = {};
  links.forEach(function (a) { byId[a.getAttribute('href').slice(1)] = a; });
  var heads = Object.keys(byId).map(function (id) { return document.getElementById(id); }).filter(Boolean);
  if (!heads.length) return;
  var current = null, queued = false;
  // The section whose heading most recently passed the top third of the
  // viewport is the one being read. A scroll listener, not an observer: an
  // observer only fires when a heading crosses its band, so a long jump
  // that lands between headings left the old entry lit.
  function update() {
    queued = false;
    var y = window.scrollY + window.innerHeight / 3, best = heads[0];
    heads.forEach(function (h) { if (h.getBoundingClientRect().top + window.scrollY <= y) best = h; });
    if (current === best.id) return;
    current = best.id;
    links.forEach(function (a) { a.classList.toggle('current', a.getAttribute('href') === '#' + best.id); });
  }
  window.addEventListener('scroll', function () {
    if (!queued) { queued = true; requestAnimationFrame(update); }
  }, { passive: true });
  update();
  // On a narrow screen a tap on an entry should close the list it came from.
  links.forEach(function (a) { a.addEventListener('click', function () { if (!wide.matches) toc.removeAttribute('open'); }); });
})();

// ---- Copy as Markdown ----
document.addEventListener('click', function (e) {
  var b = e.target.closest('.copy-md');
  if (!b) return;
  fetch(b.dataset.src).then(function (r) { return r.text(); }).then(function (t) {
    return navigator.clipboard.writeText(t);
  }).then(function () {
    var was = b.textContent; b.textContent = 'Copied'; b.classList.add('done');
    setTimeout(function () { b.textContent = was; b.classList.remove('done'); }, 1600);
  }).catch(function () { b.textContent = 'Copy failed'; });
});
