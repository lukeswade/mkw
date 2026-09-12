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
    case "triage": {
      const doms = (e.domains || []).length ? ` (${e.domains.join(", ")})` : "";
      const spared = e.spared ? ` · ${e.spared} spared` : "";
      return `[${t}]   triage: ${e.considered} candidates → ${e.to_read} to read · ${e.dropped} dropped${doms}${spared}`;
    }
    case "source_skipped": {
      // one line per triage round is the "triage" event above; the per-page
      // drops of older runs are not worth a line each
      if (String(e.reason || "").startsWith("dropped at triage")) return null;
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
  // A click lights its own entry at once rather than after the scroll settles;
  // on a narrow screen it also closes the list it came from.
  links.forEach(function (a) {
    a.addEventListener('click', function () {
      current = a.getAttribute('href').slice(1);
      links.forEach(function (b) { b.classList.toggle('current', b === a); });
      if (!wide.matches) toc.removeAttribute('open');
    });
  });
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

// ---- Run actions: a menu on phones, inline everywhere else ----
(function () {
  var wide = window.matchMedia('(min-width: 641px)');
  function apply() {
    document.querySelectorAll('.actions-menu').forEach(function (d) {
      if (wide.matches) d.setAttribute('open', ''); else d.removeAttribute('open');
    });
  }
  apply();
  wide.addEventListener('change', apply);
  // The evergreen toggle swaps the whole header back in, open.
  document.body.addEventListener('htmx:afterSwap', apply);
})();

// ---- Run lists: select-to-delete everywhere, swipe-to-delete on touch ----
(function () {
  function region(el) { return el.closest('.runs-region'); }
  function refresh(reg) {
    var n = reg.querySelectorAll('.sel input:checked').length;
    var b = reg.querySelector('.delete-selected');
    b.disabled = n === 0;
    b.textContent = n ? 'Delete selected (' + n + ')' : 'Delete selected';
    reg.querySelectorAll('.run-item-wrap').forEach(function (w) {
      var c = w.querySelector('.sel input');
      w.classList.toggle('checked', !!(c && c.checked));
    });
  }
  function setSelecting(reg, on) {
    reg.classList.toggle('selecting', on);
    // The New tab re-fetches its list every few seconds; not while it is
    // being worked on, or the checkboxes vanish under the user's finger.
    document.body.classList.toggle('list-busy', on || !!document.querySelector('.run-row.open'));
    if (!on) reg.querySelectorAll('.sel input').forEach(function (c) { c.checked = false; });
    refresh(reg);
  }
  document.addEventListener('click', function (e) {
    var t = e.target;
    if (t.closest('.select-toggle')) { setSelecting(region(t), true); return; }
    if (t.closest('.select-cancel')) { setSelecting(region(t), false); return; }
    var reg = region(t);
    if (reg && reg.classList.contains('selecting')) {
      var wrap = t.closest('.run-item-wrap');
      if (wrap && !t.closest('.evergreen-star')) {
        // In select mode a tap anywhere on the row toggles it; the link waits.
        if (!t.closest('.sel input')) {
          e.preventDefault();
          var c = wrap.querySelector('.sel input');
          c.checked = !c.checked;
        }
        refresh(reg);
        return;
      }
    }
    // The row is a div now (the title is the link, so the evergreen star can
    // live inside the badge row): a click anywhere else on it navigates,
    // unless it landed on a control or a row is swiped open.
    var item = t.closest('.run-item');
    if (item && item.dataset.href && !t.closest('a, button, input, label')
        && !document.querySelector('.run-row.open')) {
      window.location.href = item.dataset.href;
      return;
    }
    // Tapping the revealed Delete deletes that one row through the same
    // endpoint the batch uses; tapping anywhere else closes an open row.
    var del = t.closest('.swipe-delete');
    if (del) {
      var reg2 = region(del), form = reg2.querySelector('form');
      htmx.ajax('POST', '/runs/batch-delete', {
        target: reg2, swap: 'outerHTML',
        values: { ids: del.dataset.id, view: form.view.value, kind: form.kind.value }
      });
      return;
    }
    var open = document.querySelector('.run-row.open');
    if (open && !open.contains(t)) { open.classList.remove('open'); document.body.classList.remove('list-busy'); }
  });
  document.body.addEventListener('htmx:afterSwap', function () {
    if (!document.querySelector('.runs-region.selecting') && !document.querySelector('.run-row.open')) {
      document.body.classList.remove('list-busy');
    }
  });

  // Swipe: horizontal drag on a row reveals Delete behind it. Touch only;
  // touch-action:pan-y in CSS leaves vertical scrolling to the browser.
  if (!window.matchMedia('(pointer: coarse)').matches) return;
  var drag = null;
  document.addEventListener('pointerdown', function (e) {
    var wrap = e.target.closest('.run-item-wrap');
    if (!wrap || e.pointerType === 'mouse' || region(wrap).classList.contains('selecting')) return;
    drag = { wrap: wrap, row: wrap.closest('.run-row'), x: e.clientX, y: e.clientY, dx: 0, live: false };
  }, { passive: true });
  document.addEventListener('pointermove', function (e) {
    if (!drag) return;
    var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (!drag.live) {
      if (Math.abs(dx) < 10 || Math.abs(dx) < Math.abs(dy)) return;
      drag.live = true;
      drag.wrap.style.transition = 'none';
    }
    var wasOpen = drag.row.classList.contains('open');
    drag.dx = Math.max(-96, Math.min(0, dx + (wasOpen ? -88 : 0)));
    drag.wrap.style.transform = 'translateX(' + drag.dx + 'px)';
  }, { passive: true });
  function end() {
    if (!drag) return;
    var d = drag; drag = null;
    d.wrap.style.transition = ''; d.wrap.style.transform = '';
    if (!d.live) return;
    var open = d.dx < -50;
    document.querySelectorAll('.run-row.open').forEach(function (r) { if (r !== d.row) r.classList.remove('open'); });
    d.row.classList.toggle('open', open);
    document.body.classList.toggle('list-busy', open || !!document.querySelector('.runs-region.selecting'));
    // A drag must not also be a tap on the link underneath.
    var swallow = function (ev) { ev.preventDefault(); ev.stopPropagation(); };
    d.wrap.addEventListener('click', swallow, { capture: true, once: true });
    setTimeout(function () { d.wrap.removeEventListener('click', swallow, { capture: true }); }, 300);
  }
  document.addEventListener('pointerup', end, { passive: true });
  document.addEventListener('pointercancel', end, { passive: true });
})();
