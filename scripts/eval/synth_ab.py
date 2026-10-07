"""A/B: synthesis with thinking off vs on, from one run's stored sources.

    docker compose exec -T app python - <run_id> [off,on,off,on] [on_budget] \
        < scripts/eval/synth_ab.py > synth.jsonl

One JSON line per sample on stdout: timings, per-call tokens, the reasoning
text and the finished overview (post citation-validation and gap marking,
as a re-synthesis would publish it). Progress goes to stderr.

Read-only by construction: the DB is opened mode=ro, the event bus is a list,
and nothing is written to the run's directory, the FTS index or Chroma. The
candidate axis is computed once (thinking off, as production does) and shared
by every sample, so the synth calls are the only thing that differs.

The thinking arm gets a 30k output budget: reasoning is billed against
max_tokens, and at the production 8k the Sep-21 run spent it all thinking.
"""
import asyncio, copy, json, sqlite3, sys, time
from collections import Counter

from app.config import load_settings
from app.llm import prompts
from app.llm.client import LLM
from app.research import synthesizer, facets as facet_plan
from app.research.pipeline import Pipeline
from app.research.storage import RunStore, validate_citations

if len(sys.argv) < 2:
    raise SystemExit("usage: python - <run_id> [arms] [on_budget] < synth_ab.py")
RUN = sys.argv[1]
ARMS = (sys.argv[2] if len(sys.argv) > 2 else "off,on,off,on").split(",")
ON_BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 30_000
CALL_CEILING = 1500


class ReadOnlyRepo:
    def __init__(self, path):
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
        self.conn.row_factory = sqlite3.Row

    def get_run(self, rid):
        return self.conn.execute("SELECT * FROM runs WHERE id = ?", (rid,)).fetchone()

    def findings_for_run(self, rid):
        return self.conn.execute(
            "SELECT * FROM findings WHERE run_id = ? ORDER BY idx", (rid,)).fetchall()


class ListBus:
    def __init__(self):
        self.events = []

    def publish(self, run_id, kind, **kw):
        self.events.append({"kind": kind, **{k: str(v)[:300] for k, v in kw.items()}})


def log(msg):
    print(msg, file=sys.stderr, flush=True)


cfg = load_settings()
repo = ReadOnlyRepo(cfg.db_path)
bus = ListBus()
pipe = Pipeline(cfg, repo, bus, rag=None)
row = repo.get_run(RUN)
store = RunStore(cfg.research_dir / row["dir"])
meta = store.read_meta()
findings = pipe._stored_findings(RUN, store)
query = row["query"]
title = meta.get("title") or row["title"] or query[:120]
brief = meta.get("brief") or query
recency_desc = prompts.RECENCY_DESC[row["recency"]]
today = (meta.get("created_at") or "")[:10]
facets = [f for f in (meta.get("facets") or []) if isinstance(f, str)]
premises = [p for p in (meta.get("premises") or []) if isinstance(p, str)]
deliverables = [d for d in (meta.get("deliverables") or []) if isinstance(d, str)]
if isinstance(meta.get("query_facet"), dict) and isinstance(meta.get("facet_kept"), dict):
    facet_of = {str(k): str(v) for k, v in meta["query_facet"].items()}
    unanswered = facet_plan.uncovered(facets, Counter(meta["facet_kept"]))
else:  # a run from before 2026-09-10 has no stored facet plan
    facet_of = {f.query: (facet_plan.facet_for_query(f.query, facets) or "")
                for f in findings if f.query}
    unanswered = []
log(f"{len(findings)} findings, facets={facets}, unanswered={unanswered}, today={today}")

# One candidate axis for every sample, computed the production way.
_real_candidates = synthesizer.name_candidates
_cached = {}


async def _shared_candidates(llm, *, query, findings):
    return copy.deepcopy(_cached["v"])


def make_llm(arm):
    llm = LLM(cfg)
    llm.call_ceiling = CALL_CEILING
    calls = []
    if arm == "on":
        def think_on(kwargs):
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}),
                                    "chat_template_kwargs": {"enable_thinking": True}}
            return kwargs
        llm._no_thinking = think_on
        plain_chat = llm.chat

        async def chat(kind, messages, **kw):
            if kind == "synth":
                kw["max_tokens"] = ON_BUDGET
            return await plain_chat(kind, messages, **kw)
        llm.chat = chat
    completions = llm.client.chat.completions
    plain_create = completions.create

    async def create(**kw):
        t = time.time()
        resp = await plain_create(**kw)
        choice, msg = resp.choices[0], resp.choices[0].message
        extra = getattr(msg, "model_extra", None) or {}
        reasoning = (getattr(msg, "reasoning_content", None) or extra.get("reasoning_content")
                     or extra.get("reasoning") or "")
        u = resp.usage
        cdet = getattr(u, "completion_tokens_details", None)
        pdet = getattr(u, "prompt_tokens_details", None)
        calls.append({
            "secs": round(time.time() - t, 1), "finish": choice.finish_reason,
            "max_tokens": kw.get("max_tokens"),
            "flag": (kw.get("extra_body") or {}).get("chat_template_kwargs"),
            "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
            "reasoning_tokens": getattr(cdet, "reasoning_tokens", None) if cdet else None,
            "cached_tokens": getattr(pdet, "cached_tokens", None) if pdet else None,
            "reasoning_chars": len(reasoning), "content_chars": len(msg.content or ""),
            "think_tag_in_content": "<think>" in (msg.content or ""),
            "reasoning": reasoning})
        log(f"  call {len(calls)}: {calls[-1]['secs']}s finish={choice.finish_reason} "
            f"completion={u.completion_tokens} reasoning_chars={len(reasoning)}")
        return resp
    completions.create = create
    return llm, calls


async def one(arm, i):
    llm, calls = make_llm(arm)
    log(f"sample {i}: thinking {arm}")
    t0 = time.time()
    err = ""
    try:
        text = await asyncio.wait_for(synthesizer.synthesize(
            llm, query=query, title=title, brief=brief, recency_desc=recency_desc,
            today=today, state_md="", findings=findings, bus=None, run_id="",
            previous_overview="", uncovered_facets=unanswered, facet_of=facet_of,
            premises=premises, deliverables=deliverables,
            placeholder_on_failure=False), 2700 if arm == "on" else 1200)
    except Exception as e:
        text, err = "", f"{type(e).__name__}: {e}"
    secs = round(time.time() - t0, 1)
    doc = synthesizer.looks_like_document(text) if text else False
    final, removed, uncited = text, [], None
    if doc:
        final, removed = validate_citations(text, len(findings))
        final, uncited = pipe._mark_honestly(RUN, final, findings, facet_of, unanswered)
    log(f"sample {i} ({arm}) done in {secs}s, document={doc}, err={err!r}")
    return {"sample": i, "arm": arm, "secs": secs, "error": err, "is_document": doc,
            "removed_citations": sorted(removed), "uncited": uncited,
            "usage": llm.usage, "calls": calls, "overview": final}


async def main():
    base = LLM(cfg)
    _cached["v"] = await _real_candidates(base, query=query, findings=findings)
    synthesizer.name_candidates = _shared_candidates
    log(f"candidates computed once: {base.usage}")
    for i, arm in enumerate(ARMS, 1):
        print(json.dumps(await one(arm, i)), flush=True)


asyncio.run(main())
