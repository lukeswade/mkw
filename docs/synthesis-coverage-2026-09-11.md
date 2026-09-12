# Synthesis coverage: why deep runs wasted their research, and what fixed it

Handoff note, 2026-09-11. Everything here was measured on this install, not
reasoned about. Commits `593dc90`, `6f197b0`, `173ae41`.

## The problem

A depth-9 run of Matt's kept 66 sources and cited 24 of them. Checking every
completed run on the box showed why:

```
sources  cited   rate   words  w/src  depth
     66     24    36%    1388     21   d9
     35     28    80%    1316     38   d6
     67     36    54%    1326     20   d10
     29     19    66%    1481     51   d4
     23     17    74%    1077     47   d4
     89     39    44%    1304     15   d10
     74     38    51%    2103     28   d10
     75     19    25%    1492     20   d10
```

**The document came out 1000-2100 words whether the run kept 20 sources or
89.** Length was near-constant, so the citation rate was just arithmetic:
a 1,400-word document cannot carry 66 sources. Depth past ~30 sources was
buying pages nobody read, and paying full fetch + notes cost for them.

The map-reduce digest was NOT the cause. That was my first hypothesis and it
was wrong — see the A/B below, where digest and single-call cite the same
number.

## What shipped

1. **`_SINGLE_CALL_BUDGET` 28k -> 100k** (`593dc90`). A timed call measured
   marginal prefill at **402 tok/s**, and `est_tokens` over-counts real tokens
   (est 84,211 came back as 63,498 actual). Single-call synthesis measured
   23% faster than the four-batch digest at identical citation coverage.

   **Corrected the same night:** this note originally said "the model reports
   262,144 positions, so 100k est is under a third of the window". The
   MODEL does; the SERVER did not — the oMLX profile serving it capped
   requests at 65,536 and refused a 66,568-token prompt with a 400
   ("Prompt too long"), no truncation. Matt's run passed at 63,498 real
   tokens by a 2k margin; a 100k-est prompt is 67-75k real and would have
   failed the run after all its research. Now: the budget actually used is
   `single_call_budget(llm)` = min(100k, 1.25 x the served window) with the
   window a setting (`LLM_CONTEXT_TOKENS`, default 65,536) that the server's
   own refusal corrects downward; and a refused prompt is digested and sent
   again instead of failing the run (`PromptTooLong`). Set the setting to
   what the serving profile actually enforces. On this install the profile
   was then raised to 262,144 (KV is ~20 KB/token for this hybrid model, so
   the full window is ~5 GB) and verified with a 79,766-token needle prompt;
   the setting is 262,144 and the budget is back at the 100k ceiling. The
   same restart, with Hot Cache Size given as "8GB" rather than "8", made
   KV prefix reuse real: an identical 7.2k prefix went 10.5s, 1.95s, 1.89s.

2. **Length target derived from source count** (`6f197b0`).
   `target_words(n) = clamp(50 * n, 900, 5000)`, keyed on `len(findings)` —
   deliberately NOT on input tokens, so a live run and a re-synthesis of the
   same research get the same target. `max_out` is derived from the target so
   the ceiling can never be tighter than the length just requested.

3. **`_stored_findings` header strip** (`6f197b0`). The finding `.md` is a
   rendered VIEW of a finding; its header repeats the title, URL, domain and
   tier that `_note_block` emits from `citation_line()`. Feeding the whole
   file made synthesis read every source's identity twice — 13,872 of 79,963
   est tokens on this run. Now takes the `## Notes` body and parses
   `## Key facts` back into shape. Measured 25% faster (176s vs 234s), and it
   makes re-synthesis feed what a live run feeds.

4. **A section per part of the question** (`173ae41`). Synthesis is given the
   parts that have sources, with counts, and told each gets its own heading.

## The measurements

Same 66 stored sources every time, one variable per arm, no re-searching:

```
original run              1,388 words   24 of 66 cited   36%
digest @28k               1,607         31               47%
single call @100k         1,517         30               45%
hedged length target      1,480         33               50%
hedged + header strip     1,592         32               48%
firm length target        2,197         38               58%
+ structural sections     2,398         41               62%
```

**Words per cited source held at 57-58 across every arm.** The document got
73% longer and citation density did not move, which is the evidence that the
extra length carries more research rather than padding.

## Traps, each one hit for real

- **A hedged instruction is an inert instruction.** The first length block
  said "that is a target, not a quota ... if the sources do not carry that
  many words, write the shorter document". Asked for 3,300 words, produced
  1,480 — the model took the escape hatch every time. Removing it moved the
  same run to 2,197 words. The anti-padding guards can stay; the permission
  to write short cannot. This is the house rule about bounding models in code
  rather than prompts, in its inverse form.

- **Structure beats word counts.** The length target alone left the parts with
  two or three sources dissolving into a neighbour. Naming the parts and
  demanding a heading each recovered them.

- **`uncovered_facets` and the structure list can overlap.** `uncovered` comes
  from `facet_kept` (credit-gated); the groups come from `query_facet`
  (attribution). A part can be in both. Naming it in `SYNTH_STRUCTURE_BLOCK`
  while `SYNTH_COVERAGE_BLOCK` forbids writing about it puts two
  contradictory instructions in one prompt. It is subtracted, with a test.

- **`est_tokens` over-counts, by a content-dependent amount.** Measured
  real/est 0.754 on prose notes and 0.667 on word salad. Any budget constant
  compared against it is really 67-75% of its face value in real tokens.

- **The model's context is not the server's.** `max_position_embeddings`
  says what the weights can do; the serving profile says what a request may
  carry, and here the two differed by 4x. Read the limit from the server's
  refusal, never from the model card.

- **Re-synthesis used to feed ~75% more text than the live run** for identical
  sources. Fixed by (3) above. If you add another path that rebuilds
  `Finding` objects, keep it feeding what a live run feeds or every
  token-keyed heuristic diverges between paths.

## Round two, same evening: the second axis

Commits `4304ee9`, `e4b711a`, `e6035b8`, `c7279d8`.

Looking at WHICH 25 sources the 62% arm left uncited, rather than how many:
7 were relevance-4/5 listicles that should stay uncited, and the other 18
clustered by PRODUCT — all three Joplin sources, both Logseq, three Obsidian,
three vector-database guides — while the document's sections were the five
evaluation CRITERIA. A criteria-shaped document talks about whichever
exemplar the model reached for first. A candidate it never names has no
sentence its sources could be cited in, however long the document is made.
The structural fix had worked on the wrong axis for this question type.

What shipped:

1. **The metric.** The harness reports the cite rate over relevance >= 6
   as well as the raw one. The raw rate has a false ceiling; chasing 100%
   of it forces junk in, which is padding under another name.

2. **The candidate axis** (`4304ee9`). One small structured call over the
   kept sources' titles and summaries names the candidates and which
   sources cover each (`CANDIDATES` -> `CandidatesOut`). Synthesis is told
   to assess every candidate with two or more sources by name, and to
   include a candidate x criterion table (`SYNTH_CANDIDATES_BLOCK`). Gated
   in code: skipped under 8 sources, a one-source candidate is a mention
   not a row, and the block needs two candidates. Extraction failure costs
   the block, never the document.

3. **A reconciliation pass** (`e4b711a`). After the draft, code lists the
   kept sources at relevance 6+ it does not cite (cap 24) and hands them to
   one more call with the draft. The model places each where it adds
   something specific, or accounts for it on an `UNUSED: [n] — ...` line.
   Code accepts the revision only if it is still a document, kept every
   citation the draft had, and did not shrink by more than 5%; otherwise
   the draft stands. Accounted-for sources go under `## Researched but not
   used` with their reason. Costs 130-170s on a 66-source run.

The measurements, body citations only (see the first trap below), same 66
sources, one arm each:

```
original run              1,356 words   22 of 66   33%    21 of 57 strong   37%
firm + structure          2,367         40         61%    38                67%
+ candidate axis          2,750         48         73%    45                79%
+ reconciliation          3,189         55         83%    51                89%
```

Words per cited source 57-62 in every row. The reconciliation pass, run in
isolation over a FIXED draft, placed all 12 of the sources it was given and
1 of 59 new sentences reused draft wording (that one was the table header),
so the growth is content.

Traps, each hit for real:

- **`[n]` markers in the unused list counted as citations.** `cited_ids()`
  scanned the whole document and the harness had its own regex; a run
  reported 58 of 66 cited when the body cited 51. Both now stop at the
  `## Researched but not used` heading. Any new consumer of citation ids
  must use `synthesizer.cited_ids()`, not a regex.

- **The model both cited a source and listed it as UNUSED**, with
  boilerplate reasons, for 8 of 12. A claim about the document is checked
  against the document: entries the body cites are dropped in code.

- **The reasons were self-referential** — "[25] duplicates the claim
  already covered by [25]" — or pointed at nothing. A reason is kept only
  if it names a cited source other than the one it excuses; otherwise the
  entry is listed bare. A reason the reader cannot check is worse than none.

- **Arms are not repeatable at the draft level.** Two full arms of the same
  code left 12 and 18 strong sources uncited before the pass ran. To
  measure a pass, run it over a fixed stored draft (the isolated script
  pattern), not inside a fresh synthesis.

## Still open

- **Per-section length floors** were the fourth item of the plan and were
  NOT built: with the strong-source rate at 89% and the remaining six
  accounted for or judged, floors target document length, not coverage.
  The smallest section still lands at 110-160 words (release recency, few
  sources). Build it only if length per se becomes the complaint.

- **The remaining 6 strong uncited** on the reference run: [26] Studio 3T
  (MongoDB GUI, relevance 8 but arguably off-question), [34], [39], [45],
  [61] roundups, [66] a GitHub feature proposal at 9. Worth a look at
  whether the note-taker's relevance is the right proxy for "belongs in
  this document".

- **Table shape varies.** The candidate block asks for candidate x part of
  the question; one arm produced that exactly, another chose its own
  columns (interface type, search, ingestion). Both are useful tables; the
  prompt does not pin the columns and probably should not.


- **`Researched but not used` wording** for the PART-level list (from
  `_mark_honestly`) still fires for a part that has a section written from
  sources attributed to other parts. The source-level list from the
  reconciliation pass now shares the heading and carries reasons.

- **Depth guidance.** Until length scaling goes further, depth 6 uses its
  sources better than depth 9+. Worth telling anyone running deep.

- **Backlog.** `docs/review-backlog-2026-09-09.md`, triaged 2026-09-10:
  8 fixed / 3 partial / 18 open. The 3 partials are the coverage recheck
  reading headings where the findings argued for the kept sources' own text.

## Reproducing the A/B

`scripts/resynth_ab.py` clones a finished run, re-synthesizes the clone under
whatever overrides you pass, and leaves each arm as its own Library row so
the reports can be compared in the UI. No searching, no fetching — the stored
findings are reused, so a full arm costs ~3 minutes of local compute.

Reference rows currently in the Library for this work:

- `20260911_173708_...` — Matt's original depth-9 run, 66 sources, untouched
- `20260911_195922_...` — `[A/B firm-length]`
- `20260911_201521_...` — `[A/B firm+structure]`
- `20260911_204550_...` — `[A/B candidates]`
- `20260911_205554_...` / `20260911_210916_...` — `[A/B reconcile]`,
  `[A/B reconcile-v2]`: intermediate, before the two guards; safe to delete
- `20260911_211953_...` — `[A/B final]`, the shipped behaviour

The harness's `key=value` overrides only reach module constants; a pass that
is new CODE needs a deploy, then a plain arm. Each arm's JSON now carries the
stages' own log lines (candidates named; sources the pass gained or refused).

## Running the tests

`scripts/test.sh` runs the suite in the dev-compose image and is the project's
own way. `.venv` was empty until 2026-09-11 (created by `uv`, no pip, no
pytest); `uv pip install -r requirements.txt -r requirements-dev.txt` fixed
it, so `.venv/bin/python -m pytest -q` also works and is faster.

Deploy discipline is unchanged and non-negotiable: check for active runs,
`docker compose build app > log; echo $?` (gate on the build's own exit code,
not a pipe's), `up -d --force-recreate`, bounded health loop, then hash the
changed files against HEAD inside the container.
