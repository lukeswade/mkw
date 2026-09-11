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

1. **`_SINGLE_CALL_BUDGET` 28k -> 100k** (`593dc90`). The served model reports
   `max_position_embeddings 262144`. A timed call measured marginal prefill at
   **402 tok/s**, and `est_tokens` over-counts real tokens by ~33% (est 84,211
   came back as 63,498 actual). So 100k est is ~75k real, under a third of the
   window. Single-call synthesis measured 23% faster than the four-batch
   digest at identical citation coverage.

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

- **`est_tokens` over-counts by about a third.** Any budget constant compared
  against it is really ~75% of its face value in real tokens.

- **Re-synthesis used to feed ~75% more text than the live run** for identical
  sources. Fixed by (3) above. If you add another path that rebuilds
  `Finding` objects, keep it feeding what a live run feeds or every
  token-keyed heuristic diverges between paths.

## Still open

- **62% is not 100%.** The model still writes 2,398 against a 3,300 target.
  It has a strong prior toward ~1,500 words that firmer instruction only
  partly overcomes. Next lever if wanted: per-section length targets, or a
  second pass that expands sections whose sources went uncited.

- **`Researched but not used` wording.** It fires for a part that HAS a
  section when that section was written from sources attributed to other
  parts. Technically precise, reads more damning than the reality.

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
- `20260911_201521_...` — `[A/B firm+structure]`, the current shipped behaviour

## Running the tests

`scripts/test.sh` runs the suite in the dev-compose image and is the project's
own way. `.venv` was empty until 2026-09-11 (created by `uv`, no pip, no
pytest); `uv pip install -r requirements.txt -r requirements-dev.txt` fixed
it, so `.venv/bin/python -m pytest -q` also works and is faster.

Deploy discipline is unchanged and non-negotiable: check for active runs,
`docker compose build app > log; echo $?` (gate on the build's own exit code,
not a pipe's), `up -d --force-recreate`, bounded health loop, then hash the
changed files against HEAD inside the container.
