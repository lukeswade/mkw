# Scoring and triage: a measured review

2026-09-11, late. Luke asked whether the note-taker's relevance score and
the triage stage are doing their jobs, since everything downstream rests on
them. Everything here was measured on this install's own runs (91 completed
research runs, 1,960 kept sources, ~4,000 triage drops, ~2,200 read-and-
rejected pages); the experiments re-fetched real pages and re-scored them
with the deployed prompts.

## Relevance scoring — sound, with one costly rule

**The score predicts usefulness.** Citation rate by score across 1,960 kept
sources: 9 → 82%, 8 → 76%, 7 → 68%, 6 → 54%, 5 → 51%, 4 → 40%. Monotonic.
The rubric bands mean what they say.

**Repeatability is fine away from the line.** 14 kept pages scored twice
more: mean |Δ| 0.5, max 2. But 2 of 14 crossed the keep line (both pages
originally 4 → 2). `notes_recheck` exists for exactly this (3–5 rescored and
averaged) and is off.

**The economy rule costs sources.** The NOTES prompt says: decide the score
first; at ≤2, write no notes. 1,237 of 2,187 rejections sit at exactly 2 —
ten times the count at 3. On 16 pages the shipped prompt scored 2, making
notes compulsory lifted **6 of 16 (38%) over the keep line** (2→6 "Chatterbox
Multilingual v3", 3→6 "Honest Logseq Review", 2→4 a Qwen3-TTS model-card
thread); mean 2.06 → 3.25. On 13 pages originally 0–1 (true junk) the same
change moved the mean 1.85 → 2.00 — no inflation. Moving the cutoff to ≤1
changed nothing (2.06 → 2.06): the mechanism is "score before engaging", not
the number. Extrapolated: ~5 sources per run lost at the boundary; Matt's
66-source run read 23 pages it rejected, 15 of them at 2.

**`standard` is over-assigned to third-party hosts.** Of 34 "standard" sources
with a publisher, ~7 are directory or mirror pages (glama.ai, skillsllm.com,
project-awesome.org, a SourceForge mirror), a GitHub *issue* (hermes-agent
#844, "standard", 9/10), and a user's HF quant credited to "Hugging Face".
Since `standard` outranks research and practitioners in synthesis ordering,
a directory listing can lead a section. Deterministic, checkable by URL.

## Triage — right about junk, wrong about roundups

**False negatives: ~27% of fetchable drops would have been kept.** A random
sample of 33 triage-dropped pages (two runs), fetched and scored by the
note-taker: **9 would be kept, all at ≥6** (9, 9, 8, 7, 7, 7, 7, 6, 6). The
6% figure in `triage_floor`'s docstring was measured on the cap-spared
subset — the best-ranked condemned — a different population. On Matt's run,
140 drops → ~35 good pages never read, against 66 kept.

**The losses cluster on roundups.** Six of the nine are "Best X in 2026" /
"… Compared" pages that the prompt names as "listicle content farms" and
the scorer rates 6–9 as *practitioner*. For an evaluation question they are
the genre the question asks for.

**A reworded prompt changed nothing.** Old rule vs a narrowed rule ("do not
drop a roundup that names what it compares") on the same 32 candidates: the
same 8 roundups dropped in both arms, junk dropped 19/19 in both. Prompt
wording is inert here — the house lesson again.

**Junk precision is adequate:** 25 of 31 junk dropped across both runs; the
leaks were ethics-of-voice-cloning pages on a hardware question.

## What to change (proposed, not shipped at time of writing)

1. **Remove the ECONOMY RULE** from NOTES. Measured +38% recovery at the
   boundary, no junk inflation. Cost: a notes body for pages that stay
   below the bar — ~23 per deep run, a minute or two of decode.
2. **A bounded roundup spare in triage (code).** Condemned, non-video
   candidates whose title matches best / top N / vs / compared / alternatives
   / review are spared, at most 4 per round, and the scorer decides. On the
   sample the pattern catches 6/9 good and 2/24 junk. Historically 281 of
   4,015 drops match (7%) — 1.3 per round uncapped.
3. **Demote `standard` by URL shape (code).** Directory and mirror hosts,
   and issue / pull / discussion / forum-thread paths, become `aggregator`
   regardless of what the model said.
4. **Turn `notes_recheck` on** (setting). Two boundary flips in 14 is the
   case it was built for.

Expected effect on a depth-9 run: +3–6 minutes, roughly +20 kept sources.

---

# Part two: gap analysis and the per-round caps

Same night, same method: 91 completed research runs, 238 gap decisions,
1,256 queries, plus an offline gap A/B on five stored rounds (state document
and kept sources per round are in `rounds/round-NN.md`, so the exact prompt
can be rebuilt).

## Gap analysis — the prompt is sound; the count and the repeats were not

**Saturation is not what ends runs.** 4 of 238 gap decisions said saturated;
60 of 91 runs stopped on the depth limit, 11 on the source cap, 11 on two
dry rounds. Later rounds were as productive as early ones: ~19 new
candidates per search in every round, mean kept relevance 6.5 → 6.4 → 6.3 →
6.3 → 5.8 through round five. Depth was the binding constraint.

**Under-proposal.** Gap analysis returned 4.3 queries on average against a
breadth of 7 at depth 9–10 (mode 5). Offline A/B: "up to 7" → 4.8, "EXACTLY
7" → 4.6. The wording is inert; the model returns ~5. Rounds ran at about
two-thirds of capacity.

**Rephrasings.** Zero exact repeats (the filter works), but 27% of
later-round queries rephrased an earlier one (content-word Jaccard ≥ 0.6),
and those yielded **1.03 kept per query against 1.48** for novel ones. The
prompt says "do not repeat or trivially rephrase"; the model does anyway.

**Long queries.** 29% exceed the prompt's 3–8 words; kept per query 1.67 at
≤8 words vs 1.29 at ≥9 and 0.81 at 12+. Only 2% of searches return nothing,
so the cost is quality, not emptiness. Left alone for now — a length cap
would need a refill and the effect is second-order.

**Planner vs gap.** Round-one (planner) queries yield 1.78 kept each; gap
rounds 1.40. Some of that is diminishing novelty, some is the two items
above.

Shipped (`da30202`): a near-duplicate is filtered like an exact repeat
(Jaccard ≥ 0.7, tags kept aligned); a round short of its breadth is filled
with one written query per thinnest facet, novel against everything
searched; and depth buys 1.6 rounds per unit of effort (depth 10: 8 rounds,
was 5).

## Caps — which one binds, by depth

```
depth  rounds  breadth  picks/round  cap   mean kept (before)
  4      2→4      4         36        28      16.5
  6      3→5      5         44        45      13.7 (n=3)
  9      5→8      7         60        74      66
 10      5→8      7         60        85      41.3 (n=25)
```

At depth ≤5 the **source cap** binds (all 11 cap-stops were at depth ≤5,
mostly round 1–2). At depth ≥6 **rounds** bind: one depth-10 run in 25
reached its cap. Five rounds at ~13 kept per round is ~65 at best; the
measured mean was 41. The round scale addresses that directly; with the
triage and economy changes adding ~20% more kept per round, the cap should
now be reachable at depth 9–10. A depth-10 round is ~5 minutes median, so a
depth-10 run goes from ~25 to ~40 minutes of rounds.

**Left as they are, with the evidence:**

- `engine_share` (a third of a round): 62 cap events over 90 runs — active,
  not dominant. Correct.
- `per_domain` 2, with productive-domain raises and authority exemptions:
  the top domain holds 35% of kept sources on average, ≥30% in 32 of 90
  runs. High, but the runs are product-specific (a Workato question keeps
  Workato docs) and the exemptions are the reason. Not changed.
- `per_facet_cap` (a third): same shape as the engine share; no evidence
  against it in the three runs with facet accounting.
- `candidates_per_round` (breadth·8+4) and `triage_floor` (a tenth, min 3):
  the funnel after them keeps 47% of what is read; wider picks cost triage
  prompt length only. Fine.
- Dry-round rule (kept < 2, twice): fired 11 times, every one on a run that
  had genuinely run out. Fine.
- Concurrency (search 2, fetch 8, LLM 3): the LLM is the bottleneck by an
  order of magnitude; the others do not matter.

## Still open

- Query length: cap at ~10 words with a refill, if the 1.29-vs-1.67 gap is
  worth one more mechanism.
- Facet accounting exists for 3 runs only; re-measure targeting after a
  dozen more.
- The estimate on the New page derives its time from past rounds and will
  adjust on its own; its round count already reflects the new scale.
