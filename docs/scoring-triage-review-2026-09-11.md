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
