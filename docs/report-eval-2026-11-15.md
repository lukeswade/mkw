# Report eval — 2026-11-15

A frozen set of eight real runs, and a scorer, so a change to how reports are
written can be measured before it ships. Built because the 2026-11-15
finance report named two already-acquired practices as acquisition targets,
and nothing but reading it would have said so.

Harness: `scripts/eval/report_eval.py`, run through `scripts/eval/report_eval.sh`
(a throwaway dev container with the working tree's `app/`; the library is
read-only and only `data/eval/` is written). Tests: `tests/test_report_eval.py`.
Set: `data/eval/reports-2026-11-15/` (gitignored). Never re-freeze into it;
start a new dated set.

## Using it

```bash
scripts/eval/report_eval.sh score                     # the stored reports, no model
scripts/eval/report_eval.sh resynth <label>           # re-write all eight with the code in the tree
scripts/eval/report_eval.sh score <label> --judge     # score that arm; the judge needs the model
scripts/eval/report_eval.sh check                     # every known error still detected in its baseline
```

- `resynth` runs synthesis, citation validation, the citation check if it is
  on, and the closing sections, as a run would. It never touches the library.
- `--only=<case>` limits any command to one case.
- `--set=name=value` changes a setting for that arm only, for example
  `--set=fact_ledger=true`.
- Each `score` appends a line to `results.jsonl`.
- A command that used the model asks the server to unload it afterwards,
  unless a research run is using it. Pass `--keep-loaded` when another arm
  follows straight away.
- The judged pass takes about 7 minutes; a `resynth` of all eight about 41.

## The set

| Case | Depth | Sources | Known errors (substantive / citation) |
|---|---|---|---|
| `fin-gi-urology-d8` | 8 | 66 | 4 / 0 |
| `matt-amd-llm-d9` | 9 | 82 | 1 / 1 |
| `matt-gx470-timesert-d10` | 10 | 88 | 1 / 1 |
| `stt-m4pro-d3` | 3 | 25 | 1 / 1 |
| `notes-selfhost-d6` | 6 | 47 | 0 / 1 |
| `wasm-adoption-d5` | 5 | 37 | 0 / 1 |
| `qwen-m4pro-d6` | 6 | 39 | 1 / 0 |
| `kb-agents-d9` | 9 | 66 | 0 / 1 |

Per case: `meta.json` (question, plan, parts), `findings/` (every kept source
as the run stored it) and `baseline.md` (the report as it stood).

**Known errors** are in `cases.json`: 14 facts, each verified in the run's
stored sources and against its question. Two kinds:

- **Substantive (8):** wrong subject, owner, status or comparison. Examples:
  - two practices listed as targets that a source says were already bought;
  - a backer that had exited;
  - a 2021 forecast for 2025 used as a current driver;
  - a speed measured on one model and GPU credited to another;
  - a tap's depth given as the insert's.
- **Citation (6):** the figure is right and another kept source states it,
  but the sentence cites the wrong one.

Each fact has a plain statement for the judge and a pattern for the
model-free screen. `check` proves every pattern fires on its own baseline.

Three errors I had listed earlier were left out because the sources did not
prove them. One is worth knowing: "Time-Sert 4412E-111A" is not in any
source verbatim, but a forum post gives the kit as "4412E-111" and the
maker's page says aluminum inserts add an "A". It is a composition, not an
invention.

## Measures

Without a model, for every report:

| Measure | Meaning |
|---|---|
| words | length of the body |
| cited | kept sources the body cites |
| unmatched | cited sentences with a figure their sources' notes do not state (the citation check's own string pass) |
| repeats | sentences that restate one from an earlier section; summaries are exempt |
| lost | parts of the question whose kept sources all went uncited |
| subst / cite | known errors present, by kind |

With `--judge`:

| Measure | Meaning |
|---|---|
| contra | substantive facts the report contradicts. Judged one passage at a time; counted only when the quoted sentence is in the report. |
| conflicts | pairs of statements that cannot both be true. Counted only when both quotes are in the report, both name the subject, and the model confirms the pair when shown it alone. |

## Baseline (the stored reports)

| Case | Words | Cited | Unmatched | Repeats | Subst | Cite | Contra (judge) | Conflicts (judge) |
|---|---|---|---|---|---|---|---|---|
| fin-gi-urology-d8 | 3,455 | 44/66 | 2/37 | 6 | 4/4 | – | 3/4 | 1 |
| matt-amd-llm-d9 | 3,269 | 44/82 | 7/65 | 4 | 1/1 | 1/1 | 1/1 | 0 |
| matt-gx470-timesert-d10 | 3,153 | 56/88 | 4/32 | 4 | 1/1 | 1/1 | 1/1 | 0 |
| stt-m4pro-d3 | 1,559 | 24/25 | 2/30 | 0 | 1/1 | 1/1 | 1/1 | 0 |
| notes-selfhost-d6 | 3,099 | 42/47 | 3/13 | 0 | – | 1/1 | – | 0 |
| wasm-adoption-d5 | 2,349 | 30/37 | 1/27 | 1 | – | 1/1 | – | 0 |
| qwen-m4pro-d6 | 2,186 | 33/39 | 4/58 | 1 | 1/1 | – | 1/1 | 0 |
| kb-agents-d9 | 3,954 | 54/66 | 1/25 | 1 | – | 1/1 | – | 0 |
| **Total** | 23,024 | 327/450 | 24/287 | 17 | 8/8 | 6/6 | 7/8 | 1 |

The known-error columns are 8/8 and 6/6 by construction: these are the
reports the errors were found in. An arm is better when those fall.

## Today's code (arm `current`, 2026-11-16)

All eight reports rewritten by the code at `06fdedd`, one run each.

| | Stored reports | `current` |
|---|---|---|
| Substantive known errors present | 8/8 | 4/8 |
| Citation known errors present | 6/6 | 0/6 |
| Contradicted (judge) | 7/8 | 4/8 |
| Unmatched figures | 24/287 | 3/247 |
| Repeats | 17 | 7 |
| Sources cited | 327/450 | 336/450 |
| Words | 23,024 | 23,552 |

- The citation check accounts for the citation column and the unmatched
  figures.
- The four substantive errors that recur, each checked in the new text:
  - finance: New Jersey Urology offered as a target, and the 2021 shortage
    forecast used as a current driver;
  - GPU report: RTX 3080 Ti speeds applied to the RX 6700 XT;
  - Qwen report: MTP said to double Qwen3.8's speed.
- Part of 8 → 4 is chance: the stored reports were chosen for having these
  errors, and each was rewritten once. A difference of one or two facts
  between two arms means little without repeats.
- Four of this arm's timings are inflated, because fact-sheet trials shared
  the model while it ran.

## The fact sheet (arm `ledger`, 2026-11-16)

The same eight rewritten with `fact_ledger` on (`da5f7e0`), one run each.

| | Stored reports | `current` | `ledger` |
|---|---|---|---|
| Substantive known errors present | 8/8 | 4/8 | 2/8 |
| Citation known errors present | 6/6 | 0/6 | 0/6 |
| Unmatched figures | 24/287 | 3/247 | 7/275 |
| Repeats | 17 | 7 | 2 |
| Sources cited | 327/450 | 336/450 | 323/450 |
| Words | 23,024 | 23,552 | 20,729 |
| Time to rewrite all eight | – | 40.6 min | 49.7 min |

By report, substantive errors `current` → `ledger`:

| Report | `current` | `ledger` | What changed |
|---|---|---|---|
| Finance | 2 | 0 | New Jersey Urology is now something U.S. Urology Partners acquired. The stale forecast and One GI are no longer mentioned at all. |
| Qwen on M4 Pro | 1 | 0 | The MTP doubling is credited to Qwen3.6, as its source says. |
| GPU stack | 1 | 1 | Both credit a speed to Qwen3-Coder that no source gives: `current` from the RTX 3080 Ti source, `ledger` from an RX 6700 XT source about another model. |
| Time-Sert | 0 | 1 | "The insert should sit slightly below the surface (approx. 6mm)": the tap's depth again. Not in `current`. |
| Speech-to-text | 0 | 0 | |

What this does and does not show:

- Four errors to two, on one run per arm. A rewrite of the same report makes
  different mistakes each time (the Time-Sert error came and went), so this
  is suggestive, not established. A second run of each arm would say more.
- The gain is where the fact sheet was aimed: who owns what, and which
  model a result belongs to when both are named things with their own
  entries. It did nothing for a figure credited to a model that has no
  figures at all.
- It costs time and coverage: 22% longer over the set (the four pairs with
  clean timings: +21% to +63%), 12% fewer words, and 13 fewer sources
  cited. Unmatched figures rose from 3 to 7.
- Unknown: whether a fact-sheet line caused the Time-Sert error. Its checks
  test a line's figures and sources, not which thing a figure describes,
  and the arm does not keep the fact sheet it wrote.
- The judge's column for this arm is 3/8: two on the finance report that
  are wrong on reading ("General Atlantic-backed" taken as contradicting
  "General Atlantic invested"), and the Time-Sert one, which is right. It
  was run before the GPU fact below was corrected.

**A known error was corrected after this arm.** The GPU report's fact was
"the ~24 tok/s figure is an RTX 3080 Ti measurement of Qwen3.6", screened as
any passage with 24 tok/s and the RX 6700 XT. Two other kept sources do
report about 24 tok/s on an RX 6700 XT, for other models, so that screen
flagged correct sentences and missed the same figure credited to Qwen3-Coder
from a different source. The fact is now "no source gives a speed for
Qwen3-Coder 30B-A3B" (`qwen3-coder-has-no-measured-speed`), and the scorer
reads a nested bullet under its parent and a paragraph under a bold-only
line above it. All three arms above were rescored with it.

## How far to trust the judge

- **Contradicted facts: usable.** It agreed with 7 of the 8 hand-verified
  substantive errors. It is not stable: an earlier pass found the one it
  missed here and missed two others. Two changes got it to 7:
  - judging one passage at a time. Shown every passage at once, a correct
    passage excused a wrong one;
  - stating each fact so that the error contradicts it outright.
- **Conflicts: a floor, not a count.**
  - Asked to find conflicts in a whole report, the model returned 7
    checkable pairs across the set, and 6 were one statement made twice.
    Confirming each pair alone removed all 6 and kept the real one: an
    investor's exit window given as 1–2 years in one section and a 3–5 year
    hold in another.
  - It still misses most conflicts. The finance report has at least two
    more (a platform's backers stated three ways; one company's multiple as
    "mid-teens" and as 6–13x). Both are covered as known errors.
  - Finding conflicts properly needs a pass per named company. That is the
    consistency pass item 3 will build.
- The model is Qwen3.6-35B-A3B with thinking off, the same model that wrote
  the reports.

## Limits

- Eight cases, 14 known errors: enough to see a change move, not to rank
  two close arms. Add a case whenever a real report is found wrong.
- The stored reports were written by older code; seven predate the citation
  check. For an A/B, write a `current` arm first and compare against that,
  not against `baseline`.
- A pattern screen can miss an error worded a new way, and can stay silent
  when the subject is dropped. Read `subst` beside `contra`.
- Known errors are the ones found by reading. A report can be wrong in ways
  no fact covers.

## Next

1. A second run of `current` and `ledger` (about 90 minutes unattended), to
   see how much of 4 → 2 is the fact sheet and how much is chance.
2. Have `resynth` keep the fact sheet it wrote beside each report, so an
   error can be traced to a line or cleared of it.
3. The error it does not touch: a figure credited to a thing the sources
   give no figure for. That needs a check on the subject of each cited
   figure, in the citation pass.
