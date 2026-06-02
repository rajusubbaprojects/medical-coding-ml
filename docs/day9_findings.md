# Day 9 Findings — Per-code F1 Analysis

## What we did
Re-ran 5-fold CV on the 3char rollup (779 dev notes, 94 codes) to persist
OOF predictions, then computed per-code precision, recall, F1, support,
false negatives, and false positives.

## Headline numbers (unchanged from Day 8b)
- OOF micro F1: 0.821 +/- 0.019
- OOF macro F1: 0.619

## Per-code breakdown
| Bucket | Codes | F1 == 0 | Mean F1 | Min F1 | Max F1 |
|--------|-------|---------|---------|--------|--------|
| support 1-5   | 22 | 9  | 0.262 | 0.000 | 0.889 |
| support 6-15  | 20 | 4  | 0.552 | 0.000 | 0.909 |
| support 16-30 | 20 | 0  | 0.720 | 0.160 | 0.900 |
| support 31-50 |  8 | 0  | 0.799 | 0.621 | 0.937 |
| support 51+   | 24 | 0  | 0.857 | 0.714 | 0.938 |

## Key finding: failure is support-driven, not representation-driven
- 13 codes have F1 = 0. All have support <= 11.
- 10 of those 13 have pred+ = 0: the model never predicts them at all.
- Filtered micro F1 (support >= 10, 60 codes): **0.837** vs 0.821 overall.
  The 34 tail codes move micro F1 by only 0.016 — they are rare enough
  that they barely affect micro averaging.
- Those same tail codes devastate macro F1 (mean 0.361 vs 0.765 for the
  learnable codes) — macro treats every code equally regardless of frequency.

## Notable outliers
- **C18** (colon cancer, support=3): pred+=3, all false positives. The model
  fires on oncology language that appears in other contexts.
- **R57** (shock, support=4): pred+=9, 8 false positives. "Shock" language
  appears broadly in discharge notes.
- **M27** (jaw disorders, support=18): recall only 0.111 despite reasonable
  support. Likely a text-signal problem — jaw conditions may not have
  distinctive language in Gemini-generated notes.
- **S83** (knee dislocation, support=11): precision=1.0 but recall=0.182.
  The model is confident when it fires but mostly doesn't fire.

## Transformer baseline decision
**Deferred.** The failure modes are support-driven. A transformer baseline
would measure "which model fails less on 3-example classes" — not an
informative comparison. BERT does not conjure signal that isn't there.

The right intervention is more data per code, not a better encoder.

## Recommended next steps
1. **Encounter-grouped notes (Day 10a)**: restructure generation from
   "1 note per patient" to "1 note per Synthea encounter." Each patient
   has multiple encounters; this multiplies available notes and increases
   per-code support for rare conditions without touching the model at all.
2. **After data expansion**: re-run CV, check whether tail codes recover,
   then run transformer comparison on a dataset where support isn't the
   confound.
3. **Final eval (Day 11)**: held-out test bucket (202 notes) still untouched.
   Run this before any demo work.

## Artifacts
- `data/oof/oof_3char.csv` — OOF predictions (73,226 rows, gitignored)
- `data/oof/per_code_stats_3char.csv` — per-code stats (94 rows, gitignored)
- `src/baseline/run_day9.py` — CV runner that persists OOF predictions
- `src/baseline/per_code_analysis.py` — per-code analysis script
