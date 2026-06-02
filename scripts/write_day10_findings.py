from pathlib import Path

Path("docs/day10_findings.md").write_text(
"""# Day 10 Findings — Final Held-Out Test Evaluation

## What we did
Ran final evaluation on the held-out test bucket (202 notes, never touched
during training or CV). Fixed a stale model bundle discovered during eval
(bundle was from Day 6, 75 notes/36 codes; retrained on full 779-note dev set
before running test eval).

## Results

| Metric | Dev CV (Day 8b) | Test (Day 10) |
|--------|----------------|---------------|
| Micro F1 | 0.821 +/- 0.019 | **0.826** |
| Macro F1 | 0.619 (OOF) | 0.612 |
| Subset accuracy | — | 0.421 |
| Hamming loss | — | 0.014 |
| N notes | 779 | 202 |
| N codes | 94 | 94 |

## Key finding: clean generalization
Test micro F1 (0.826) is above the dev CV mean (0.821) and well within
the +/- 0.019 confidence band. No overfitting. The pipeline generalizes
consistently to unseen notes from the same synthetic distribution.

## Notable per-code results on test
- K06 (periodontal, support=77): F1 0.94 — dominant code, near-perfect
- D64 (anemia, support=34): F1 0.93
- S93 (ankle ligament, support=15): F1 0.93
- G89 (pain codes, support=24): F1 0.78 — recall was the weak point (0.67)
- M84 (bone disorder, support=20): F1 0.69 — recall 0.55, hardest high-support code

## Bug found and fixed
`models/baseline_3char.joblib` was a stale Day 6 bundle (75 notes, 36 codes).
CV was always correct (retrains each fold from scratch), but the saved bundle
was never updated after the Day 8b scale-up. Fixed by rerunning:
`python -m src.baseline.train --rollup 3char`
Bundle now: 779 notes, 94 codes, 17.4s fit time.

## Caveats (unchanged)
- All notes are Gemini-generated synthetic text. Real clinical notes would
  score significantly lower due to abbreviations, negation, and copy-paste.
- The test split has a known minor data-leak wart (~9 notes wrong-bucketed
  from Day 7; documented in day7_findings.md; accepted).
- This is the only honest test number. It will not be re-run.

## What's next
- Day 10b: encounter-grouped note generation (more data, higher tail-code support)
- Day 11: transformer baseline on expanded dataset
- Day 12: CLI demo + README rewrite
""")
print("wrote docs/day10_findings.md")
