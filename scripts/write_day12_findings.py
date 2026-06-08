from pathlib import Path

Path("docs/day12_findings.md").write_text(
"""# Day 12 Findings — Encounter Final Test Eval

## Day 12a: Final held-out test evaluation (encounter dataset)

### Results

| Metric | Encounter CV (Day 11) | Encounter Test (Day 12a) |
|--------|-----------------------|--------------------------|
| Micro F1 | 0.972 +/- 0.005 | **0.981** |
| Macro F1 | 0.895 (OOF 0.917) | **0.927** |
| Subset accuracy | — | 0.952 |
| Hamming loss | — | 0.0008 |
| N notes | 1,581 dev | 392 test |
| N codes | 74 | 74 |

### Key finding: clean generalization
Test numbers exceed CV means and are well within the confidence band.
No overfitting. 95.2% of test notes have every code predicted correctly.

### Full project results summary

| Dataset | Notes (dev) | Micro F1 (CV) | Micro F1 (test) |
|---------|-------------|---------------|-----------------|
| Patient-grouped (Day 8b/10) | 779 | 0.821 +/- 0.019 | 0.826 |
| Encounter-grouped (Day 11/12) | 1,581 | 0.972 +/- 0.005 | **0.981** |

### Honest interpretation
The encounter task is easier than the patient-grouped task by construction:
- Avg 1.6 codes/note vs 4.2 — fewer predictions per note
- Notes are focused on a single visit, not a lifetime problem list
- TF-IDF on Gemini-generated text is near-perfect text matching:
  Gemini writes from the code list, TF-IDF reads back to the code list

The 0.981 number is internally consistent and valid for this synthetic
pipeline. Real clinical notes (abbreviations, negation, copy-paste cruft,
implicit diagnoses) would score significantly lower. This caveat must
appear in any README, demo, or writeup.

### Notable recoveries vs patient-grouped dataset
- S35 (vascular injury): F1 0.000 (support=4) -> 1.000 (support=14)
- M41 (scoliosis): F1 0.000 -> recovered with higher support
- The encounter approach directly fixed the support problem diagnosed in Day 9

## Next steps
- Day 12b: transformer baseline comparison on encounter dataset
- Day 13: CLI demo + README rewrite
""")
print("wrote docs/day12_findings.md")
