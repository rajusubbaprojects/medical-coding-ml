# Day 11 Findings — Encounter-Grouped Note Generation

## What we did
Restructured note generation from patient-grouped (1 note per patient lifetime)
to encounter-grouped (1 note per Synthea encounter). Generated 2,000 encounter
notes via Gemini, loaded to BigQuery, built an 80/20 split, and ran 5-fold CV.

## Generation stats
- Encounters attempted: 2,000
- Notes generated: 1,993 (7 failures, accepted)
- Cost: $3.51
- Run ID: 20260604-204938
- Strategy: encounter-multilabel (v1-encounter-multilabel prompt)

## Dataset comparison

| Dataset | Notes (dev) | Codes | Avg labels/note | Micro F1 | Macro F1 |
|---------|-------------|-------|-----------------|----------|----------|
| Patient-grouped (Day 8b) | 779 | 94 | 4.2 | 0.821 +/- 0.019 | 0.619 |
| Encounter-grouped (Day 11) | 1,581 | 74 | 1.6 | 0.972 +/- 0.005 | 0.895 |

## Key findings

### Large improvement in both metrics
Micro F1: 0.821 -> 0.972 (+0.151)
Macro F1: 0.619 -> 0.895 (+0.276)
Std tightened from +/- 0.019 to +/- 0.005 — much more consistent across folds.

### Two contributing factors (not separable without ablation)
1. More data: 1,581 vs 779 dev notes doubles training signal
2. Simpler task: 1.6 avg codes/note vs 4.2 means fewer predictions per note,
   fewer chances to be wrong, and tighter TF-IDF signal per code

The macro F1 improvement (0.619 -> 0.895) is the more meaningful number —
it treats all 74 codes equally and is less dominated by high-frequency codes.

### Honest caveat: not a pure apples-to-apples comparison
Encounter notes are a genuinely easier task than patient-grouped notes.
A patient note with 8 codes requires the model to predict all 8 correctly;
an encounter note with 1-2 codes is a much simpler problem. The improvement
reflects both better data AND a simpler target distribution.

### Code coverage
74 codes in encounter dataset vs 94 in patient dataset. Some rare codes
(F32, G44, R11) still don't reach min_code_freq=3 even with 1,993 notes —
consistent with Day 9 finding that these are genuinely rare in Synthea.

## Infrastructure changes
- Added encounter_id column to notes_synth BQ table
- Added encounter_id to NOTES_SCHEMA in load_to_bq.py
- New: src/notes/select_encounters.py — fetch encounters from BQ
- New: src/notes/run_encounter_generation.py — generation loop keyed on encounter_id
- New: src/baseline/load_encounter_data.py — load encounter split for CV/eval
- New: src/baseline/run_encounter_cv.py — CV runner for encounter notes
- New: scripts/make_encounter_split.py — one-shot split builder
- Split: data/splits/encounter_split.json (1,595 dev / 398 test encounter_ids)

## Next steps
- Day 12a: final eval on encounter test bucket (398 notes, untouched)
- Day 12b: transformer baseline comparison on encounter dataset
- Day 13: CLI demo + README rewrite
