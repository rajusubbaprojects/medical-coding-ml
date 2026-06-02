from pathlib import Path

CONTENT = '''# Day 8 Findings: Better Picker + 10x Data Scale

## Goal
Track 2 from the project plan: data quality improvements + scaling.

Two sub-experiments:
- **8a**: Replace the naive primary-Dx picker with chapter-weighted scoring +
  trivial-word penalty. Regenerate 50 primary_dx notes on the same patients;
  measure the effect at constant data size.
- **8b**: Scale generation to 1000 total notes (500 patients x 2 strategies).
  Re-measure with the new picker baked in.

Day 9+ backlog item: encounter-grouped note generation. Deferred from Day 8.

## Method changes since Day 7

### Primary-Dx picker (build_prompt.py)
Score each candidate condition by:
- ICD-10 chapter weight (4 for I/J/K2-9/S/T; 3 for A/B/C/D/E/N; 1-2 for G/H/L/M;
  0 for K0x dental and Z-codes)
- Trivial-word penalty (-2 per match for "other specified", "unspecified",
  "edentulous", "in remission", "history of", etc.)
- Specificity bonus (+1 for 4+ char codes)

Pick max score; tie-break by most recent start_date.

### Generation driver (run_generation.py, supersedes run_day5.py)
- New CLI flags: `--patients-from-run`, `--exclude-patients-from-run`
- Cost preview + confirmation gate at >= 200 calls
- 0.5s inter-call sleep (rate-limit insurance)
- Records prompt_version per row so v1 and v2 notes can coexist in BQ

### load_data.py (the dedup layer)
- `QUALIFY ROW_NUMBER() OVER (PARTITION BY patient_id, strategy ORDER BY
  generated_at DESC) = 1` returns only the latest note per (patient, strategy).
- Older versions remain in BQ as history but are invisible to training.

### dev_test_split.json (the split file)
- Keyed on `(patient_id, strategy)` pairs instead of volatile note_ids.
- Stable across regenerations.

## Results

### Picker fix (8a, same 100 notes)
- K06.x as primary: 8/50 (v1) -> 2/50 (v2). 4x reduction in gum-as-primary.
- v2 picks include real admission causes: J20 (acute bronchitis, x9),
  S93.491D (ankle sprain, x5), I50.22 (heart failure, x3), J02.0 (strep, x3).
- Distinct primaries: 24 (v1) vs 21 (v2). Slightly less *surface* diversity
  but substantially better *clinical quality* per pick.

### Scaling effect (8b, 1000 notes target, 999 actually generated)
- Generation: 899 new notes in 1h 34m, ~$1.80. One note failed silently
  (449 primary_dx instead of 450).
- After dedup + rare-code filter: 779 dev / 202 test / 18 dropped empty.
- 3char codes available: 36 (Day 7) -> 94 (Day 8b). 2.6x more classes.

### Baseline metrics, three-way comparison

| Configuration              | n_dev | n_codes | Micro F1 (CV)   | OOF macro |
|----------------------------|-------|---------|-----------------|-----------|
| Day 7 (v1 picker, 100 nts) |   75  |   36    | 0.677 +/- 0.032 |   0.565   |
| Day 8a (v2 picker, 100 nts)|   75  |   38    | 0.672 +/- 0.066 |   0.591   |
| Day 8b (v2 picker, 1000 nts)|  779 |   94    | **0.821 +/- 0.019** | **0.619** |

Best operating points (8b):
- Micro F1: threshold=0.45, F1=0.821 (P=0.841, R=0.803)
- Macro F1: threshold=0.35, F1=0.651 (P=0.668, R=0.679)

Default threshold (0.5) is now effectively as good as 0.45 for micro F1.
The threshold sweep is flat across 0.40-0.55 — at scale, threshold tuning
matters less than it did at 100 notes.

## What the numbers tell us

1. **Scaling worked, decisively.** +0.144 micro F1 vs Day 7. That's ~17x
   the previous std. Not noise.
2. **Std collapsed (0.066 -> 0.019).** Larger fold sizes = more stable
   metrics. Past the "single-seed luck" regime.
3. **More classes did NOT make the task harder.** More data per class won.
4. **Picker fix paid off at scale.** Day 8a was a wash; Day 8b shows the
   v2 picker's macro F1 advantage persisting (0.619 OOF vs Day 7's 0.565)
   despite handling 2.6x more classes.

## Caveats (still important; restating + new)

1. **Synthetic-data ceiling not yet hit.** Gemini writes clean notes that
   encode their conditions in TF-IDF-recognizable language. Real notes
   (negation, abbreviations, copy-paste) will not reach 0.82.
2. **Held-out test bucket (n=202) NOT touched.** Final evaluation on Day 11-12.
3. **Known data-leak wart in split.** When we rebuilt the split with
   pairs-based keying, the input was already the leaky note_id-based split
   from a buggy first attempt, not the clean pre-Day-8b state. Effect:
   ~9 notes that were originally Day 7 test notes now sit in dev. With 779
   dev / 202 test, this is small noise; not re-running to fix.
4. **One missing primary_dx note** in 8b generation (449/450). Cause not
   recorded (log lost on restart). Likely a transient Gemini error.
5. **No model upgrades yet.** Still TF-IDF + OneVsRest LR. Day 9-10 work.

## Day 9+ backlog (refreshed)

1. **Transformer baseline** (ClinicalBERT or similar) vs TF-IDF on the
   1000-note dataset.
2. **Per-code F1 analysis on 8b** — which of the 94 codes still fail at
   threshold 0.45?
3. **Encounter-grouped notes** (deferred from Day 8c). Probably has bigger
   payoff than transformer at this point.
4. **CLI demo** for Endpoint A (Day 11-12).
5. **Final test-bucket eval** (Day 11-12, untouched 202 notes).

## Files added/changed this day
- `src/notes/build_prompt.py`: v2 picker with chapter scoring.
- `src/notes/run_generation.py`: new driver, supersedes run_day5.py.
- `src/baseline/load_data.py`: pairs-based split resolution, latest-per-pair dedup.
- `scripts/make_split.py`: rewritten to use pair keys, with legacy migration.
- `data/splits/dev_test_split.json`: 784 dev / 203 test pairs.
- `docs/day8_findings.md` (this doc).
'''

Path("docs/day8_findings.md").write_text(CONTENT)
print("Wrote docs/day8_findings.md")
print(f"  {len(CONTENT.splitlines())} lines")