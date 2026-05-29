# Day 7 Findings: Stable Metrics + Threshold Tuning

## Goal
Move from Day 6's single-split metric to numbers we can trust. Three sub-tracks:
1. Carve out a stable, stratified dev/test split.
2. 5-fold cross-validation on the dev bucket.
3. Threshold sweep on out-of-fold predictions; pick an operating point.

## Method

### Stratified dev/test split
- 96 notes with at least one 3char label.
- 80/20 multi-label stratified split (iterative-stratification), seed=42.
- Result: 77 dev, 19 test. Split committed to `data/splits/dev_test_split.json`.
- All 42 codes in the test set also appear in dev. Strong stratification.
- The 19-note test set is held out; not touched in Day 7.

### Cross-validation
- 5-fold multi-label stratified CV on the dev bucket.
- Per fold: ~60 train / ~15 val. TF-IDF + OneVsRest LR pipeline (unchanged).
- Out-of-fold (OOF) predictions: every dev note gets one prediction made by
  a model that did not see it.

### Threshold sweep
- Probabilities from OOF predictions.
- Sweep threshold from 0.10 to 0.90 in steps of 0.05.
- Pick best operating point for micro F1 and macro F1.

## Results

### CV: comparison across rollups (default threshold 0.5)

| rollup  | Day 6 single-split micro F1 | Day 7 CV mean micro F1 | Day 7 std | OOF macro F1 |
|---------|------------------------------|------------------------|-----------|--------------|
| dirty   | 0.637                        | 0.659                  | 0.075     | 0.503        |
| 3char   | 0.685                        | 0.677                  | 0.032     | 0.565        |
| chapter | 0.742                        | 0.719                  | 0.073     | 0.656        |

### Threshold sweep on 3char

Best operating point (both micro and macro): threshold = 0.45.

| metric          | threshold=0.5 | threshold=0.45 | delta   |
|-----------------|---------------|----------------|---------|
| Micro F1        | 0.677         | 0.730          | +0.053  |
| Macro F1 (OOF)  | 0.565         | 0.695          | +0.130  |
| Micro precision | 0.84          | 0.74           | -0.10   |
| Micro recall    | 0.57          | 0.72           | +0.15   |

Curve shape: clean unimodal peak at 0.45 for both micro and macro. No noise.

## Headline

**Tuned baseline (3char rollup, threshold=0.45):**
- Micro F1: 0.730
- Macro F1: 0.695

These are out-of-fold cross-validated numbers on 75 dev notes. The held-out
test set (19 notes) is untouched.

## What changed from Day 6

**Day 6's +0.048 dirty -> 3char improvement was inside the noise.** With CV
error bars (dirty 0.659 +/- 0.075, 3char 0.677 +/- 0.032), the micro F1 gap
of 0.018 is not statistically meaningful.

**But cleaning bought us stability.** 3char has more than 2x lower std than
either dirty or chapter (0.032 vs ~0.073). At this dataset size, that's a
more meaningful win than a marginal micro F1 bump.

**Threshold tuning was a real win** (+0.053 micro, +0.130 macro). The default
threshold of 0.5 was leaving recall on the table; 0.45 captures more true
labels without inviting many false positives.

## Caveats (unchanged from Day 6, restated)

1. **Synthetic-data ceiling.** Gemini writes very clean notes. TF-IDF maps
   "admitted with acute bronchitis" -> J20 trivially. Real notes (negation,
   abbreviations, copy-paste) will not yield 0.730.
2. **Test bucket still untouched.** Day 7 results are dev-bucket CV. Final
   test evaluation comes on Day 11-12 against the held-out 19 notes.
3. **No hyperparameter tuning beyond threshold.** TF-IDF and LR settings are
   the Day 6 defaults. Day 8 or 9 work.

## Files added this day
- `scripts/make_split.py`, `data/splits/dev_test_split.json`
- `src/baseline/cv.py`
- `src/baseline/threshold.py`, `models/thresholds_3char.json`
- `src/baseline/run_day7.py`
- `src/baseline/load_data.py` refactored (bucket-aware)
- `src/baseline/train.py`, `src/baseline/evaluate.py` refactored

## Day 8 backlog (refreshed)
Day 7 closed Track 1. Track 2 (data quality round 2) is next:
1. Fix primary-Dx picker (chapter scoring + trivial-word penalty).
2. Encounter-grouped note generation.
3. Scale to 500-1000 notes.
4. Re-run Day 7 metrics on the larger dataset; compare.

Then Track 3 (transformer baseline) follows on Days 9-10.
