from pathlib import Path

CONTENT = '''# Day 6 Findings: Data Cleaning + Baseline Classifier

## Goal
Clean the synthetic note labels (filter Z-codes, collapse ICD-10 fan-out), then
train a TF-IDF + logistic-regression baseline to get a first ICD-10 prediction
metric. Compare three label rollups to quantify whether cleaning helped.

## Data
- 100 synthetic discharge notes (from Day 5), stored in `notes_synth`.
- Labels in `notes_labels` (1684 raw rows, 236 distinct ICD-10 codes).
- Two clean views built: `notes_labels_clean_3char`, `notes_labels_clean_chapter`.

### Rollups
| rollup  | description                                  | distinct codes |
|---------|----------------------------------------------|----------------|
| dirty   | raw codes, fan-out + Z-codes included        | 236 (153 after rare-filter) |
| 3char   | rolled to 3-char family, Z-codes dropped     | 49 (36 after rare-filter)   |
| chapter | rolled to first letter, Z-codes dropped      | 16 (12 after rare-filter)   |

Fan-out collapse was dramatic: 16.8 -> 3.9 labels/note (3char). Six notes were
dropped entirely because all their labels were Z-codes.

## Model
- Features: TF-IDF, word (1-2 gram) + char_wb (3-5 gram), FeatureUnion.
- Classifier: OneVsRest(LogisticRegression, class_weight="balanced").
- Split: 80/20 random, seed=42. Rare codes (<3 notes) dropped.

## Results

| rollup  | n_codes | micro F1 | macro F1 | subset acc | hamming |
|---------|---------|----------|----------|------------|---------|
| dirty   | 153     | 0.637    | 0.388    | 0.150      | 0.056   |
| 3char   | 36      | 0.685    | 0.477    | 0.316      | 0.050   |
| chapter | 12      | 0.742    | 0.560    | 0.421      | 0.101   |

### Reading the table
- **Cleaning helped (dirty -> 3char):** +0.048 micro F1, and subset accuracy
  MORE THAN DOUBLED (0.150 -> 0.316). Collapsing fan-out means a note no longer
  has to predict K06 AND K06.8 AND K06.9 to score an exact match.
- **Coarsening helped further (3char -> chapter)** but mostly by making the task
  easier (12 classes), not by improving the model. Chapter is the "sanity check"
  baseline; 3char is the real target.
- **Hamming loss is worst for chapter** because with only 12 columns each error
  costs proportionally more. Micro/macro F1 are the trustworthy metrics here.

### Per-code observations (3char)
- High-support clinical codes are near-perfect on precision: D64 (anemia),
  J20 (bronchitis), K06 (gingival), I10 (HTN), H66/H67 (otitis) all at P=1.00.
- Recall is the weak spot (0.25-0.71): the model is conservative, missing true
  labels rather than inventing wrong ones.
- Rare codes (support 1-2): T45, J32, M17/M84/M94 all at F1=0.00. Not enough
  examples to learn. Expected.

## Caveats (IMPORTANT)
1. **Synthetic-data inflation.** Gemini writes "admitted with acute bronchitis"
   nearly verbatim, so TF-IDF maps text -> code trivially. Real clinical notes
   (abbreviations, negation, copy-paste) will NOT yield 0.685. This number
   validates pipeline self-consistency, not real-world coding ability.
2. **Tiny test set (n=19).** Single random split; metrics have high variance.
   Day 7 should use k-fold cross-validation.
3. **No threshold tuning.** Predictions at default 0.5. Recall could likely be
   improved by lowering the threshold.

## Day 7 backlog
1. k-fold CV for stable metrics (replace single split).
2. Threshold sweep to trade precision for recall.
3. Scale to 500-1000 notes; re-measure (does more data lift 3char?).
4. Fix the primary-Dx picker (chapter scoring) before scaling.
5. Try a transformer encoder (e.g. ClinicalBERT) vs the TF-IDF baseline.
'''

Path("docs/day6_findings.md").write_text(CONTENT)
print("Wrote docs/day6_findings.md")
print(f"  {len(CONTENT.splitlines())} lines")