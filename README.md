# Medical Coding ML

An end-to-end ML pipeline for automated ICD-10 code prediction from clinical
discharge notes, built as part of a healthcare AI portfolio.

> **Note**: This is a research/portfolio project using synthetic data.
> It is not a clinically deployable system.

---

## Results

| Dataset | Notes (dev) | Codes | Micro F1 (CV) | Test F1 |
|---------|-------------|-------|---------------|---------|
| Patient-grouped | 779 | 94 | 0.821 ± 0.019 | **0.826** |
| Encounter-grouped | 1,581 | 74 | 0.972 ± 0.005 | **0.981** |

Both test numbers are from held-out sets touched exactly once, after all
training and tuning decisions were finalized on the dev set.

---

## Architecture
Synthea synthetic EHR (BigQuery)
|
v
conditions_billable view (Z-codes filtered)
|
v
Gemini 2.5 Flash (Vertex AI) -- generates discharge notes from code lists
|
v
notes_synth + notes_labels (BigQuery)
|
v
TF-IDF (word 1-2gram + char_wb 3-5gram)

OneVsRest LogisticRegression
|
v
Multi-label ICD-10 classifier


Two dataset strategies:
- **Patient-grouped**: one note per patient, lifetime conditions (4.2 codes/note avg)
- **Encounter-grouped**: one note per clinical visit (1.6 codes/note avg, more realistic)

---

## Key engineering decisions

**Why TF-IDF + Logistic Regression (not BERT)?**
Per-code F1 analysis showed failures were support-driven — codes with fewer
than 10 training examples scored near zero regardless of model complexity.
The bottleneck was data, not representation. Transformer baseline deferred
until after data expansion.

**Why encounter-grouped notes?**
Patient-grouped notes bundle a lifetime of conditions into one note, creating
an artificially hard multi-label problem. Switching to encounter-grouped
generation doubled dataset size, raised per-code support for rare conditions,
and improved macro F1 from 0.619 to 0.895.

**Why synthetic data?**
Real clinical notes require IRB approval and data use agreements. Synthea +
Gemini gives a complete, reproducible pipeline that demonstrates the full ML
engineering workflow without privacy concerns.

---

## Honest caveats

1. **Synthetic notes are easy for TF-IDF.** Gemini writes "admitted with
   acute bronchitis" which maps trivially to J20. Real clinical notes use
   abbreviations, negation, copy-paste boilerplate, and implicit diagnoses.
   Production performance would be significantly lower.

2. **Synthea has a narrow condition distribution.** Some ICD-10 codes
   (F32 depression, G44 headache, R11 nausea) appear in fewer than 10
   encounters in the synthetic population and remain unlearnable.

3. **The encounter model solves an easier task.** 1.6 codes/note vs 4.2
   means fewer predictions per note. The micro F1 improvement reflects
   both better data and a simpler target distribution.

---

## How to run

### Prerequisites
- Python 3.11
- GCP project with BigQuery and Vertex AI enabled
- `gcloud auth application-default login`

```bash
git clone https://github.com/rajusubbaprojects/medical-coding-ml
cd medical-coding-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Demo
```bash
# Predict from a note file
python -m src.demo.predict note.txt

# Use patient-grouped model
python -m src.demo.predict note.txt --model patient

# Lower confidence threshold
python -m src.demo.predict note.txt --threshold 0.3

# Read from stdin
echo "Patient admitted with acute bronchitis..." | python -m src.demo.predict -
```

### Retrain and evaluate
```bash
python -m src.baseline.train --rollup 3char
python -m src.baseline.run_encounter_cv
python -m src.baseline.evaluate --rollup 3char --bucket test
python -m src.baseline.per_code_analysis --min-support 10
```

---

## Findings log

| Day | Work | Headline |
|-----|------|----------|
| 5 | LLM note synthesis | 100 notes, $0.18 |
| 6 | First baseline | Micro F1 0.685 |
| 7 | Stratified CV | 0.677 +/- 0.032 |
| 8 | Scale to 1,000 notes | 0.821 +/- 0.019 |
| 9 | Per-code F1 analysis | Failure is support-driven |
| 10 | Patient test eval | 0.826 held-out |
| 11 | 2,000 encounter notes | 0.972 +/- 0.005 |
| 12 | Encounter test eval | 0.981 held-out |
| 13 | CLI demo + README | python -m src.demo.predict |

---

## Infrastructure

- **Cloud**: GCP (BigQuery + Vertex AI)
- **LLM**: Gemini 2.5 Flash (~$3.50 per 2,000 notes)
- **ML**: scikit-learn (TF-IDF, LogisticRegression, MultiLabelBinarizer)
- **Data**: Synthea 1,000-patient synthetic EHR dataset
- **Versioning**: GitHub + joblib model bundles

---

## What is next

- Transformer baseline (DistilBERT vs TF-IDF) on encounter dataset
- Streamlit demo app
- Real clinical note evaluation (MIMIC-III)
- Blog post: From Synthea to ICD-10: Building a Medical Coding ML Pipeline
