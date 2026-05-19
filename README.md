# Medical Coding ML System

An end-to-end MLOps system that suggests ICD-10 and CPT codes from clinical notes, deployed on GCP. Built to compare three modeling approaches (fine-tuned ClinicalBERT, RAG over code descriptions, and a hybrid retrieval + ranker) and serve the best one with canary deploys, drift detection, and automated retraining.

**Data:** Uses [Synthea](https://github.com/synthetichealth/synthea) (MITRE's open-source synthetic patient generator) to produce realistic clinical data without PHI. Free-text notes are synthesized from Synthea's structured output using an LLM, with the original ICD-10 codes preserved as ground-truth labels. The system is designed to swap in real EHR data (e.g., MIMIC-IV) via configuration; only the ingestion layer changes.

## Architecture

```
Synthea (synthetic patients)
        ↓
LLM note synthesis (Vertex AI)
        ↓
   GCS bucket → BigQuery → Vertex AI Feature Store
                              ↓
                    Vertex AI Pipelines
                    ├── ClinicalBERT (fine-tune)
                    ├── RAG (frontier LLM + code embeddings)
                    └── Hybrid (retrieval + cross-encoder)
                              ↓
                       MLflow tracking
                              ↓
                    Vertex AI Model Registry
                              ↓
              Artifact Registry (container images)
                              ↓
                  Cloud Run (FastAPI, canary split)
                              ↓
        Cloud Monitoring ← logs, latency, errors
                  ↓
       Vertex AI Model Monitoring (drift)
                  ↓
       Cloud Function → triggers retraining
```

## Status

| Component | State |
|---|---|
| GCP project + APIs | ✅ |
| Terraform skeleton | 🔄 |
| Synthea data generation | ⬜ |
| LLM note synthesis | ⬜ |
| BigQuery schema | ⬜ |
| Approach 1: ClinicalBERT | ⬜ |
| Approach 2: RAG | ⬜ |
| Approach 3: Hybrid | ⬜ |
| FastAPI + Cloud Run | ⬜ |
| Canary deploys | ⬜ |
| Drift detection | ⬜ |
| Retraining trigger | ⬜ |
| CI/CD (GitHub Actions) | ⬜ |
| Blog post | ⬜ |

## Quickstart

### Prerequisites

- Google Cloud SDK (`gcloud`) authenticated
- Terraform >= 1.5
- Python 3.11
- Docker
- Java 11+ (for running Synthea)

### Setup

```bash
# 1. Configure
cp .env.example .env
# Edit .env with your PROJECT_ID, REGION, etc.

# 2. Provision infra
cd terraform
terraform init
terraform apply

# 3. Generate synthetic data
cd ../data/synthea
./run_synthea -p 1000   # 1000 synthetic patients

# 4. Synthesize free-text notes and load to BigQuery
python data/ingest.py --source synthea --output gs://${DATA_BUCKET}/notes/

# 5. Run a training pipeline
python training/pipelines/vertex_pipeline.py --approach clinicalbert

# 6. Local serving for testing
cd serving && docker build -t medical-coding-api . && docker run -p 8080:8080 medical-coding-api
```

## Three approaches: why and how

This project deliberately compares three modeling strategies. The blog post writes itself from the comparison.

**Approach 1 — Fine-tuned ClinicalBERT.** Multi-label classifier on the top 100-500 most common codes. Cheap inference (~50ms), but limited code coverage and no native explainability.

**Approach 2 — RAG with frontier LLM.** Embed official ICD-10/CPT code descriptions, retrieve top candidates for each note, have the LLM pick and rank with cited reasoning. Covers the full code set, explainable, but pricier and slower (~1-3s per note).

**Approach 3 — Hybrid retrieval + ranker.** Retrieval narrows ~80k codes to ~20 candidates, then a fine-tuned cross-encoder ranks them. Best of both: low latency, full code coverage, retrieval traces give explainability.

The serving layer ships the winner. The comparison itself is the blog post.

## Data: why synthetic, not real

Real clinical notes (e.g., MIMIC-IV) require institutional credentialing and weeks of approval, and any work with them comes with PHI-handling constraints that complicate having a public portfolio repo. For a portfolio project, Synthea is a better fit:

- **No PHI** — repo, notebooks, sample outputs are all safe to publish
- **Fast iteration** — generate 1k or 100k patients in minutes
- **Ground-truth labels** — Synthea attaches real ICD-10 codes to every condition
- **Realistic distributions** — patients have demographically and epidemiologically grounded conditions and care pathways

The data ingestion layer is designed as a swappable adapter. To switch to MIMIC-IV or any real EHR source later, only the ingestion code changes; downstream training, serving, and evaluation remain identical.

See `docs/data-access.md` for the full data pipeline.

## Project layout

```
medical-coding-ml/
├── README.md
├── NOTES.md                    # running log of decisions, blockers, lessons
├── .github/workflows/          # CI and deploy pipelines
├── terraform/                  # infrastructure as code
├── data/
│   ├── schemas/                # BigQuery table schemas
│   └── synthea/                # Synthea config and run scripts
├── notebooks/                  # EDA, evaluation, ad-hoc analysis
├── training/
│   ├── approach_1_clinicalbert/
│   ├── approach_2_rag/
│   ├── approach_3_hybrid/
│   └── pipelines/              # Vertex AI Pipelines DSL
├── serving/                    # FastAPI app + Dockerfile
├── monitoring/                 # drift detection + Cloud Monitoring dashboards
├── tests/                      # pytest suite
└── docs/                       # architecture and how-to docs
```

## Tech stack

- **Cloud**: Google Cloud Platform (Vertex AI, BigQuery, Cloud Run, Artifact Registry, Cloud Monitoring, Cloud Functions)
- **IaC**: Terraform
- **CI/CD**: GitHub Actions with Workload Identity Federation
- **Serving**: FastAPI, Docker, Cloud Run with canary traffic splits
- **ML**: PyTorch, Transformers (ClinicalBERT), sentence-transformers (embeddings), scikit-learn
- **Experiment tracking**: MLflow on Cloud Run
- **Data**: Synthea, BigQuery, GCS

## Disclaimer

This is a portfolio project, not a clinical tool. The model outputs are not validated for clinical use. All data used is synthetic and contains no real patient information.