# Notes — building the medical coding ML system

A running log of decisions, blockers, and lessons. The blog post will be assembled from these notes at the end of Week 4.

## Format

Each entry: date, what happened, why it matters.

---

## 2026-05-18 — Day 1: GCP foundation

**What:** Created GCP project `medical-coding-ml-9848`, linked billing to new $300/90-day credit account, enabled 11 APIs, set $50/month budget alert.

**Why it matters:** Infrastructure plumbing is its own engineering problem, separate from ML. Recruiters underestimate this; production ML engineers don't.

**Snags hit:**
- Previous billing account was closed (residue from an old trial). Created a new one.
- ADC requires explicit consent on the cloud-platform scope. Failed the first attempt because I didn't check all the boxes.
- Pasted a placeholder billing account ID literally instead of substituting my real one. Caught by the API.
- Budget API couldn't see the new project for ~15 min after creation. Worked around by creating the budget at the billing-account level. GCP services have eventual consistency between each other.

**Lessons:**
1. `gcloud auth login` and `gcloud auth application-default login` are separate. Both required.
2. CLI confirmations don't guarantee the UI is ready — check `OPEN: True` on billing accounts before using.
3. Budget alerts email; they don't cap spend. Discipline is reacting to alerts.

---
---

## 2026-05-18 — Day 1 evening: Pivoted from MIMIC-IV to Synthea synthetic data

**What:** Started CITI registration for PhysioNet credentialed access to MIMIC-IV, discovered the Independent Learner path costs ~$165 (CITI Data or Specimens Only Research course + HIPAA module). Pivoted to using Synthea (MITRE's open-source synthetic patient generator) instead.

**Why this is actually a better decision for the project:**
1. **Time to first data: minutes, not 10 days.** Synthea runs locally, generates as many patients as you want, ships with ICD-10 codes.
2. **No PHI handling risk.** The repo can be fully public from day one. Notebooks, EDA outputs, sample predictions — all safe to commit.
3. **Stronger data-governance story for interviews.** "I designed the system to use synthetic data so the public repo doesn't touch PHI, with a swap-in path to real EHR data via a config change." That's a more sophisticated narrative than "I worked with MIMIC."
4. **Cost goes to compute, not credentialing.** $165 reallocated to Vertex AI training jobs.

**Trade-offs accepted:**
- Synthea generates structured encounter data, not free-text notes. We'll synthesize notes from the structured data using an LLM. This becomes its own interesting subsystem (and a great section of the blog post on "how to bootstrap clinical NLP without real notes").
- "Trained on MIMIC" looks better on a resume to clinical-ML hiring managers than "trained on Synthea." Trade I'm willing to make for time-to-shipped.

**System design implication:** Build the data ingestion layer with a clean abstraction so MIMIC-IV can be swapped in later without changing training code.

## Decision log

Track decisions where the trade-off is interesting enough to write about.

### Decision: BigQuery vs Cloud SQL for note storage
- Chose BigQuery
- Why: query-heavy analytical workload, columnar storage suits text, free tier is generous
- Alternative considered: Cloud SQL Postgres — better for transactional writes, but we don't need them

### Decision: Cloud Run vs Vertex AI Endpoints for serving
- Chose Cloud Run
- Why: canary traffic splitting is built in via revisions, cheaper at low QPS, portable container
- Alternative: Vertex AI Endpoints — better for GPU inference, autoscaling per model, but pricier

### Decision: Workload Identity Federation vs service account keys for GitHub Actions
- Chose WIF
- Why: no long-lived secrets to rotate, this is how production teams do it
- Alternative: service account JSON key — simpler to set up, but a security anti-pattern

---

## Blockers

Things that took longer than expected — gold for the blog post.

### Blocker: closed billing account on a fresh GCP account
- What broke: `gcloud beta billing accounts list` showed `OPEN: False`
- How long to fix: ~15 min
- The fix: created a new billing account in the console, got the $300 credit
- What I'd do differently: check the OPEN column before assuming auth is the problem

---

## Surprises

Things I didn't expect — also gold.

### Surprise: building the API skeleton before the model
- Production-style MLOps inverts the typical learning order. Usually you start with a notebook, train a model, then think about serving. With a stub Predictor returning hardcoded codes, the API can be live on day one and the model swaps in later. The platform is the product; the model is just a swappable component.

**Started CITI training in parallel during day-1 setup.** Critical path: PhysioNet approval is 3-10 days; can't start data work until then. Working on infrastructure while waiting.


---

## 2026-05-19 — Day 2: Infrastructure live

**What:** `terraform apply` created 34 resources cleanly: 11 API enablements, 3 service accounts (training/serving/GitHub Actions), 13 IAM bindings, 2 GCS buckets, 1 BigQuery dataset, 1 Artifact Registry repo, 1 Workload Identity Pool, 1 provider, 1 WIF binding.

**Why it matters:** The platform exists. From here, every piece of code I write has a real place to run. The project is no longer "files in git" — it's a system with cloud presence.

**Workload Identity Federation worked first try.** No JSON service account keys stored anywhere. The OIDC trust between this repo and GCP is established by Terraform; auth at deploy time is short-lived tokens. This is the modern standard and a strong interview talking point.

**Lessons:**
- `terraform plan` showed exactly 34 resources matching my mental model — no surprises is a good sign. When plan shows resources you didn't expect, your code disagrees with your intent.
- The `for_each` pattern over a list of roles makes IAM bindings dense and readable. Adding a permission is one line in a list.
- Lock file (`.terraform.lock.hcl`) committed for reproducibility — anyone else who clones the repo will pin to provider v5.45.2 the same way I am.
- Updated `.gitignore` to allow committing `.terraform.lock.hcl` — industry practice moved toward committing this file for reproducibility.

**Synthea setup (evening):** Installed OpenJDK 17 (system was on Java 8). Downloaded Synthea 188MB JAR. Smoke test: 1 patient generated successfully — "Kamala553 Conroy74", 21-y/o female from Harvard MA, complete simulated medical history in 2.2MB of FHIR JSON. By default Synthea outputs FHIR only; CSV will need a config flag flip tomorrow before bulk generation.

**Notable:** Java 17 throws a deprecation warning about `sun.reflect` — harmless, Synthea was written for older Java. Each patient generates ~2.2MB of FHIR JSON, so 1k patients ≈ 2GB. CSV output will be much smaller and BigQuery-friendly.

---

## 2026-05-20 — Day 3: Synthetic data exists, code distribution surprises

**What:** Generated 1,125 patients with Synthea. 40k conditions, 65k encounters, 209MB total CSV.

**The surprise:** Synthea outputs SNOMED CT, not ICD-10, and most top codes aren't billable diagnoses at all:
- 7,844 "Medication review due (situation)" — admin code
- 2,841 "Full-time employment (finding)" — SDOH
- 1,114 "Social isolation (finding)" — SDOH
- Real billable codes (gingivitis, sinusitis, obesity) are further down

**Why it matters:** This isn't a flaw in Synthea — it's accurate to real EHR data. SNOMED has 4 broad categories (situation, finding, disorder, procedure), only `disorder` and `procedure` map cleanly to ICD-10 billing codes. This means the clinical NLP problem has a filtering step built in: the ML pipeline needs to recognize which codes are billable, not just predict any code present in the notes.

**Plan:** Don't try to make Synthea output ICD-10. Instead, map SNOMED → ICD-10 at ingest time using NLM's public mapping table. Codes with no ICD-10 equivalent (most "situation" and many "finding" codes) drop out — exactly what hospital coders do.

**Vocabulary size:** 262 distinct SNOMED codes in this dataset. After ICD-10 filtering tomorrow, expect ~80-150 billable codes — a manageable multi-label problem.

---

## 2026-05-20 — Day 3: Data pipeline complete

**What:** Generated 1,125 synthetic patients with Synthea (config file: CSV exporter on, FHIR off, 6 tables included). Wrote `data/upload_to_gcs.py` to push the timestamped run folder to `gs://medical-coding-ml-9848-data/synthea/v1/`. Wrote `data/bigquery_setup.py` to create 6 external tables in the `medical_coding` BigQuery dataset. First query (top-20 conditions) ran successfully against the data in BigQuery.

**Numbers:**
- 1,125 patients (asked for 1,000; Synthea generates overflow to compensate for early deaths)
- 40,071 conditions across 262 distinct SNOMED codes
- 65,326 encounters
- 209 MiB total CSV data
- 38 seconds to upload to GCS
- BigQuery queries run sub-2-second against external tables

**Why external tables, not native:** External tables query CSV directly from GCS without copying data. Slower per query than native tables but zero BigQuery storage cost and trivial to update — generate `v2/`, repoint table, done. Will switch to native tables when the schema is stable, for query speed during training.

**Two design patterns introduced today:**
1. **Versioned data paths** (`synthea/v1/`, `synthea/v2/`). Every regeneration gets a new version. Easy rollback, easy A/B between dataset versions.
2. **Idempotent scripts.** `upload_to_gcs.py` and `bigquery_setup.py` both support being run multiple times safely. `bigquery_setup.py` uses `exists_ok=True`; `upload_to_gcs.py` overwrites existing GCS blobs cleanly. This matters for production-style CI/CD where you can't assume "first run."

**The ICD-10 question.** Synthea outputs SNOMED CT. Configured `exporter.use_icd10_codes = true` but it didn't change the CSV `SYSTEM` column (still SNOMED-CT). Decided to handle SNOMED→ICD-10 mapping at ingest time tomorrow using NLM's public mapping table. This is also the realistic production pattern: EHR systems are SNOMED-native; the billing layer translates to ICD-10. Better story than fighting Synthea's config.

**Top SNOMED codes in our data are revealing:**
- Most frequent codes are administrative (`Medication review due` 7,844) and social determinants of health (`Full-time employment` 2,841, `Social isolation` 1,114)
- Actual billable diagnoses (Gingivitis, Viral sinusitis, Obesity) appear further down
- This is accurate to real EHR data. The ICD-10 mapping step will naturally filter administrative codes since they have no billable equivalent.

---

## 2026-05-20 — Portfolio shape decision

**Question that came up today:** Project 1 on AWS wasn't Terraformed. Should I retrofit it?

**Decision:** No. Instead, deliberate split across 4 projects:
- Projects 1, 4: no Terraform (rapid prototypes, model-focused)
- Projects 2, 3: with Terraform (production MLOps, infrastructure-focused)

**Why this is stronger than "Terraform everywhere":** Most ML engineers can ship a model OR build production infra, not both. A portfolio showing both modalities — knowing when to reach for IaC and when to ship fast — is more senior-signal than uniform projects. Each project has a role.

**Framing for resume:** Projects 1 and 4 are "rapid prototype" pieces. Projects 2 and 3 are "production MLOps" pieces. Title each accordingly so reviewers see the deliberate split.

---

---

## 2026-05-21 — Day 4: Adding the ICD-10 mapping layer

**Goal:** Synthea outputs SNOMED CT codes. ML model should predict ICD-10 (what hospitals bill on). Need a SNOMED→ICD-10 mapping layer.

**Tool research:**
- **UMLS** (NLM) is the authoritative source. Requires manual review, 3 business days.
- **OHDSI Athena** is the production-grade open alternative. Instant signup, mapping built on top of UMLS + OHDSI curation. Well-respected in clinical research.

**Decision:** Pivot to Athena. Don't wait 3 days. When UMLS approval lands later, we can swap mapping sources — the downstream SQL view is unchanged because the abstraction holds. (Bonus: "I migrated vocabulary sources mid-project without breaking downstream code" is itself a story for the blog.)

**Vocabulary selection:**
- SNOMED — source (matches what Synthea produces)
- ICD10CM — target (US clinical modification, NCHS-maintained, what billers use)
- Skipped: CPT4 (paid AMA license), MedDRA (paid MSSO license); RxNorm/LOINC (not needed for diagnosis prediction)
- Latest releases used: SNOMED 28-Feb-2025, ICD10CM 30-Sep-2025

**Athena download submitted:** 10:17 AM, PENDING. Expected ~500MB–1GB ZIP containing CONCEPT.csv, CONCEPT_RELATIONSHIP.csv, and ~6 reference tables. Build time 5–30 min.

**Plan when ZIP arrives:**
1. Extract locally
2. Upload to GCS at `vocabularies/v1/`
3. Load into BigQuery as **native tables** (not external — reference data, queried in every training run, deserves columnar storage)
4. Write SQL view `conditions_billable` joining `conditions` → SNOMED concept → ICD10CM concept
5. EDA on the billable code distribution

**Architectural rule learned today: native vs external tables.** Synthea data → external (versioned, swappable, regenerated). Vocabulary data → native (stable reference, queried often). Split by access pattern, not by file size.

**Tooling gotcha:** Athena Arachne ≠ Athena vocabularies. Two different OHDSI tools with similar branding. Went to the wrong one first. (`athena.ohdsi.org` is the vocabulary tool, not `arachnenetwork.com`.)

