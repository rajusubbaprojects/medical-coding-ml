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

