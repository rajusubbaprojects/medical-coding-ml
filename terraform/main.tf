terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ---------------------------------------------------------------------------
# APIs — declared here so they're re-enabled if disabled
# ---------------------------------------------------------------------------
locals {
  apis = [
    "aiplatform.googleapis.com",
    "bigquery.googleapis.com",
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudfunctions.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "iam.googleapis.com",
    "storage.googleapis.com",
    "iamcredentials.googleapis.com",
  ]
}

resource "google_project_service" "apis" {
  for_each = toset(local.apis)
  service  = each.value

  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# GCS buckets
# ---------------------------------------------------------------------------
resource "google_storage_bucket" "data" {
  name          = "${var.project_id}-data"
  location      = var.region
  storage_class = "STANDARD"

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
}

resource "google_storage_bucket" "artifacts" {
  name          = "${var.project_id}-artifacts"
  location      = var.region
  storage_class = "STANDARD"

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
}

# ---------------------------------------------------------------------------
# BigQuery dataset
# ---------------------------------------------------------------------------
resource "google_bigquery_dataset" "medical_coding" {
  dataset_id  = "medical_coding"
  location    = var.region
  description = "Clinical notes and ICD/CPT code labels"
}

# ---------------------------------------------------------------------------
# Artifact Registry for container images
# ---------------------------------------------------------------------------
resource "google_artifact_registry_repository" "main" {
  location      = var.region
  repository_id = "medical-coding"
  description   = "Container images for training and serving"
  format        = "DOCKER"

  depends_on = [google_project_service.apis]
}