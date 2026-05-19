# ---------------------------------------------------------------------------
# Service accounts
# ---------------------------------------------------------------------------
resource "google_service_account" "training" {
  account_id   = "vertex-training"
  display_name = "Vertex AI training jobs"
}

resource "google_service_account" "serving" {
  account_id   = "cloud-run-serving"
  display_name = "Cloud Run serving"
}

resource "google_service_account" "github_actions" {
  account_id   = "github-actions"
  display_name = "GitHub Actions deploy"
}

# ---------------------------------------------------------------------------
# IAM bindings — what each service account is allowed to do
# ---------------------------------------------------------------------------

# Training SA needs Vertex AI, BigQuery, GCS, Artifact Registry
resource "google_project_iam_member" "training_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/bigquery.dataViewer",
    "roles/bigquery.jobUser",
    "roles/storage.objectAdmin",
    "roles/artifactregistry.writer",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.training.email}"
}

# Serving SA needs to read models from registry, write metrics
resource "google_project_iam_member" "serving_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/storage.objectViewer",
    "roles/monitoring.metricWriter",
    "roles/logging.logWriter",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.serving.email}"
}

# GitHub Actions SA needs to deploy
resource "google_project_iam_member" "github_actions_roles" {
  for_each = toset([
    "roles/run.admin",
    "roles/artifactregistry.writer",
    "roles/iam.serviceAccountUser",
    "roles/cloudbuild.builds.editor",
  ])
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.github_actions.email}"
}

# ---------------------------------------------------------------------------
# Workload Identity Federation — GitHub OIDC, no static keys
# ---------------------------------------------------------------------------
resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github-pool"
  display_name              = "GitHub Actions"
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.actor"      = "assertion.actor"
  }

  attribute_condition = "assertion.repository == \"${var.github_repo}\""

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# Bind the GitHub repo to the GitHub Actions service account
resource "google_service_account_iam_member" "github_wif" {
  service_account_id = google_service_account.github_actions.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repo}"
}

# ---------------------------------------------------------------------------
# Outputs — paste these into GitHub repo secrets after apply
# ---------------------------------------------------------------------------
output "github_actions_sa_email" {
  value       = google_service_account.github_actions.email
  description = "Set as GHA_SERVICE_ACCOUNT in GitHub Actions secrets"
}

output "workload_identity_provider" {
  value       = google_iam_workload_identity_pool_provider.github.name
  description = "Set as WIF_PROVIDER in GitHub Actions secrets"
}

output "project_id" {
  value       = var.project_id
  description = "Set as PROJECT_ID in GitHub Actions secrets"
}