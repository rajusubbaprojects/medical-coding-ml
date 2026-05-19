variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for regional resources"
  type        = string
  default     = "us-central1"
}

variable "github_repo" {
  description = "GitHub repo in the form owner/repo, used for Workload Identity Federation"
  type        = string
}