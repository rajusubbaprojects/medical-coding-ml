"""Load OHDSI Athena vocabularies into GCS and BigQuery.

Run AFTER you've extracted the Athena vocabulary ZIP locally to
data/vocabularies/<version>/ (containing CONCEPT.csv, CONCEPT_RELATIONSHIP.csv, etc.)

Steps performed:
  1. Upload CONCEPT.csv and CONCEPT_RELATIONSHIP.csv to GCS.
  2. Create native BigQuery tables loading from those GCS files.

Why native (not external) tables: vocabulary data is reference data — stable,
queried in every training run, no benefit to re-parsing CSV from GCS each time.
Synthea data uses external tables (versioned, swappable, regenerated frequently).

Usage:
    python data/load_vocabularies.py
    python data/load_vocabularies.py --version athena_v20250827
    python data/load_vocabularies.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from google.cloud import bigquery, storage


DEFAULT_VERSION = "athena_v20250827"
DEFAULT_LOCAL_ROOT = Path("data/vocabularies")

# Only the files we need for the SNOMED→ICD-10 mapping.
# Athena will produce more (CONCEPT_ANCESTOR, CONCEPT_SYNONYM, etc.), ignored for now.
VOCABULARY_FILES = {
    "concept": "CONCEPT.csv",
    "concept_relationship": "CONCEPT_RELATIONSHIP.csv",
}


def upload_to_gcs(
    local_dir: Path,
    bucket_name: str,
    version: str,
    dry_run: bool = False,
) -> dict[str, str]:
    """Upload vocabulary CSVs to GCS. Returns map of table_name -> gcs_uri."""
    # Pre-flight: verify all files exist before starting any uploads.
    missing = [f for f in VOCABULARY_FILES.values() if not (local_dir / f).exists()]
    if missing:
        print(f"ERROR: required files missing in {local_dir}:", file=sys.stderr)
        for f in missing:
            print(f"  - {f}", file=sys.stderr)
        print("\nExtract the Athena ZIP into this directory first.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[1/2] Uploading to GCS")
    print(f"  Source:      {local_dir}")
    print(f"  Destination: gs://{bucket_name}/vocabularies/{version}/")

    uris = {}
    client = None if dry_run else storage.Client()
    bucket = None if dry_run else client.bucket(bucket_name)

    for table_name, filename in VOCABULARY_FILES.items():
        local_path = local_dir / filename
        size_mb = local_path.stat().st_size / 1_048_576
        gcs_path = f"vocabularies/{version}/{filename}"
        gcs_uri = f"gs://{bucket_name}/{gcs_path}"
        uris[table_name] = gcs_uri

        if dry_run:
            print(f"  [dry-run] would upload {filename} ({size_mb:.1f} MB)")
        else:
            blob = bucket.blob(gcs_path)
            print(f"  uploading {filename} ({size_mb:.1f} MB)...", end="", flush=True)
            blob.upload_from_filename(str(local_path))
            print(" done")

    return uris


def load_to_bigquery(
    project_id: str,
    dataset_id: str,
    table_uris: dict[str, str],
    dry_run: bool = False,
) -> None:
    """Load CSVs from GCS into native BigQuery tables (REPLACE semantics)."""
    print(f"\n[2/2] Loading native tables in BigQuery")
    print(f"  Project: {project_id}")
    print(f"  Dataset: {dataset_id}")

    if dry_run:
        for table_name, uri in table_uris.items():
            print(f"  [dry-run] would load {uri} -> {project_id}.{dataset_id}.{table_name}")
        return

    client = bigquery.Client(project=project_id)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        # Athena exports are TAB-separated, not comma-separated, and use no
        # quoting at all (concept_name fields contain raw quotes that aren't
        # escaped). Setting quote_character="" tells BigQuery to treat every
        # byte between tabs as literal data, no special quote handling.
        field_delimiter="\t",
        quote_character="",
        allow_quoted_newlines=False,
    )

    for table_name, uri in table_uris.items():
        table_id = f"{project_id}.{dataset_id}.{table_name}"
        print(f"  loading {uri}")
        print(f"      -> {table_id}", end="", flush=True)
        load_job = client.load_table_from_uri(uri, table_id, job_config=job_config)
        load_job.result()  # blocks until done
        table = client.get_table(table_id)
        print(f" ({table.num_rows:,} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"Local subdirectory under data/vocabularies/ (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("PROJECT_ID"),
        help="GCP project ID. Defaults to $PROJECT_ID env var.",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("DATA_BUCKET"),
        help="GCS bucket name. Defaults to $DATA_BUCKET env var.",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("BQ_DATASET", "medical_coding"),
        help="BigQuery dataset name. Defaults to $BQ_DATASET or 'medical_coding'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without uploading or loading.",
    )
    args = parser.parse_args()

    if not args.project or not args.bucket:
        print("ERROR: PROJECT_ID and DATA_BUCKET must be set (env or --flags)", file=sys.stderr)
        sys.exit(1)

    local_dir = DEFAULT_LOCAL_ROOT / args.version
    if not local_dir.exists():
        print(f"ERROR: local directory not found: {local_dir}", file=sys.stderr)
        print(f"Extract the Athena ZIP there first.", file=sys.stderr)
        sys.exit(1)

    uris = upload_to_gcs(local_dir, args.bucket, args.version, args.dry_run)
    load_to_bigquery(args.project, args.dataset, uris, args.dry_run)

    print(f"\nDone. Tables in {args.project}.{args.dataset}:")
    for table_name in VOCABULARY_FILES:
        print(f"  {table_name}")


if __name__ == "__main__":
    main()