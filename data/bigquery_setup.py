"""Create BigQuery external tables pointing at Synthea CSVs in GCS.

Run this after upload_to_gcs.py. The external tables let you query the CSVs
directly without copying data into BigQuery storage. Schema is inferred from
the CSV headers.

Usage:
    python data/bigquery_setup.py
    python data/bigquery_setup.py --version v1
    python data/bigquery_setup.py --recreate   # drop and recreate tables
"""

from __future__ import annotations

import argparse
import os
import sys

from google.cloud import bigquery


DEFAULT_VERSION = "v1"

# Map: table name in BigQuery -> CSV filename in GCS
TABLES = {
    "patients": "patients.csv",
    "encounters": "encounters.csv",
    "conditions": "conditions.csv",
    "procedures": "procedures.csv",
    "medications": "medications.csv",
    "observations": "observations.csv",
}


def create_external_table(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    table_name: str,
    gcs_uri: str,
    recreate: bool = False,
) -> None:
    """Create one external table backed by a GCS CSV."""
    table_id = f"{project_id}.{dataset_id}.{table_name}"

    if recreate:
        client.delete_table(table_id, not_found_ok=True)
        print(f"  dropped {table_id}")

    external_config = bigquery.ExternalConfig("CSV")
    external_config.source_uris = [gcs_uri]
    external_config.options.skip_leading_rows = 1
    external_config.options.allow_quoted_newlines = True
    external_config.autodetect = True  # infer schema from header + sample rows

    table = bigquery.Table(table_id)
    table.external_data_configuration = external_config

    table = client.create_table(table, exists_ok=True)
    print(f"  created external table {table_id} -> {gcs_uri}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"GCS version prefix to read from (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("PROJECT_ID"),
        help="GCP project ID. Defaults to $PROJECT_ID env var.",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("BQ_DATASET", "medical_coding"),
        help="BigQuery dataset name. Defaults to $BQ_DATASET or 'medical_coding'.",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("DATA_BUCKET"),
        help="GCS bucket name. Defaults to $DATA_BUCKET env var.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop existing tables before creating (useful if schema changed).",
    )
    args = parser.parse_args()

    missing = [name for name, val in [("PROJECT_ID", args.project), ("DATA_BUCKET", args.bucket)] if not val]
    if missing:
        print(f"ERROR: missing required env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    client = bigquery.Client(project=args.project)

    print(f"Project: {args.project}")
    print(f"Dataset: {args.dataset}")
    print(f"Source:  gs://{args.bucket}/synthea/{args.version}/")
    print()

    for table_name, csv_file in TABLES.items():
        gcs_uri = f"gs://{args.bucket}/synthea/{args.version}/{csv_file}"
        create_external_table(
            client=client,
            project_id=args.project,
            dataset_id=args.dataset,
            table_name=table_name,
            gcs_uri=gcs_uri,
            recreate=args.recreate,
        )

    print()
    print(f"Done. {len(TABLES)} external tables ready in {args.project}.{args.dataset}.")


if __name__ == "__main__":
    main()