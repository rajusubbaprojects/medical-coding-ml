"""Upload Synthea CSV output to GCS.

Usage:
    python data/upload_to_gcs.py
    python data/upload_to_gcs.py --run-dir data/synthea/output/csv/2026_05_20T13_18_03Z --version v1
    python data/upload_to_gcs.py --dry-run

By default, uploads the most recent Synthea run to gs://<DATA_BUCKET>/synthea/v1/.
Set DATA_BUCKET via .env or environment variable.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from google.cloud import storage


SYNTHEA_OUTPUT_ROOT = Path("data/synthea/output/csv")
DEFAULT_VERSION = "v1"


def find_latest_run() -> Path:
    """Return the most recent Synthea CSV run directory."""
    if not SYNTHEA_OUTPUT_ROOT.exists():
        raise FileNotFoundError(
            f"No Synthea output found at {SYNTHEA_OUTPUT_ROOT}. "
            "Generate data first with: java -jar synthea-with-dependencies.jar -c synthea.properties -p N"
        )

    runs = sorted(SYNTHEA_OUTPUT_ROOT.iterdir(), reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run subdirectories under {SYNTHEA_OUTPUT_ROOT}")

    return runs[0]


def upload_run(run_dir: Path, bucket_name: str, version: str, dry_run: bool = False) -> None:
    """Upload every CSV in run_dir to gs://bucket_name/synthea/<version>/<filename>."""
    csv_files = sorted(run_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Source:      {run_dir}")
    print(f"Destination: gs://{bucket_name}/synthea/{version}/")
    print(f"Files:       {len(csv_files)}")
    print()

    if dry_run:
        for csv in csv_files:
            size_mb = csv.stat().st_size / 1_048_576
            print(f"  [dry-run] would upload {csv.name} ({size_mb:.1f} MB)")
        return

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for csv in csv_files:
        blob_path = f"synthea/{version}/{csv.name}"
        blob = bucket.blob(blob_path)
        size_mb = csv.stat().st_size / 1_048_576
        print(f"  uploading {csv.name} ({size_mb:.1f} MB) -> gs://{bucket_name}/{blob_path}")
        blob.upload_from_filename(str(csv))

    print()
    print(f"Done. Uploaded {len(csv_files)} files to gs://{bucket_name}/synthea/{version}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Synthea run directory to upload. Defaults to the most recent one under data/synthea/output/csv/.",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"Version prefix in GCS (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("DATA_BUCKET"),
        help="GCS bucket name. Defaults to $DATA_BUCKET env var.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading.",
    )
    args = parser.parse_args()

    if not args.bucket:
        print(
            "ERROR: bucket not specified. Set DATA_BUCKET in .env or pass --bucket.",
            file=sys.stderr,
        )
        sys.exit(1)

    run_dir = args.run_dir or find_latest_run()
    upload_run(run_dir=run_dir, bucket_name=args.bucket, version=args.version, dry_run=args.dry_run)


if __name__ == "__main__":
    main()