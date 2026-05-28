from pathlib import Path

CONTENT = '''"""BigQuery sink for generated notes.

Creates two tables (idempotent) and writes batches via load jobs:
  - notes_synth:  one row per note (with ARRAY<STRING> labels)
  - notes_labels: one row per (note_id, code) pair
"""

from __future__ import annotations

import logging
from typing import Any

from google.cloud import bigquery

PROJECT = "medical-coding-ml-9848"
DATASET = "medical_coding"
NOTES_TABLE = "notes_synth"
LABELS_TABLE = "notes_labels"

logger = logging.getLogger(__name__)

_client: bigquery.Client | None = None


def get_client() -> bigquery.Client:
    global _client
    if _client is None:
        _client = bigquery.Client(project=PROJECT)
    return _client


NOTES_SCHEMA = [
    bigquery.SchemaField("note_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("patient_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("strategy", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("prompt_version", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("model", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("generated_at", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("note_text", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("labels", "STRING", mode="REPEATED"),
    bigquery.SchemaField("primary_code", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("input_tokens", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("output_tokens", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("finish_reason", "STRING", mode="NULLABLE"),
]

LABELS_SCHEMA = [
    bigquery.SchemaField("note_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("icd10_code", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("icd10_description", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("is_primary", "BOOL", mode="REQUIRED"),
]


def ensure_tables() -> None:
    """Create both tables if they don't exist. Safe to call repeatedly."""
    client = get_client()
    dataset_ref = bigquery.DatasetReference(PROJECT, DATASET)

    for name, schema, partition_field in [
        (NOTES_TABLE, NOTES_SCHEMA, "generated_at"),
        (LABELS_TABLE, LABELS_SCHEMA, None),
    ]:
        table_ref = dataset_ref.table(name)
        table = bigquery.Table(table_ref, schema=schema)
        if partition_field:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field=partition_field,
            )
        client.create_table(table, exists_ok=True)
        logger.info("Ensured table: %s.%s.%s", PROJECT, DATASET, name)


def insert_notes(notes_rows: list[dict[str, Any]],
                 labels_rows: list[dict[str, Any]]) -> None:
    """Insert via load job (cheaper than streaming, fine for our batch sizes)."""
    client = get_client()

    def _load(rows: list[dict[str, Any]], table: str, schema: list[bigquery.SchemaField]) -> None:
        if not rows:
            logger.warning("No rows to load for %s", table)
            return
        table_id = f"{PROJECT}.{DATASET}.{table}"
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        )
        job = client.load_table_from_json(rows, table_id, job_config=job_config)
        job.result()  # wait
        logger.info("Loaded %d rows into %s", len(rows), table_id)

    _load(notes_rows, NOTES_TABLE, NOTES_SCHEMA)
    _load(labels_rows, LABELS_TABLE, LABELS_SCHEMA)


def main() -> None:
    """Smoke test: ensure tables exist, then describe them."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_tables()

    client = get_client()
    for name in [NOTES_TABLE, LABELS_TABLE]:
        table = client.get_table(f"{PROJECT}.{DATASET}.{name}")
        print(f"\\n{name}: {table.num_rows} rows, {len(table.schema)} columns")
        for f in table.schema:
            print(f"  {f.name:18s} {f.field_type:10s} {f.mode}")


if __name__ == "__main__":
    main()
'''

Path("src/notes/load_to_bq.py").write_text(CONTENT)
print("Wrote src/notes/load_to_bq.py")
print(f"  {len(CONTENT.splitlines())} lines")