"""Select patients + their billable condition lists for note synthesis.

Queries the `conditions_billable` view and returns one dict per patient
with their most-recent N conditions, sorted by start_date DESC.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

from google.cloud import bigquery

PROJECT = "medical-coding-ml-9848"
DATASET = "medical_coding"
VIEW = "conditions_billable"
MIN_CONDITIONS = 3
MAX_CONDITIONS_PER_PATIENT = 20
DEFAULT_N_PATIENTS = 50


def fetch_patients(n_patients: int = DEFAULT_N_PATIENTS,
                   min_conditions: int = MIN_CONDITIONS,
                   max_conditions: int = MAX_CONDITIONS_PER_PATIENT,
                   ) -> list[dict[str, Any]]:
    """Return a list of patient dicts with their condition lists."""
    client = bigquery.Client(project=PROJECT)

    query = f"""
    WITH eligible AS (
      SELECT patient_id
      FROM `{PROJECT}.{DATASET}.{VIEW}`
      GROUP BY patient_id
      HAVING COUNT(DISTINCT icd10_code) >= @min_conditions
      ORDER BY FARM_FINGERPRINT(patient_id)
      LIMIT @n_patients
    ),
    deduped AS (
      SELECT
        c.patient_id,
        c.icd10_code,
        ANY_VALUE(c.icd10_description)   AS icd10_description,
        ANY_VALUE(c.snomed_code)         AS snomed_code,
        ANY_VALUE(c.snomed_description)  AS snomed_description,
        MIN(c.condition_start_date)      AS start_date,
        MAX(c.condition_stop_date)       AS stop_date,
        ANY_VALUE(c.encounter_id)        AS encounter_id
      FROM `{PROJECT}.{DATASET}.{VIEW}` c
      JOIN eligible USING (patient_id)
      GROUP BY c.patient_id, c.icd10_code
    )
    SELECT *
    FROM deduped
    ORDER BY patient_id, start_date DESC
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("min_conditions", "INT64", min_conditions),
            bigquery.ScalarQueryParameter("n_patients", "INT64", n_patients),
        ]
    )

    rows = client.query(query, job_config=job_config).result()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.patient_id].append({
            "icd10_code": row.icd10_code,
            "icd10_description": row.icd10_description,
            "snomed_code": row.snomed_code,
            "snomed_description": row.snomed_description,
            "start_date": str(row.start_date) if row.start_date else None,
            "stop_date": str(row.stop_date) if row.stop_date else None,
            "encounter_id": row.encounter_id,
        })

    return [
        {"patient_id": pid, "conditions": conds[:max_conditions]}
        for pid, conds in grouped.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=DEFAULT_N_PATIENTS)
    parser.add_argument("--min-conditions", type=int, default=MIN_CONDITIONS)
    parser.add_argument("--max-conditions", type=int, default=MAX_CONDITIONS_PER_PATIENT)
    parser.add_argument("--dump", action="store_true")
    args = parser.parse_args()

    patients = fetch_patients(n_patients=args.limit,
                              min_conditions=args.min_conditions,
                              max_conditions=args.max_conditions)

    if args.dump:
        print(json.dumps(patients, indent=2))
        return

    print(f"Fetched {len(patients)} patients "
          f"(capped at {args.max_conditions} conditions each).")
    for p in patients[:3]:
        print(f"\nPatient {p['patient_id']}: {len(p['conditions'])} conditions")
        for c in p["conditions"][:5]:
            print(f"  {c['icd10_code']:8s} {c['start_date']}  {c['icd10_description']}")
        if len(p["conditions"]) > 5:
            print(f"  ... and {len(p['conditions']) - 5} more")


if __name__ == "__main__":
    main()
