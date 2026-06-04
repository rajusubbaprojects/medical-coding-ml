from pathlib import Path

Path("src/notes/select_encounters.py").write_text(
"""\"\"\"Select encounters + their billable condition lists for note synthesis.

Unlike select_patients.py (which groups all conditions per patient lifetime),
this module groups conditions per encounter. Each encounter becomes one note.
This increases dataset size and improves per-code support for rare conditions.

Filters:
  - Z-codes excluded (same as notes_labels_clean views)
  - Encounters with fewer than min_conditions non-Z billable codes excluded
  - Capped at max_conditions codes per encounter
\"\"\"

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

from google.cloud import bigquery

PROJECT = "medical-coding-ml-9848"
DATASET = "medical_coding"
VIEW = "conditions_billable"
MIN_CONDITIONS = 2
MAX_CONDITIONS_PER_ENCOUNTER = 10
DEFAULT_N_ENCOUNTERS = 200


def fetch_encounters(
    n_encounters: int = DEFAULT_N_ENCOUNTERS,
    min_conditions: int = MIN_CONDITIONS,
    max_conditions: int = MAX_CONDITIONS_PER_ENCOUNTER,
) -> list[dict[str, Any]]:
    \"\"\"Return a list of encounter dicts with their condition lists.

    Each dict has keys: encounter_id, patient_id, conditions (list).
    Conditions are non-Z codes only, sorted by start_date DESC, capped at max_conditions.
    \"\"\"
    client = bigquery.Client(project=PROJECT)

    query = f\"\"\"
    WITH eligible AS (
      SELECT encounter_id, patient_id
      FROM `{PROJECT}.{DATASET}.{VIEW}`
      WHERE NOT STARTS_WITH(icd10_code, 'Z')
      GROUP BY encounter_id, patient_id
      HAVING COUNT(DISTINCT icd10_code) >= @min_conditions
      ORDER BY FARM_FINGERPRINT(encounter_id)
      LIMIT @n_encounters
    ),
    deduped AS (
      SELECT
        c.encounter_id,
        c.patient_id,
        c.icd10_code,
        ANY_VALUE(c.icd10_description)  AS icd10_description,
        ANY_VALUE(c.snomed_code)        AS snomed_code,
        ANY_VALUE(c.snomed_description) AS snomed_description,
        MIN(c.condition_start_date)     AS start_date,
        MAX(c.condition_stop_date)      AS stop_date
      FROM `{PROJECT}.{DATASET}.{VIEW}` c
      JOIN eligible USING (encounter_id, patient_id)
      WHERE NOT STARTS_WITH(c.icd10_code, 'Z')
      GROUP BY c.encounter_id, c.patient_id, c.icd10_code
    )
    SELECT *
    FROM deduped
    ORDER BY encounter_id, start_date DESC
    \"\"\"

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("min_conditions", "INT64", min_conditions),
            bigquery.ScalarQueryParameter("n_encounters", "INT64", n_encounters),
        ]
    )

    rows = client.query(query, job_config=job_config).result()

    grouped: dict[str, dict[str, Any]] = {}
    conditions_by_enc: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        eid = row.encounter_id
        if eid not in grouped:
            grouped[eid] = {
                "encounter_id": eid,
                "patient_id": row.patient_id,
            }
        conditions_by_enc[eid].append({
            "icd10_code": row.icd10_code,
            "icd10_description": row.icd10_description,
            "snomed_code": row.snomed_code,
            "snomed_description": row.snomed_description,
            "start_date": str(row.start_date) if row.start_date else None,
            "stop_date": str(row.stop_date) if row.stop_date else None,
            "encounter_id": eid,
        })

    return [
        {
            **grouped[eid],
            "conditions": conditions_by_enc[eid][:max_conditions],
        }
        for eid in grouped
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=DEFAULT_N_ENCOUNTERS)
    parser.add_argument("--min-conditions", type=int, default=MIN_CONDITIONS)
    parser.add_argument("--max-conditions", type=int, default=MAX_CONDITIONS_PER_ENCOUNTER)
    parser.add_argument("--dump", action="store_true")
    args = parser.parse_args()

    encounters = fetch_encounters(
        n_encounters=args.limit,
        min_conditions=args.min_conditions,
        max_conditions=args.max_conditions,
    )

    if args.dump:
        print(json.dumps(encounters[:3], indent=2))
        return

    print(f"Fetched {len(encounters)} encounters.")
    for e in encounters[:3]:
        print(f"\\nEncounter {e['encounter_id'][:8]} (patient {e['patient_id'][:8]}): "
              f"{len(e['conditions'])} conditions")
        for c in e["conditions"][:5]:
            print(f"  {c['icd10_code']:8s} {c['start_date']}  {c['icd10_description']}")


if __name__ == "__main__":
    main()
"""
)
print("wrote src/notes/select_encounters.py")
