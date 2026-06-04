"""One-shot script: build dev/test split for encounter-generated notes.

Reads encounter_ids from notes_synth where strategy='encounter-multilabel',
splits 80/20 by encounter_id (not patient_id — encounters are the unit here).
Writes data/splits/encounter_split.json.

Run once after generation completes. Safe to re-run only if you want to
rebuild the split (invalidates any existing CV results).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from google.cloud import bigquery

PROJECT = "medical-coding-ml-9848"
DATASET = "medical_coding"
SPLIT_PATH = Path("data/splits/encounter_split.json")
SEED = 42
TEST_FRACTION = 0.20


def main() -> None:
    client = bigquery.Client(project=PROJECT)

    query = f"""
        SELECT DISTINCT encounter_id
        FROM `{PROJECT}.{DATASET}.notes_synth`
        WHERE strategy = 'encounter-multilabel'
          AND encounter_id IS NOT NULL
        ORDER BY encounter_id
    """
    rows = list(client.query(query).result())
    encounter_ids = [r.encounter_id for r in rows]
    print(f"Found {len(encounter_ids)} encounter notes in BQ.")

    rng = random.Random(SEED)
    shuffled = encounter_ids[:]
    rng.shuffle(shuffled)

    n_test = int(len(shuffled) * TEST_FRACTION)
    test_ids = set(shuffled[:n_test])
    dev_ids = set(shuffled[n_test:])

    print(f"Split: {len(dev_ids)} dev / {len(test_ids)} test")

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps({
        "format": "encounter_ids",
        "seed": SEED,
        "test_fraction": TEST_FRACTION,
        "n_dev": len(dev_ids),
        "n_test": len(test_ids),
        "dev": sorted(dev_ids),
        "test": sorted(test_ids),
    }, indent=2))
    print(f"Wrote {SPLIT_PATH}")


if __name__ == "__main__":
    main()
