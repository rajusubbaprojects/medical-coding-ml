"""One-shot: build a stable dev/test split over the note IDs.

Run once at the start of Day 7. Writes data/splits/dev_test_split.json.
Commit the JSON so every future run uses the same split.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from google.cloud import bigquery
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

PROJECT = "medical-coding-ml-9848"
SEED = 42
TEST_FRACTION = 0.20   # 20% held out
SPLIT_PATH = Path("data/splits/dev_test_split.json")


def main() -> None:
    client = bigquery.Client(project=PROJECT)

    # Pull notes + their 3char labels (joining only the rolled-up clean labels)
    notes = client.query(f"""
        SELECT note_id
        FROM `{PROJECT}.medical_coding.notes_synth`
        ORDER BY note_id
    """).to_dataframe()

    labels = client.query(f"""
        SELECT note_id, label_code
        FROM `{PROJECT}.medical_coding.notes_labels_clean_3char`
    """).to_dataframe()

    # Group labels per note (sets, then sorted lists)
    per_note: dict[str, list[str]] = defaultdict(list)
    for _, row in labels.iterrows():
        per_note[row["note_id"]].append(row["label_code"])

    # Drop notes with zero labels (same rule as load_data uses)
    notes_with_labels = notes[notes["note_id"].isin(per_note.keys())].reset_index(drop=True)
    print(f"Notes with at least one 3char label: {len(notes_with_labels)}")

    # Build the multi-hot matrix for stratification
    all_codes = sorted({c for codes in per_note.values() for c in codes})
    code_to_idx = {c: i for i, c in enumerate(all_codes)}
    y = np.zeros((len(notes_with_labels), len(all_codes)), dtype=int)
    for i, nid in enumerate(notes_with_labels["note_id"]):
        for c in per_note[nid]:
            y[i, code_to_idx[c]] = 1

    # Multi-label stratified split: a single fold of 1/test_fraction gives us
    # one test set with stratification. n_splits=5 -> ~20% test.
    n_splits = round(1 / TEST_FRACTION)
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(mskf.split(notes_with_labels["note_id"].values, y))

    dev_ids = sorted(notes_with_labels["note_id"].iloc[train_idx].tolist())
    test_ids = sorted(notes_with_labels["note_id"].iloc[test_idx].tolist())

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "test_fraction": TEST_FRACTION,
        "n_notes_total": len(notes_with_labels),
        "n_dev": len(dev_ids),
        "n_test": len(test_ids),
        "rollup_used_for_strat": "3char",
        "dev": dev_ids,
        "test": test_ids,
    }
    SPLIT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {SPLIT_PATH}")
    print(f"  dev:  {len(dev_ids)} notes")
    print(f"  test: {len(test_ids)} notes")

    # Quick sanity: how many distinct codes in each bucket?
    dev_codes = {c for nid in dev_ids for c in per_note[nid]}
    test_codes = {c for nid in test_ids for c in per_note[nid]}
    print(f"  distinct codes in dev:  {len(dev_codes)}")
    print(f"  distinct codes in test: {len(test_codes)}")
    print(f"  codes in test but not dev: {sorted(test_codes - dev_codes)}")


if __name__ == "__main__":
    main()