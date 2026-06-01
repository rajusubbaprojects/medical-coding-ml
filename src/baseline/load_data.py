"""Load notes + labels from BigQuery for a specified bucket.

Day 8: now picks the LATEST version per (patient_id, strategy) pair so that
re-generated notes (e.g., with a new prompt version) supersede older ones.

Dev/test split lives in data/splits/dev_test_split.json (committed).

Three rollup flavors:
  - "dirty":   raw notes_labels (~150 codes after rare-filter)
  - "3char":   notes_labels_clean_3char (~36 codes after rare-filter)
  - "chapter": notes_labels_clean_chapter (~12 codes after rare-filter)

Two buckets:
  - "dev":  used for training + CV. Touch freely.
  - "test": held out. Touch only for final evaluation.
  - "all":  both, for analysis only. Do not train on this.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from google.cloud import bigquery
from sklearn.preprocessing import MultiLabelBinarizer

PROJECT = "medical-coding-ml-9848"
DATASET = "medical_coding"
SPLIT_PATH = Path("data/splits/dev_test_split.json")

# Pull the LATEST note per (patient_id, strategy). This is the canonical
# query for "current notes" everywhere in the pipeline.
_LATEST_NOTES_SQL = f"""
SELECT note_id, note_text, patient_id, strategy, prompt_version
FROM `{PROJECT}.{DATASET}.notes_synth`
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY patient_id, strategy
    ORDER BY generated_at DESC
) = 1
"""

_LABEL_QUERIES = {
    "dirty":
        f"""
        SELECT l.note_id, l.icd10_code AS label_code
        FROM `{PROJECT}.{DATASET}.notes_labels` l
        JOIN ({_LATEST_NOTES_SQL}) n USING (note_id)
        """,
    "3char":
        f"""
        SELECT l.note_id, l.label_code
        FROM `{PROJECT}.{DATASET}.notes_labels_clean_3char` l
        JOIN ({_LATEST_NOTES_SQL}) n USING (note_id)
        """,
    "chapter":
        f"""
        SELECT l.note_id, l.label_code
        FROM `{PROJECT}.{DATASET}.notes_labels_clean_chapter` l
        JOIN ({_LATEST_NOTES_SQL}) n USING (note_id)
        """,
}


@dataclass
class LoadedBucket:
    X: list[str]
    y: np.ndarray
    label_names: list[str]
    note_ids: list[str]
    info: dict[str, Any]


def _load_split() -> dict[str, Any]:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing split file: {SPLIT_PATH}. "
            "Run `python scripts/make_split.py` first."
        )
    return json.loads(SPLIT_PATH.read_text())


def _resolve_split_to_current_notes(split: dict[str, Any],
                                    note_id_lookup: dict[str, str]) -> dict[str, set[str]]:
    """Map old note_ids in the split to current note_ids if they were regenerated.

    The split file stores note_ids from the time the split was created. If we
    regenerated some notes (new run_id, new note_id, same patient + strategy),
    the split's note_ids would no longer match. We resolve by (patient, strategy)
    -> current note_id via the lookup.
    """
    def _resolve(old_id: str) -> str | None:
        # note_id format: "<run_id>_<patient_id>_<strategy>"
        # We rebuild the lookup key from the suffix.
        parts = old_id.split("_", 1)
        if len(parts) != 2:
            return None
        # parts[1] is "<patient_id>_<strategy>"
        return note_id_lookup.get(parts[1])

    resolved = {}
    for bucket in ("dev", "test"):
        ids = set()
        for old_id in split[bucket]:
            new_id = _resolve(old_id)
            if new_id:
                ids.add(new_id)
        resolved[bucket] = ids
    return resolved


def load_data(
    rollup: str = "3char",
    *,
    bucket: str = "dev",
    min_code_freq: int = 3,
) -> LoadedBucket:
    """Return data for the requested bucket of the dev/test split."""
    if rollup not in _LABEL_QUERIES:
        raise ValueError(f"Unknown rollup: {rollup!r}")
    if bucket not in ("dev", "test", "all"):
        raise ValueError(f"Unknown bucket: {bucket!r}")

    client = bigquery.Client(project=PROJECT)

    # 1. Pull current notes (latest version per patient + strategy)
    notes_df = client.query(_LATEST_NOTES_SQL).to_dataframe()

    # 2. Build lookup: "<patient_id>_<strategy>" -> current note_id
    notes_df["lookup_key"] = notes_df["patient_id"] + "_" + notes_df["strategy"]
    note_id_lookup = dict(zip(notes_df["lookup_key"], notes_df["note_id"]))

    # 3. Resolve the committed split's note_ids to current ones
    split = _load_split()
    resolved = _resolve_split_to_current_notes(split, note_id_lookup)
    dev_ids = resolved["dev"]
    test_ids = resolved["test"]

    if bucket == "dev":
        wanted_ids = dev_ids
    elif bucket == "test":
        wanted_ids = test_ids
    else:
        wanted_ids = dev_ids | test_ids

    notes_df = notes_df[notes_df["note_id"].isin(wanted_ids)].reset_index(drop=True)

    # 4. Pull labels (already filtered to current notes by the JOIN in SQL)
    labels_df = client.query(_LABEL_QUERIES[rollup]).to_dataframe()

    # 5. Label vocabulary from DEV ONLY, rare-filter applied uniformly
    dev_labels = labels_df[labels_df["note_id"].isin(dev_ids)]
    code_counts = dev_labels["label_code"].value_counts()
    keep_codes = set(code_counts[code_counts >= min_code_freq].index)
    dropped_rare = set(code_counts.index) - keep_codes

    labels_df = labels_df[
        labels_df["note_id"].isin(wanted_ids)
        & labels_df["label_code"].isin(keep_codes)
    ]

    # 6. Group labels per note
    per_note = (
        labels_df.groupby("note_id")["label_code"]
        .apply(set)
        .to_dict()
    )
    notes_df["labels"] = notes_df["note_id"].map(lambda nid: per_note.get(nid, set()))

    # 7. Drop notes with zero labels
    before = len(notes_df)
    notes_df = notes_df[notes_df["labels"].map(len) > 0].reset_index(drop=True)
    dropped_empty = before - len(notes_df)

    # 8. Binarize
    mlb = MultiLabelBinarizer(classes=sorted(keep_codes))
    y = mlb.fit_transform(notes_df["labels"])
    label_names = list(mlb.classes_)

    info = {
        "rollup": rollup,
        "bucket": bucket,
        "n_notes": len(notes_df),
        "n_notes_dropped_empty": dropped_empty,
        "n_codes_kept": len(label_names),
        "n_codes_dropped_rare": len(dropped_rare),
        "min_code_freq": min_code_freq,
        "avg_labels_per_note": float(y.sum(axis=1).mean()) if len(y) else 0.0,
    }
    return LoadedBucket(
        X=notes_df["note_text"].tolist(),
        y=y,
        label_names=label_names,
        note_ids=notes_df["note_id"].tolist(),
        info=info,
    )


def main() -> None:
    for rollup in ("dirty", "3char", "chapter"):
        for bucket in ("dev", "test"):
            ds = load_data(rollup=rollup, bucket=bucket)
            print(f"\n--- rollup={rollup} bucket={bucket} ---")
            for k, v in ds.info.items():
                print(f"  {k:25s} {v}")


if __name__ == "__main__":
    main()
