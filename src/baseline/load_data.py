"""Load notes + labels from BigQuery for a specified bucket.

Dev/test split lives in data/splits/dev_test_split.json (committed). This
module reads that file and returns only the notes for the requested bucket.

Three rollup flavors:
  - "dirty":   raw notes_labels (~150 codes after rare-filter)
  - "3char":   notes_labels_clean_3char (~36 codes after rare-filter)
  - "chapter": notes_labels_clean_chapter (~12 codes after rare-filter)

Two buckets:
  - "dev":  77 notes, used for training + CV. Touch freely.
  - "test": 19 notes, held out. Touch only for final evaluation.
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

_LABEL_QUERIES = {
    "dirty":
        f"""
        SELECT note_id, icd10_code AS label_code
        FROM `{PROJECT}.{DATASET}.notes_labels`
        """,
    "3char":
        f"""
        SELECT note_id, label_code
        FROM `{PROJECT}.{DATASET}.notes_labels_clean_3char`
        """,
    "chapter":
        f"""
        SELECT note_id, label_code
        FROM `{PROJECT}.{DATASET}.notes_labels_clean_chapter`
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


def load_data(
    rollup: str = "3char",
    *,
    bucket: str = "dev",
    min_code_freq: int = 3,
) -> LoadedBucket:
    """Return data for the requested bucket of the dev/test split.

    Args:
        rollup:        one of "dirty", "3char", "chapter"
        bucket:        one of "dev", "test", "all"
        min_code_freq: drop codes appearing in fewer than this many notes
                       in the DEV bucket (applied uniformly to dev and test
                       so label vocabulary stays consistent)
    """
    if rollup not in _LABEL_QUERIES:
        raise ValueError(f"Unknown rollup: {rollup!r}")
    if bucket not in ("dev", "test", "all"):
        raise ValueError(f"Unknown bucket: {bucket!r}")

    split = _load_split()
    dev_ids = set(split["dev"])
    test_ids = set(split["test"])
    if bucket == "dev":
        wanted_ids = dev_ids
    elif bucket == "test":
        wanted_ids = test_ids
    else:
        wanted_ids = dev_ids | test_ids

    client = bigquery.Client(project=PROJECT)

    # 1. All notes (we filter to the bucket in Python — small dataset, simple)
    notes_df = client.query(f"""
        SELECT note_id, note_text
        FROM `{PROJECT}.{DATASET}.notes_synth`
    """).to_dataframe()
    notes_df = notes_df[notes_df["note_id"].isin(wanted_ids)].reset_index(drop=True)

    # 2. All labels for the requested rollup
    labels_df = client.query(_LABEL_QUERIES[rollup]).to_dataframe()

    # 3. Determine the label vocabulary from DEV ONLY, then apply rare-filter.
    #    This ensures dev and test share the same label space.
    dev_labels = labels_df[labels_df["note_id"].isin(dev_ids)]
    code_counts = dev_labels["label_code"].value_counts()
    keep_codes = set(code_counts[code_counts >= min_code_freq].index)
    dropped_rare = set(code_counts.index) - keep_codes

    labels_df = labels_df[
        labels_df["note_id"].isin(wanted_ids)
        & labels_df["label_code"].isin(keep_codes)
    ]

    # 4. Group labels per note
    per_note = (
        labels_df.groupby("note_id")["label_code"]
        .apply(set)
        .to_dict()
    )
    notes_df["labels"] = notes_df["note_id"].map(lambda nid: per_note.get(nid, set()))

    # 5. Drop notes with zero labels (no signal)
    before = len(notes_df)
    notes_df = notes_df[notes_df["labels"].map(len) > 0].reset_index(drop=True)
    dropped_empty = before - len(notes_df)

    # 6. Binarize (fit on a sorted label list so order is deterministic)
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
    """Smoke test: load all (rollup, bucket) combinations and print info."""
    for rollup in ("dirty", "3char", "chapter"):
        for bucket in ("dev", "test"):
            ds = load_data(rollup=rollup, bucket=bucket)
            print(f"\n--- rollup={rollup} bucket={bucket} ---")
            for k, v in ds.info.items():
                print(f"  {k:25s} {v}")


if __name__ == "__main__":
    main()
