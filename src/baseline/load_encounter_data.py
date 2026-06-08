"""Load encounter-grouped notes + labels from BigQuery.

Parallel to load_data.py but uses the encounter split
(data/splits/encounter_split.json) keyed on encounter_id.

Label vocab is built from dev encounters only, same convention as load_data.py.
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
SPLIT_PATH = Path("data/splits/encounter_split.json")


_NOTES_SQL = f"""
    SELECT note_id, note_text, patient_id, encounter_id
    FROM `{PROJECT}.{DATASET}.notes_synth`
    WHERE strategy = 'encounter-multilabel'
      AND encounter_id IS NOT NULL
"""

_LABELS_SQL = f"""
    SELECT l.note_id, l.label_code
    FROM `{PROJECT}.{DATASET}.notes_labels_clean_3char` l
    JOIN ({_NOTES_SQL}) n USING (note_id)
"""


@dataclass
class LoadedBucket:
    X: list[str]
    y: np.ndarray
    label_names: list[str]
    note_ids: list[str]
    info: dict[str, Any]


def _load_split() -> dict[str, set[str]]:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing split file: {SPLIT_PATH}. "
            "Run scripts/make_encounter_split.py first."
        )
    s = json.loads(SPLIT_PATH.read_text())
    if s.get("format") != "encounter_ids":
        raise ValueError(f"Unexpected split format: {s.get('format')!r}")
    return {"dev": set(s["dev"]), "test": set(s["test"])}


def load_encounter_data(
    *,
    bucket: str = "dev",
    min_code_freq: int = 3,
) -> LoadedBucket:
    """Return encounter notes for the requested bucket."""
    if bucket not in ("dev", "test", "all"):
        raise ValueError(f"Unknown bucket: {bucket!r}")

    client = bigquery.Client(project=PROJECT)

    notes_df = client.query(_NOTES_SQL).to_dataframe()
    labels_df = client.query(_LABELS_SQL).to_dataframe()

    split = _load_split()
    dev_eids = split["dev"]
    test_eids = split["test"]

    if bucket == "dev":
        wanted_eids = dev_eids
    elif bucket == "test":
        wanted_eids = test_eids
    else:
        wanted_eids = dev_eids | test_eids

    notes_df = notes_df[notes_df["encounter_id"].isin(wanted_eids)].reset_index(drop=True)
    wanted_note_ids = set(notes_df["note_id"])

    # Label vocab from dev only
    dev_note_ids = set(
        notes_df["note_id"][notes_df["encounter_id"].isin(dev_eids)]
        if bucket != "all"
        else notes_df["note_id"][notes_df["encounter_id"].isin(dev_eids)]
    )
    # Need all dev note_ids regardless of current bucket
    all_notes_df = client.query(_NOTES_SQL).to_dataframe()
    dev_note_ids = set(all_notes_df[all_notes_df["encounter_id"].isin(dev_eids)]["note_id"])

    dev_labels = labels_df[labels_df["note_id"].isin(dev_note_ids)]
    code_counts = dev_labels["label_code"].value_counts()
    keep_codes = set(code_counts[code_counts >= min_code_freq].index)

    labels_df = labels_df[
        labels_df["note_id"].isin(wanted_note_ids)
        & labels_df["label_code"].isin(keep_codes)
    ]

    per_note = (
        labels_df.groupby("note_id")["label_code"]
        .apply(set)
        .to_dict()
    )
    notes_df["labels"] = notes_df["note_id"].map(lambda nid: per_note.get(nid, set()))

    before = len(notes_df)
    notes_df = notes_df[notes_df["labels"].map(len) > 0].reset_index(drop=True)
    dropped_empty = before - len(notes_df)

    mlb = MultiLabelBinarizer(classes=sorted(keep_codes))
    y = mlb.fit_transform(notes_df["labels"])
    label_names = list(mlb.classes_)

    info = {
        "bucket": bucket,
        "n_notes": len(notes_df),
        "n_notes_dropped_empty": dropped_empty,
        "n_codes_kept": len(label_names),
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
    for bucket in ("dev", "test"):
        ds = load_encounter_data(bucket=bucket)
        print(f"\n--- bucket={bucket} ---")
        for k, v in ds.info.items():
            print(f"  {k:25s} {v}")


if __name__ == "__main__":
    main()
