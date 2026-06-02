"""Load notes + labels from BigQuery for a specified bucket.

Day 8 fix: split file is now keyed on (patient_id, strategy) pairs instead
of note_ids, so regenerations don't invalidate it.

Three rollup flavors:
  - "dirty":   raw notes_labels (lots of codes; fan-out included)
  - "3char":   notes_labels_clean_3char (cleanest target)
  - "chapter": notes_labels_clean_chapter (sanity-check coarsening)

Three buckets:
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


def _pair(patient_id: str, strategy: str) -> str:
    return f"{patient_id}::{strategy}"


def _load_split() -> dict[str, set[str]]:
    """Return {bucket: set of pairs}. Asserts pairs format."""
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing split file: {SPLIT_PATH}. "
            "Run `python scripts/make_split.py` first."
        )
    s = json.loads(SPLIT_PATH.read_text())
    if s.get("format") != "pairs":
        raise ValueError(
            f"Split file is not in pairs format (got format={s.get('format')!r}). "
            "Re-run scripts/make_split.py to migrate."
        )
    return {"dev": set(s["dev"]), "test": set(s["test"])}


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

    # 1. Pull current notes
    notes_df = client.query(_LATEST_NOTES_SQL).to_dataframe()
    notes_df["pair"] = notes_df["patient_id"] + "::" + notes_df["strategy"]

    # 2. Resolve split to current note_ids via pairs
    split = _load_split()
    dev_pairs = split["dev"]
    test_pairs = split["test"]

    if bucket == "dev":
        wanted_pairs = dev_pairs
    elif bucket == "test":
        wanted_pairs = test_pairs
    else:
        wanted_pairs = dev_pairs | test_pairs

    notes_df = notes_df[notes_df["pair"].isin(wanted_pairs)].reset_index(drop=True)
    wanted_ids = set(notes_df["note_id"])

    # 3. Pull labels (already filtered to current notes by SQL JOIN)
    labels_df = client.query(_LABEL_QUERIES[rollup]).to_dataframe()

    # 4. Label vocabulary determined from DEV only
    pair_to_nid = dict(zip(notes_df["pair"], notes_df["note_id"]))  # current bucket only
    # We need ALL dev pairs' note_ids for vocab, not just current bucket. Re-query.
    all_notes = client.query(_LATEST_NOTES_SQL).to_dataframe()
    all_notes["pair"] = all_notes["patient_id"] + "::" + all_notes["strategy"]
    dev_note_ids = set(all_notes[all_notes["pair"].isin(dev_pairs)]["note_id"])

    dev_labels = labels_df[labels_df["note_id"].isin(dev_note_ids)]
    code_counts = dev_labels["label_code"].value_counts()
    keep_codes = set(code_counts[code_counts >= min_code_freq].index)
    dropped_rare = set(code_counts.index) - keep_codes

    labels_df = labels_df[
        labels_df["note_id"].isin(wanted_ids)
        & labels_df["label_code"].isin(keep_codes)
    ]

    # 5. Group labels per note
    per_note = (
        labels_df.groupby("note_id")["label_code"]
        .apply(set)
        .to_dict()
    )
    notes_df["labels"] = notes_df["note_id"].map(lambda nid: per_note.get(nid, set()))

    # 6. Drop notes with zero labels
    before = len(notes_df)
    notes_df = notes_df[notes_df["labels"].map(len) > 0].reset_index(drop=True)
    dropped_empty = before - len(notes_df)

    # 7. Binarize
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
