from pathlib import Path

CONTENT = '''"""Load notes + labels from BigQuery into train/test splits.

Three rollup flavors (chosen via param):
  - "dirty":   raw notes_labels (236 codes, fan-out included)
  - "3char":   notes_labels_clean_3char (~49 codes, fan-out dedup'd, no Z)
  - "chapter": notes_labels_clean_chapter (~16 codes, easy-mode)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

PROJECT = "medical-coding-ml-9848"
DATASET = "medical_coding"


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
class DataSplit:
    X_train: list[str]
    X_test: list[str]
    y_train: np.ndarray
    y_test: np.ndarray
    label_names: list[str]
    info: dict[str, Any]


def load_data(
    rollup: str = "3char",
    *,
    min_code_freq: int = 3,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DataSplit:
    """Return train/test split as a DataSplit.

    Args:
        rollup:        one of "dirty", "3char", "chapter"
        min_code_freq: drop codes appearing in fewer than this many notes
        test_size:     fraction of notes held out
        random_state:  for reproducibility
    """
    if rollup not in _LABEL_QUERIES:
        raise ValueError(f"Unknown rollup: {rollup!r}")

    client = bigquery.Client(project=PROJECT)

    # 1. Pull notes
    notes_df = client.query(f"""
        SELECT note_id, note_text
        FROM `{PROJECT}.{DATASET}.notes_synth`
    """).to_dataframe()

    # 2. Pull labels (chosen flavor)
    labels_df = client.query(_LABEL_QUERIES[rollup]).to_dataframe()

    # 3. Drop ultra-rare codes
    code_counts = labels_df["label_code"].value_counts()
    keep_codes = set(code_counts[code_counts >= min_code_freq].index)
    dropped_codes = set(code_counts.index) - keep_codes
    labels_df = labels_df[labels_df["label_code"].isin(keep_codes)]

    # 4. Group labels per note (set, to dedupe just in case)
    per_note_labels = (
        labels_df.groupby("note_id")["label_code"]
        .apply(set)
        .to_dict()
    )

    # Notes with zero labels after rare-code filter are kept but get empty labels.
    notes_df["labels"] = notes_df["note_id"].map(
        lambda nid: per_note_labels.get(nid, set())
    )

    # 5. Drop notes with zero labels (no signal to learn from)
    before = len(notes_df)
    notes_df = notes_df[notes_df["labels"].map(len) > 0].reset_index(drop=True)
    dropped_empty_notes = before - len(notes_df)

    # 6. Multi-label binarize
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(notes_df["labels"])
    label_names = list(mlb.classes_)

    # 7. Train/test split
    X = notes_df["note_text"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    info = {
        "rollup": rollup,
        "n_notes_total": len(notes_df),
        "n_notes_dropped_empty": dropped_empty_notes,
        "n_codes_kept": len(label_names),
        "n_codes_dropped_rare": len(dropped_codes),
        "min_code_freq": min_code_freq,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "avg_labels_per_note": float(y.sum(axis=1).mean()),
    }
    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_names=label_names,
        info=info,
    )


def main() -> None:
    """Smoke test: load all three rollups and print info dicts."""
    for rollup in ("dirty", "3char", "chapter"):
        ds = load_data(rollup=rollup)
        print(f"\\n--- rollup={rollup} ---")
        for k, v in ds.info.items():
            print(f"  {k:25s} {v}")
        print(f"  example labels: {ds.label_names[:6]}{'...' if len(ds.label_names) > 6 else ''}")


if __name__ == "__main__":
    main()
'''

Path("src/baseline/load_data.py").write_text(CONTENT)
print("Wrote src/baseline/load_data.py")
print(f"  {len(CONTENT.splitlines())} lines")