"""Build the stable dev/test split, keyed on (patient_id, strategy) pairs.

Why pairs instead of note_ids: note_ids embed run_id, so they change when we
regenerate a batch. The logical identity of "this note in our project" is
(patient, strategy), which is stable across regenerations.

Logic:
  - Original pairs (already in the previously-committed split) keep their
    bucket assignments. Hard rule.
  - New pairs (not in the previous split) get a fresh stratified 80/20 split.
  - Stratification key: 3char labels (our real target rollup).
  - Same seed (42) for reproducibility.

Run once after generation; writes data/splits/dev_test_split.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from google.cloud import bigquery
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

PROJECT = "medical-coding-ml-9848"
SEED = 42
TEST_FRACTION = 0.20
SPLIT_PATH = Path("data/splits/dev_test_split.json")


def _pair(patient_id: str, strategy: str) -> str:
    """Stable identity for a note across regenerations."""
    return f"{patient_id}::{strategy}"


def _load_existing() -> tuple[set[str], set[str], bool]:
    """Return (dev_pairs, test_pairs, is_legacy_note_id_format)."""
    if not SPLIT_PATH.exists():
        return set(), set(), False

    prev = json.loads(SPLIT_PATH.read_text())
    dev_raw = prev.get("dev", [])
    test_raw = prev.get("test", [])
    if not dev_raw:
        return set(), set(), False

    # Detect format: pairs contain "::", legacy note_ids contain "_" and start with a timestamp
    sample = dev_raw[0]
    is_pair = "::" in sample
    if is_pair:
        return set(dev_raw), set(test_raw), False

    # Legacy: convert note_id -> (patient_id, strategy)
    # note_id format: "<run_id>_<patient_id>_<strategy>"
    # split on first underscore to drop run_id, then last underscore separates strategy
    def to_pair(nid: str) -> str | None:
        parts = nid.split("_", 1)
        if len(parts) != 2:
            return None
        rest = parts[1]  # "<patient_id>_<strategy>"
        # patient_id contains hyphens (UUIDs); strategy is one of two known values
        for strat in ("multilabel", "primary_dx"):
            suffix = f"_{strat}"
            if rest.endswith(suffix):
                pid = rest[: -len(suffix)]
                return _pair(pid, strat)
        return None

    dev_pairs = {p for p in (to_pair(n) for n in dev_raw) if p}
    test_pairs = {p for p in (to_pair(n) for n in test_raw) if p}
    return dev_pairs, test_pairs, True


def _current_pairs(client: bigquery.Client) -> dict[str, str]:
    """Pair -> latest note_id."""
    rows = client.query(f"""
        SELECT patient_id, strategy, note_id
        FROM `{PROJECT}.medical_coding.notes_synth`
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY patient_id, strategy
            ORDER BY generated_at DESC
        ) = 1
    """).to_dataframe()
    return {_pair(r.patient_id, r.strategy): r.note_id for _, r in rows.iterrows()}


def _labels_3char_by_pair(client: bigquery.Client, pair_to_nid: dict[str, str]) -> dict[str, list[str]]:
    rows = client.query(f"""
        SELECT l.note_id, l.label_code
        FROM `{PROJECT}.medical_coding.notes_labels_clean_3char` l
        JOIN (
            SELECT note_id
            FROM `{PROJECT}.medical_coding.notes_synth`
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY patient_id, strategy
                ORDER BY generated_at DESC
            ) = 1
        ) n USING (note_id)
    """).to_dataframe()

    nid_to_pair = {nid: pair for pair, nid in pair_to_nid.items()}
    per_pair: dict[str, list[str]] = defaultdict(list)
    for _, row in rows.iterrows():
        pair = nid_to_pair.get(row["note_id"])
        if pair:
            per_pair[pair].append(row["label_code"])
    return per_pair


def _stratified_split(pairs: list[str], per_pair: dict[str, list[str]],
                      seed: int, test_fraction: float
                      ) -> tuple[list[str], list[str], list[str]]:
    """Multi-label stratified split. Returns (dev, test, skipped_empty)."""
    with_labels = [p for p in pairs if per_pair.get(p)]
    skipped = [p for p in pairs if not per_pair.get(p)]

    if not with_labels:
        return [], [], skipped

    all_codes = sorted({c for p in with_labels for c in per_pair[p]})
    code_to_idx = {c: i for i, c in enumerate(all_codes)}
    y = np.zeros((len(with_labels), len(all_codes)), dtype=int)
    for i, p in enumerate(with_labels):
        for c in per_pair[p]:
            y[i, code_to_idx[c]] = 1

    n_splits = round(1 / test_fraction)
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(mskf.split(np.array(with_labels), y))

    dev = sorted([with_labels[i] for i in train_idx])
    test = sorted([with_labels[i] for i in test_idx])
    return dev, test, skipped


def main() -> None:
    client = bigquery.Client(project=PROJECT)

    pair_to_nid = _current_pairs(client)
    current_pairs = set(pair_to_nid.keys())
    print(f"Current pairs in BQ: {len(current_pairs)}")

    existing_dev, existing_test, was_legacy = _load_existing()
    if was_legacy:
        print(f"Existing split was in legacy note_id format; converted to pairs.")
    print(f"Existing split: dev={len(existing_dev)} test={len(existing_test)}")

    per_pair = _labels_3char_by_pair(client, pair_to_nid)

    # Partition
    carried_dev = sorted(existing_dev & current_pairs)
    carried_test = sorted(existing_test & current_pairs)
    new_pairs = sorted(current_pairs - existing_dev - existing_test)
    orphaned = (existing_dev | existing_test) - current_pairs
    print(f"Carried over: dev={len(carried_dev)} test={len(carried_test)}")
    print(f"New pairs to split: {len(new_pairs)}")
    if orphaned:
        print(f"Orphaned (in old split, not in current BQ): {len(orphaned)} — dropped")

    new_dev, new_test, skipped = _stratified_split(
        new_pairs, per_pair, seed=SEED, test_fraction=TEST_FRACTION
    )
    print(f"New split: dev={len(new_dev)} test={len(new_test)} skipped_empty={len(skipped)}")

    final_dev = sorted(carried_dev + new_dev)
    final_test = sorted(carried_test + new_test)

    overlap = set(final_dev) & set(final_test)
    assert not overlap, f"Dev/test overlap: {overlap}"

    dev_codes = {c for p in final_dev for c in per_pair.get(p, [])}
    test_codes = {c for p in final_test for c in per_pair.get(p, [])}
    only_in_test = sorted(test_codes - dev_codes)

    payload: dict[str, Any] = {
        "seed": SEED,
        "test_fraction": TEST_FRACTION,
        "format": "pairs",  # marker for future loaders
        "n_notes_total": len(final_dev) + len(final_test),
        "n_dev": len(final_dev),
        "n_test": len(final_test),
        "n_carried_dev": len(carried_dev),
        "n_carried_test": len(carried_test),
        "n_new_dev": len(new_dev),
        "n_new_test": len(new_test),
        "rollup_used_for_strat": "3char",
        "dev": final_dev,
        "test": final_test,
    }

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(payload, indent=2))
    print()
    print(f"Wrote {SPLIT_PATH}")
    print(f"  total: {payload['n_notes_total']}")
    print(f"  dev:   {payload['n_dev']} ({len(carried_dev)} carried + {len(new_dev)} new)")
    print(f"  test:  {payload['n_test']} ({len(carried_test)} carried + {len(new_test)} new)")
    print(f"  codes in test but not dev: {len(only_in_test)} {only_in_test if len(only_in_test) <= 10 else '...'}")


if __name__ == "__main__":
    main()