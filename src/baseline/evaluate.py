"""Evaluate a trained baseline model on the DEV bucket.

NOTE: this evaluates on dev for now (matches Day 6 behavior). The dev bucket
is what we tune on; final test bucket evaluation happens on Day 11-12 and gets
its own driver. Day 7 work (CV, thresholds) all happens on the dev bucket.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)

from src.baseline.load_data import load_data
from src.baseline.train import load_bundle

logger = logging.getLogger(__name__)


def evaluate(rollup: str = "3char", *, bucket: str = "dev",
             top_n_codes: int = 20) -> dict[str, Any]:
    """Load the trained bundle, predict on the requested bucket, compute metrics."""
    ds = load_data(rollup=rollup, bucket=bucket)
    bundle = load_bundle(rollup=rollup)

    assert bundle.label_names == ds.label_names, (
        "Label order mismatch between training and evaluation. "
        "Did the data change between runs?"
    )

    y_pred = bundle.pipeline.predict(ds.X)

    micro_f1 = f1_score(ds.y, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(ds.y, y_pred, average="macro", zero_division=0)
    subset_acc = float((ds.y == y_pred).all(axis=1).mean())
    hamming = hamming_loss(ds.y, y_pred)

    p, r, f, support = precision_recall_fscore_support(
        ds.y, y_pred, average=None, zero_division=0
    )
    per_code = sorted(
        [
            {
                "code": ds.label_names[i],
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(support[i]),
            }
            for i in range(len(ds.label_names))
        ],
        key=lambda d: (-d["support"], d["code"]),
    )

    return {
        "rollup": rollup,
        "bucket": bucket,
        "n_notes": int(len(ds.X)),
        "n_codes": len(ds.label_names),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "subset_accuracy": subset_acc,
        "hamming_loss": float(hamming),
        "per_code": per_code,
        "_y_true": ds.y,
        "_y_pred": y_pred,
        "_label_names": ds.label_names,
        "_X": ds.X,
    }


def print_report(metrics: dict[str, Any], top_n_codes: int = 20) -> None:
    print(f"\n=== rollup={metrics['rollup']} bucket={metrics['bucket']} ===")
    print(f"  n_notes:          {metrics['n_notes']}")
    print(f"  n_codes:          {metrics['n_codes']}")
    print(f"  micro F1:         {metrics['micro_f1']:.3f}")
    print(f"  macro F1:         {metrics['macro_f1']:.3f}")
    print(f"  subset accuracy:  {metrics['subset_accuracy']:.3f}")
    print(f"  hamming loss:     {metrics['hamming_loss']:.3f}")

    print(f"\n  --- top {top_n_codes} codes by support ---")
    print(f"  {'code':10s} {'support':>8s} {'precision':>10s} {'recall':>8s} {'f1':>6s}")
    for entry in metrics["per_code"][:top_n_codes]:
        print(f"  {entry['code']:10s} {entry['support']:>8d} "
              f"{entry['precision']:>10.2f} {entry['recall']:>8.2f} {entry['f1']:>6.2f}")

    y_true = metrics["_y_true"]
    y_pred = metrics["_y_pred"]
    labels = metrics["_label_names"]
    X = metrics["_X"]
    idx = next((i for i, row in enumerate(y_true) if row.sum() > 0), 0)
    true_codes = [labels[j] for j in np.where(y_true[idx] == 1)[0]]
    pred_codes = [labels[j] for j in np.where(y_pred[idx] == 1)[0]]
    print(f"\n  --- sample prediction (idx={idx}) ---")
    print(f"  truth:     {sorted(true_codes)}")
    print(f"  predicted: {sorted(pred_codes)}")
    print(f"  note head: {X[idx][:200].replace(chr(10), ' ')}...")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollup", choices=["dirty", "3char", "chapter"], default="3char")
    parser.add_argument("--bucket", choices=["dev", "test"], default="dev")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    metrics = evaluate(rollup=args.rollup, bucket=args.bucket)
    print_report(metrics, top_n_codes=args.top)


if __name__ == "__main__":
    main()
