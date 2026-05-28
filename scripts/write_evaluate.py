from pathlib import Path

CONTENT = '''"""Evaluate a trained baseline model on the held-out test set.

Prints metrics + per-code breakdown + one sample prediction.
Returns a metrics dict for cross-rollup comparison.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)

from src.baseline.load_data import load_data
from src.baseline.train import load_bundle

logger = logging.getLogger(__name__)


def evaluate(rollup: str = "3char", *, top_n_codes: int = 20) -> dict[str, Any]:
    """Load the trained bundle, predict on test set, compute metrics."""
    ds = load_data(rollup=rollup)
    bundle = load_bundle(rollup=rollup)

    # Defensive: label order must match between training and now.
    # MultiLabelBinarizer is deterministic on identical input, but assert anyway.
    assert bundle.label_names == ds.label_names, (
        "Label order mismatch between training and evaluation. "
        "Did the data change between runs?"
    )

    y_pred = bundle.pipeline.predict(ds.X_test)

    micro_f1 = f1_score(ds.y_test, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(ds.y_test, y_pred, average="macro", zero_division=0)
    subset_acc = float((ds.y_test == y_pred).all(axis=1).mean())
    hamming = hamming_loss(ds.y_test, y_pred)

    # Per-code (per-class) P/R/F1
    p, r, f, support = precision_recall_fscore_support(
        ds.y_test, y_pred, average=None, zero_division=0
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
        "n_test": int(len(ds.X_test)),
        "n_codes": len(ds.label_names),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "subset_accuracy": subset_acc,
        "hamming_loss": float(hamming),
        "per_code": per_code,
        "_y_true": ds.y_test,
        "_y_pred": y_pred,
        "_label_names": ds.label_names,
        "_X_test": ds.X_test,
    }


def print_report(metrics: dict[str, Any], top_n_codes: int = 20) -> None:
    print(f"\\n=== rollup={metrics['rollup']} ===")
    print(f"  n_test:           {metrics['n_test']}")
    print(f"  n_codes:          {metrics['n_codes']}")
    print(f"  micro F1:         {metrics['micro_f1']:.3f}")
    print(f"  macro F1:         {metrics['macro_f1']:.3f}")
    print(f"  subset accuracy:  {metrics['subset_accuracy']:.3f}")
    print(f"  hamming loss:     {metrics['hamming_loss']:.3f}")

    print(f"\\n  --- top {top_n_codes} codes by test support ---")
    print(f"  {'code':10s} {'support':>8s} {'precision':>10s} {'recall':>8s} {'f1':>6s}")
    for entry in metrics["per_code"][:top_n_codes]:
        print(f"  {entry['code']:10s} {entry['support']:>8d} "
              f"{entry['precision']:>10.2f} {entry['recall']:>8.2f} {entry['f1']:>6.2f}")

    # One sample prediction for eyeball QA.
    y_true = metrics["_y_true"]
    y_pred = metrics["_y_pred"]
    labels = metrics["_label_names"]
    X_test = metrics["_X_test"]

    # Pick the first test example that has at least one true label.
    idx = next((i for i, row in enumerate(y_true) if row.sum() > 0), 0)

    true_codes = [labels[j] for j in np.where(y_true[idx] == 1)[0]]
    pred_codes = [labels[j] for j in np.where(y_pred[idx] == 1)[0]]

    print(f"\\n  --- sample prediction (test idx={idx}) ---")
    print(f"  truth:     {sorted(true_codes)}")
    print(f"  predicted: {sorted(pred_codes)}")
    print(f"  note head: {X_test[idx][:200].replace(chr(10), ' ')}...")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollup", choices=["dirty", "3char", "chapter"], default="3char")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    metrics = evaluate(rollup=args.rollup)
    print_report(metrics, top_n_codes=args.top)


if __name__ == "__main__":
    main()
'''

Path("src/baseline/evaluate.py").write_text(CONTENT)
print("Wrote src/baseline/evaluate.py")
print(f"  {len(CONTENT.splitlines())} lines")