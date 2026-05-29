from pathlib import Path

CONTENT = '''"""k-fold cross-validation on the dev bucket.

Multi-label stratified k-fold (default k=5). For each fold:
  - train a fresh pipeline on the fold's training portion
  - predict on the fold's validation portion
  - score metrics

Out-of-fold (OOF) predictions: each dev note ends up with exactly one
prediction, made by a model that did NOT see it. We use these for the
threshold sweep in step 4.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)

from src.baseline.load_data import load_data
from src.baseline.train import build_pipeline

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    rollup: str
    n_folds: int
    per_fold: list[dict[str, float]]
    mean_micro_f1: float
    std_micro_f1: float
    mean_macro_f1: float
    std_macro_f1: float
    oof_y_true: np.ndarray
    oof_y_pred: np.ndarray
    oof_y_proba: np.ndarray
    label_names: list[str]


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


def cross_validate(rollup: str = "3char", *, n_folds: int = 5,
                   seed: int = 42) -> CVResult:
    """Run k-fold CV on the dev bucket and return per-fold + OOF results."""
    dev = load_data(rollup=rollup, bucket="dev")
    X = np.asarray(dev.X, dtype=object)
    y = dev.y
    label_names = dev.label_names

    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_pred = np.zeros_like(y)
    oof_proba = np.zeros(y.shape, dtype=float)
    per_fold: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(mskf.split(X, y), start=1):
        logger.info("fold %d/%d: train=%d val=%d",
                    fold_idx, n_folds, len(train_idx), len(val_idx))

        pipe = build_pipeline()
        pipe.fit(X[train_idx].tolist(), y[train_idx])

        y_pred_fold = pipe.predict(X[val_idx].tolist())
        # decision_function-style probabilities for the threshold sweep
        if hasattr(pipe, "predict_proba"):
            y_proba_fold = pipe.predict_proba(X[val_idx].tolist())
        else:
            # OneVsRest with liblinear LR always supports predict_proba
            raise RuntimeError("Pipeline does not support predict_proba")

        oof_pred[val_idx] = y_pred_fold
        oof_proba[val_idx] = y_proba_fold

        scores = _score(y[val_idx], y_pred_fold)
        scores["fold"] = fold_idx
        scores["n_train"] = int(len(train_idx))
        scores["n_val"] = int(len(val_idx))
        per_fold.append(scores)
        logger.info("  micro_f1=%.3f macro_f1=%.3f",
                    scores["micro_f1"], scores["macro_f1"])

    micros = np.array([f["micro_f1"] for f in per_fold])
    macros = np.array([f["macro_f1"] for f in per_fold])

    return CVResult(
        rollup=rollup,
        n_folds=n_folds,
        per_fold=per_fold,
        mean_micro_f1=float(micros.mean()),
        std_micro_f1=float(micros.std()),
        mean_macro_f1=float(macros.mean()),
        std_macro_f1=float(macros.std()),
        oof_y_true=y,
        oof_y_pred=oof_pred,
        oof_y_proba=oof_proba,
        label_names=label_names,
    )


def print_cv_report(result: CVResult) -> None:
    print(f"\\n=== CV: rollup={result.rollup} k={result.n_folds} ===")
    print(f"  {'fold':>4s} {'n_train':>8s} {'n_val':>6s} "
          f"{'micro_f1':>9s} {'macro_f1':>9s} {'subset':>7s} {'hamming':>8s}")
    for f in result.per_fold:
        print(f"  {f['fold']:>4d} {f['n_train']:>8d} {f['n_val']:>6d} "
              f"{f['micro_f1']:>9.3f} {f['macro_f1']:>9.3f} "
              f"{f['subset_accuracy']:>7.3f} {f['hamming_loss']:>8.3f}")
    print(f"\\n  mean micro F1: {result.mean_micro_f1:.3f} +/- {result.std_micro_f1:.3f}")
    print(f"  mean macro F1: {result.mean_macro_f1:.3f} +/- {result.std_macro_f1:.3f}")

    # Also score the OOF predictions as a sanity check (should match approximately
    # the per-fold means, since every dev note appears in exactly one val set).
    oof_scores = _score(result.oof_y_true, result.oof_y_pred)
    print(f"\\n  OOF micro F1: {oof_scores['micro_f1']:.3f}")
    print(f"  OOF macro F1: {oof_scores['macro_f1']:.3f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollup", choices=["dirty", "3char", "chapter"], default="3char")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = cross_validate(rollup=args.rollup, n_folds=args.k, seed=args.seed)
    print_cv_report(result)


if __name__ == "__main__":
    main()
'''

Path("src/baseline/cv.py").write_text(CONTENT)
print("Wrote src/baseline/cv.py")
print(f"  {len(CONTENT.splitlines())} lines")