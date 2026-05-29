"""Threshold sweep over out-of-fold predictions on the dev bucket.

Loads OOF probabilities from a fresh CV run, sweeps thresholds from 0.10
to 0.90, finds:
  - threshold that maximizes micro F1
  - threshold that maximizes macro F1

Saves both choices + their metrics to models/thresholds_3char.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.baseline.cv import cross_validate

MODELS_DIR = Path("models")

logger = logging.getLogger(__name__)


def sweep_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    low: float = 0.10,
    high: float = 0.90,
    step: float = 0.05,
) -> list[dict[str, float]]:
    """Compute metrics at each threshold. Returns list of dicts."""
    thresholds = np.arange(low, high + 1e-9, step)
    rows: list[dict[str, float]] = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        rows.append({
            "threshold": float(round(t, 3)),
            "micro_precision": float(micro_p),
            "micro_recall": float(micro_r),
            "micro_f1": float(micro_f),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f),
        })
    return rows


def pick_best(rows: list[dict[str, float]], metric: str) -> dict[str, float]:
    return max(rows, key=lambda r: r[metric])


def print_curve(rows: list[dict[str, float]], best_micro_t: float, best_macro_t: float) -> None:
    print(f"\n  {'thresh':>7s} {'mP':>5s} {'mR':>5s} {'mF1':>5s} "
          f"{'MP':>5s} {'MR':>5s} {'MF1':>5s}  marker")
    for r in rows:
        marker = ""
        if r["threshold"] == best_micro_t:
            marker += " <- best micro"
        if r["threshold"] == best_macro_t:
            marker += " <- best macro"
        print(f"  {r['threshold']:>7.2f} "
              f"{r['micro_precision']:>5.2f} {r['micro_recall']:>5.2f} {r['micro_f1']:>5.2f} "
              f"{r['macro_precision']:>5.2f} {r['macro_recall']:>5.2f} {r['macro_f1']:>5.2f}"
              f"{marker}")


def run(rollup: str = "3char", *, save: bool = True) -> dict[str, Any]:
    logger.info("Running CV to get OOF probabilities (rollup=%s)", rollup)
    cv = cross_validate(rollup=rollup)
    logger.info("CV mean micro F1: %.3f +/- %.3f", cv.mean_micro_f1, cv.std_micro_f1)

    rows = sweep_thresholds(cv.oof_y_true, cv.oof_y_proba)
    best_micro = pick_best(rows, "micro_f1")
    best_macro = pick_best(rows, "macro_f1")

    print_curve(rows, best_micro["threshold"], best_macro["threshold"])
    print(f"\n  best for micro: threshold={best_micro['threshold']:.2f} "
          f"-> micro_f1={best_micro['micro_f1']:.3f}, "
          f"micro_precision={best_micro['micro_precision']:.3f}, "
          f"micro_recall={best_micro['micro_recall']:.3f}")
    print(f"  best for macro: threshold={best_macro['threshold']:.2f} "
          f"-> macro_f1={best_macro['macro_f1']:.3f}, "
          f"macro_precision={best_macro['macro_precision']:.3f}, "
          f"macro_recall={best_macro['macro_recall']:.3f}")

    payload: dict[str, Any] = {
        "rollup": rollup,
        "default_threshold": 0.5,
        "best_for_micro_f1": best_micro,
        "best_for_macro_f1": best_macro,
        "sweep": rows,
        "cv_mean_micro_f1": cv.mean_micro_f1,
        "cv_std_micro_f1": cv.std_micro_f1,
        "cv_mean_macro_f1": cv.mean_macro_f1,
        "cv_std_macro_f1": cv.std_macro_f1,
    }

    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        path = MODELS_DIR / f"thresholds_{rollup}.json"
        path.write_text(json.dumps(payload, indent=2))
        logger.info("Saved thresholds + sweep to %s", path)

    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollup", choices=["dirty", "3char", "chapter"], default="3char")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    run(rollup=args.rollup, save=not args.no_save)


if __name__ == "__main__":
    main()
