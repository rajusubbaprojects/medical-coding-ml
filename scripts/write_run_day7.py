from pathlib import Path

CONTENT = '''"""Day 7 driver: CV + threshold sweep + consolidated report.

Runs:
  1. 5-fold CV on the 3char rollup, dev bucket
  2. Threshold sweep on the OOF probabilities
  3. Prints a consolidated report with the headline numbers
  4. Saves thresholds + curve to models/thresholds_3char.json

Does NOT touch the held-out test bucket.
"""

from __future__ import annotations

import argparse
import logging

from src.baseline.cv import cross_validate, print_cv_report
from src.baseline.threshold import (
    sweep_thresholds,
    pick_best,
    print_curve,
)

logger = logging.getLogger(__name__)


def run(rollup: str = "3char") -> None:
    print("=" * 64)
    print(f"DAY 7 RESULTS  (rollup={rollup}, dev bucket only)")
    print("=" * 64)

    # 1. Cross-validation
    cv = cross_validate(rollup=rollup)
    print_cv_report(cv)

    # 2. Threshold sweep on OOF predictions
    print("\\n" + "-" * 64)
    print("THRESHOLD SWEEP  (operating on out-of-fold probabilities)")
    print("-" * 64)
    sweep = sweep_thresholds(cv.oof_y_true, cv.oof_y_proba)
    best_micro = pick_best(sweep, "micro_f1")
    best_macro = pick_best(sweep, "macro_f1")
    print_curve(sweep, best_micro["threshold"], best_macro["threshold"])

    # 3. Headline summary
    print("\\n" + "=" * 64)
    print("HEADLINE")
    print("=" * 64)
    print(f"  CV mean micro F1 (threshold=0.5):  "
          f"{cv.mean_micro_f1:.3f} +/- {cv.std_micro_f1:.3f}")
    print(f"  CV mean macro F1 (threshold=0.5):  "
          f"{cv.mean_macro_f1:.3f} +/- {cv.std_macro_f1:.3f}")
    print()
    print(f"  Best operating point for MICRO F1:")
    print(f"    threshold = {best_micro['threshold']:.2f}")
    print(f"    micro F1  = {best_micro['micro_f1']:.3f} "
          f"(P={best_micro['micro_precision']:.3f}, "
          f"R={best_micro['micro_recall']:.3f})")
    print()
    print(f"  Best operating point for MACRO F1:")
    print(f"    threshold = {best_macro['threshold']:.2f}")
    print(f"    macro F1  = {best_macro['macro_f1']:.3f} "
          f"(P={best_macro['macro_precision']:.3f}, "
          f"R={best_macro['macro_recall']:.3f})")
    print()
    print("  Note: test bucket NOT used. Final evaluation deferred to Day 11-12.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollup", choices=["dirty", "3char", "chapter"], default="3char")
    args = parser.parse_args()
    run(rollup=args.rollup)


if __name__ == "__main__":
    main()
'''

Path("src/baseline/run_day7.py").write_text(CONTENT)
print("Wrote src/baseline/run_day7.py")
print(f"  {len(CONTENT.splitlines())} lines")