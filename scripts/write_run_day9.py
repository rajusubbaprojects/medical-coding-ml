from pathlib import Path

src = Path("src/baseline/run_day9.py")
src.write_text(
"""\"\"\"Day 9: re-run CV and save OOF predictions for per-code analysis.\"\"\"

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.baseline.cv import cross_validate, print_cv_report

logger = logging.getLogger(__name__)

OOF_DIR = Path("data/oof")


def run(rollup: str = "3char", n_folds: int = 5, seed: int = 42) -> None:
    logger.info("Running %d-fold CV on rollup=%s", n_folds, rollup)
    result = cross_validate(rollup=rollup, n_folds=n_folds, seed=seed)
    print_cv_report(result)

    OOF_DIR.mkdir(parents=True, exist_ok=True)

    n_notes, n_codes = result.oof_y_true.shape
    rows = []
    for note_idx in range(n_notes):
        for code_idx, code_name in enumerate(result.label_names):
            rows.append({
                "note_idx": note_idx,
                "code": code_name,
                "y_true": int(result.oof_y_true[note_idx, code_idx]),
                "y_pred": int(result.oof_y_pred[note_idx, code_idx]),
                "y_proba": float(result.oof_y_proba[note_idx, code_idx]),
            })

    df = pd.DataFrame(rows)
    out_path = OOF_DIR / f"oof_{rollup}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved OOF predictions: %s (%d rows)", out_path, len(df))
    print(f"\\nOOF CSV written to {out_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollup", choices=["dirty", "3char", "chapter"], default="3char")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(rollup=args.rollup, n_folds=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
"""
)
print("wrote src/baseline/run_day9.py")
