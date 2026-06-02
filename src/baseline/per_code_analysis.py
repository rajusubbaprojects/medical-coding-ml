"""Per-code F1 analysis on OOF predictions.

Loads data/oof/oof_3char.csv and prints a ranked breakdown:
  - support (how many notes truly have this code)
  - precision, recall, F1
  - false negative rate, false positive rate
Sorted by F1 ascending so worst codes appear first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


OOF_DIR = Path("data/oof")


def load_oof(rollup: str = "3char") -> pd.DataFrame:
    path = OOF_DIR / f"oof_{rollup}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No OOF file at {path}. Run run_day9.py first.")
    return pd.read_csv(path)


def per_code_stats(df: pd.DataFrame) -> pd.DataFrame:
    codes = df["code"].unique()
    codes.sort()

    rows = []
    for code in codes:
        sub = df[df["code"] == code]
        y_true = sub["y_true"].values
        y_pred = sub["y_pred"].values

        support = int(y_true.sum())
        pred_positive = int(y_pred.sum())

        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        # False negatives: truly positive, predicted negative
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        # False positives: truly negative, predicted positive
        fp = int(((y_true == 0) & (y_pred == 1)).sum())

        rows.append({
            "code": code,
            "support": support,
            "pred_positive": pred_positive,
            "precision": round(float(p), 3),
            "recall": round(float(r), 3),
            "f1": round(float(f1), 3),
            "fn": fn,
            "fp": fp,
        })

    return pd.DataFrame(rows).sort_values("f1", ascending=True).reset_index(drop=True)


def print_report(stats: pd.DataFrame, n_worst: int = 20, n_best: int = 10, min_support: int = 0) -> None:
    total_codes = len(stats)
    zero_f1 = (stats["f1"] == 0).sum()
    low_f1 = (stats["f1"] < 0.5).sum()

    print(f"\n=== Per-code F1 summary (OOF, 3char rollup) ===")
    print(f"  Total codes : {total_codes}")
    print(f"  F1 == 0     : {zero_f1}")
    print(f"  F1 < 0.5    : {low_f1}")
    print(f"  F1 >= 0.8   : {(stats['f1'] >= 0.8).sum()}")

    print(f"\n--- Worst {n_worst} codes (lowest F1) ---")
    print(f"  {'code':<8} {'support':>8} {'pred+':>6} {'precision':>10} {'recall':>8} {'f1':>6} {'fn':>5} {'fp':>5}")
    for _, row in stats.head(n_worst).iterrows():
        print(f"  {row['code']:<8} {row['support']:>8} {row['pred_positive']:>6} "
              f"{row['precision']:>10.3f} {row['recall']:>8.3f} {row['f1']:>6.3f} "
              f"{row['fn']:>5} {row['fp']:>5}")

    print(f"\n--- Best {n_best} codes (highest F1) ---")
    print(f"  {'code':<8} {'support':>8} {'pred+':>6} {'precision':>10} {'recall':>8} {'f1':>6} {'fn':>5} {'fp':>5}")
    for _, row in stats.tail(n_best).iloc[::-1].iterrows():
        print(f"  {row['code']:<8} {row['support']:>8} {row['pred_positive']:>6} "
              f"{row['precision']:>10.3f} {row['recall']:>8.3f} {row['f1']:>6.3f} "
              f"{row['fn']:>5} {row['fp']:>5}")

    # Filtered summary (min_support threshold)
    if min_support > 0:
        filtered = stats[stats["support"] >= min_support]
        excluded = stats[stats["support"] < min_support]
        import numpy as np
        y_true_all = None  # recompute from filtered codes only not available here; use mean
        print(f"\n--- Filtered summary (support >= {min_support}) ---")
        print(f"  Codes included : {len(filtered)} / {total_codes}")
        print(f"  Codes excluded : {len(excluded)} (support < {min_support})")
        print(f"  Mean F1 (included codes) : {filtered['f1'].mean():.3f}")
        print(f"  Mean F1 (excluded codes) : {excluded['f1'].mean():.3f}")
        # Weighted micro-F1 proxy: sum(TP) / (sum(TP) + 0.5*sum(FP+FN))
        def micro_f1_from_stats(df):
            tp = (df["support"] - df["fn"]).sum()
            fp = df["fp"].sum()
            fn = df["fn"].sum()
            denom = tp + 0.5 * (fp + fn)
            return tp / denom if denom > 0 else 0.0
        print(f"  Micro F1 proxy (included): {micro_f1_from_stats(filtered):.3f}")
        print(f"  Micro F1 proxy (all codes): {micro_f1_from_stats(stats[stats['support']>0]):.3f}")

    # Support buckets — does support predict F1?
    print(f"\n--- F1 by support bucket ---")
    bins = [0, 5, 15, 30, 50, 9999]
    labels = ["1-5", "6-15", "16-30", "31-50", "51+"]
    stats2 = stats[stats["support"] > 0].copy()
    stats2["bucket"] = pd.cut(stats2["support"], bins=bins, labels=labels)
    bucket_summary = (
        stats2.groupby("bucket", observed=True)["f1"]
        .agg(["count", "mean", "min", "max"])
        .rename(columns={"count": "n_codes", "mean": "mean_f1", "min": "min_f1", "max": "max_f1"})
    )
    print(f"  {'support':>10} {'n_codes':>8} {'mean_f1':>9} {'min_f1':>8} {'max_f1':>8}")
    for bucket, row in bucket_summary.iterrows():
        print(f"  {str(bucket):>10} {int(row['n_codes']):>8} {row['mean_f1']:>9.3f} "
              f"{row['min_f1']:>8.3f} {row['max_f1']:>8.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollup", choices=["dirty", "3char", "chapter"], default="3char")
    parser.add_argument("--worst", type=int, default=20)
    parser.add_argument("--best", type=int, default=10)
    parser.add_argument("--min-support", type=int, default=0,
                        help="Exclude codes with support < N from metric summary")
    parser.add_argument("--save-csv", action="store_true",
                        help="Save per-code stats to data/oof/per_code_stats_<rollup>.csv")
    args = parser.parse_args()

    df = load_oof(args.rollup)
    stats = per_code_stats(df)
    print_report(stats, n_worst=args.worst, n_best=args.best, min_support=args.min_support)

    if args.save_csv:
        out = OOF_DIR / f"per_code_stats_{args.rollup}.csv"
        stats.to_csv(out, index=False)
        print(f"\nSaved per-code stats to {out}")


if __name__ == "__main__":
    main()
