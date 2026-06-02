from pathlib import Path

path = Path("src/baseline/per_code_analysis.py")
src = path.read_text()

# 1. Add --min-support arg
old_arg = '    parser.add_argument("--save-csv", action="store_true",'
new_arg = '''    parser.add_argument("--min-support", type=int, default=0,
                        help="Exclude codes with support < N from metric summary")
    parser.add_argument("--save-csv", action="store_true",'''
src = src.replace(old_arg, new_arg)

# 2. Pass min_support into print_report call
old_call = '    print_report(stats, n_worst=args.worst, n_best=args.best)'
new_call = '    print_report(stats, n_worst=args.worst, n_best=args.best, min_support=args.min_support)'
src = src.replace(old_call, new_call)

# 3. Add min_support param + filtered summary block to print_report
old_def = 'def print_report(stats: pd.DataFrame, n_worst: int = 20, n_best: int = 10) -> None:'
new_def = 'def print_report(stats: pd.DataFrame, n_worst: int = 20, n_best: int = 10, min_support: int = 0) -> None:'
src = src.replace(old_def, new_def)

old_bucket = '    # Support buckets — does support predict F1?'
new_filtered = '''    # Filtered summary (min_support threshold)
    if min_support > 0:
        filtered = stats[stats["support"] >= min_support]
        excluded = stats[stats["support"] < min_support]
        import numpy as np
        y_true_all = None  # recompute from filtered codes only not available here; use mean
        print(f"\\n--- Filtered summary (support >= {min_support}) ---")
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

    # Support buckets — does support predict F1?'''
src = src.replace(old_bucket, new_filtered)

path.write_text(src)
print("patched src/baseline/per_code_analysis.py")
