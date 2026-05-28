"""Day 6 driver: train + evaluate all rollups, print a comparison table.

Answers the Day 6 question: did data cleaning (Z-filter + hierarchy rollup)
actually help? Compares dirty vs 3char vs chapter on the same metrics.
"""

from __future__ import annotations

import argparse
import logging

from src.baseline.train import train
from src.baseline.evaluate import evaluate, print_report

logger = logging.getLogger(__name__)

ROLLUPS = ["dirty", "3char", "chapter"]


def run(rollups: list[str], *, verbose: bool) -> None:
    results = []
    for rollup in rollups:
        logger.info("=== Training + evaluating: %s ===", rollup)
        train(rollup=rollup, save=True)
        metrics = evaluate(rollup=rollup)
        results.append(metrics)
        if verbose:
            print_report(metrics)

    # Comparison table
    print("\n" + "=" * 64)
    print("COMPARISON")
    print("=" * 64)
    header = f"{'rollup':10s} {'n_codes':>8s} {'micro_f1':>9s} {'macro_f1':>9s} {'subset':>7s} {'hamming':>8s}"
    print(header)
    print("-" * len(header))
    for m in results:
        print(f"{m['rollup']:10s} {m['n_codes']:>8d} "
              f"{m['micro_f1']:>9.3f} {m['macro_f1']:>9.3f} "
              f"{m['subset_accuracy']:>7.3f} {m['hamming_loss']:>8.3f}")
    print()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollups", nargs="+", default=ROLLUPS,
                        choices=ROLLUPS)
    parser.add_argument("--verbose", action="store_true",
                        help="Print full per-code report for each rollup")
    args = parser.parse_args()
    run(args.rollups, verbose=args.verbose)


if __name__ == "__main__":
    main()
