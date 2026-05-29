from pathlib import Path

CONTENT = '''"""Train a multi-label baseline classifier on the DEV bucket.

Model: TF-IDF (word + char n-grams) -> OneVsRest(LogisticRegression).
Loads dev bucket via load_data; saves bundle (model + vectorizer + labels) to disk.
Test bucket is held out and never seen here.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from src.baseline.load_data import load_data, LoadedBucket

MODELS_DIR = Path("models")

logger = logging.getLogger(__name__)


@dataclass
class TrainedBundle:
    pipeline: Pipeline
    label_names: list[str]
    info: dict[str, Any]


def build_pipeline() -> Pipeline:
    """TF-IDF features + OneVsRest logistic regression."""
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=20000,
        sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        max_features=20000,
        sublinear_tf=True,
    )
    features = FeatureUnion([("word", word_vec), ("char", char_vec)])
    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
        ),
        n_jobs=1,
    )
    return Pipeline([("features", features), ("clf", clf)])


def train(rollup: str = "3char", *, save: bool = True) -> TrainedBundle:
    """Fit a baseline classifier on the dev bucket of the chosen rollup."""
    logger.info("Loading dev bucket (rollup=%s)", rollup)
    dev: LoadedBucket = load_data(rollup=rollup, bucket="dev")
    logger.info("Train: %d notes, %d labels", dev.info["n_notes"], dev.info["n_codes_kept"])

    pipeline = build_pipeline()
    t0 = time.time()
    pipeline.fit(dev.X, dev.y)
    fit_seconds = time.time() - t0
    logger.info("Trained in %.1fs", fit_seconds)

    info = dict(dev.info)
    info["fit_seconds"] = fit_seconds
    bundle = TrainedBundle(
        pipeline=pipeline,
        label_names=dev.label_names,
        info=info,
    )

    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        path = MODELS_DIR / f"baseline_{rollup}.joblib"
        joblib.dump(
            {
                "pipeline": bundle.pipeline,
                "label_names": bundle.label_names,
                "info": bundle.info,
            },
            path,
        )
        logger.info("Saved bundle to %s", path)

    return bundle


def load_bundle(rollup: str = "3char") -> TrainedBundle:
    path = MODELS_DIR / f"baseline_{rollup}.joblib"
    raw = joblib.load(path)
    return TrainedBundle(
        pipeline=raw["pipeline"],
        label_names=raw["label_names"],
        info=raw["info"],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollup", choices=["dirty", "3char", "chapter"], default="3char")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    bundle = train(rollup=args.rollup, save=not args.no_save)
    print(f"\\n--- bundle info (rollup={args.rollup}) ---")
    for k, v in bundle.info.items():
        if isinstance(v, float):
            print(f"  {k:25s} {v:.3f}")
        else:
            print(f"  {k:25s} {v}")
    print(f"  n_labels                  {len(bundle.label_names)}")


if __name__ == "__main__":
    main()
'''

Path("src/baseline/train.py").write_text(CONTENT)
print("Wrote src/baseline/train.py")
print(f"  {len(CONTENT.splitlines())} lines")