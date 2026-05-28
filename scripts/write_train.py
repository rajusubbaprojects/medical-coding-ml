from pathlib import Path

CONTENT = '''"""Train a multi-label baseline classifier on note text.

Model: TF-IDF (word + char n-grams) -> OneVsRest(LogisticRegression).
Loads data via load_data; saves bundle (model + vectorizer + labels) to disk.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from src.baseline.load_data import load_data, DataSplit

MODELS_DIR = Path("models")

logger = logging.getLogger(__name__)


@dataclass
class TrainedBundle:
    pipeline: Pipeline
    label_names: list[str]
    info: dict[str, Any]


def build_pipeline() -> Pipeline:
    """TF-IDF features + OneVsRest logistic regression.

    Word n-grams capture phrases ('acute bronchitis').
    Char n-grams capture morphology + small typos ('bronchit-').
    Stacked as a FeatureUnion, fed to a one-vs-rest LR head.
    """
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
    """Fit a baseline classifier on the chosen rollup."""
    logger.info("Loading data (rollup=%s)", rollup)
    ds: DataSplit = load_data(rollup=rollup)
    logger.info("Train: %d notes, %d labels", ds.info["n_train"], ds.info["n_codes_kept"])

    pipeline = build_pipeline()
    t0 = time.time()
    pipeline.fit(ds.X_train, ds.y_train)
    fit_seconds = time.time() - t0
    logger.info("Trained in %.1fs", fit_seconds)

    info = dict(ds.info)
    info["fit_seconds"] = fit_seconds
    bundle = TrainedBundle(
        pipeline=pipeline,
        label_names=ds.label_names,
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