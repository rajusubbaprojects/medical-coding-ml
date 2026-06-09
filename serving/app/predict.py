"""Prediction logic for the serving layer.

Loads model bundles from /models (mounted or copied into container).
Exposes a predict() function used by the FastAPI app.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
DESCRIPTIONS_PATH = Path(os.environ.get("DESCRIPTIONS_PATH",
                         "data/icd10_3char_descriptions.csv"))

MODEL_BUNDLES = {
    "encounter": "baseline_encounter.joblib",
    "patient":   "baseline_3char.joblib",
}

_cache: dict[str, Any] = {}


def _load_bundle(model_name: str) -> dict:
    if model_name not in _cache:
        path = MODELS_DIR / MODEL_BUNDLES[model_name]
        if not path.exists():
            raise FileNotFoundError(f"Model bundle not found: {path}")
        _cache[model_name] = joblib.load(path)
    return _cache[model_name]


def _load_descriptions() -> dict[str, str]:
    if "descriptions" not in _cache:
        if not DESCRIPTIONS_PATH.exists():
            _cache["descriptions"] = {}
        else:
            with open(DESCRIPTIONS_PATH) as f:
                reader = csv.DictReader(f)
                _cache["descriptions"] = {
                    row["code_3char"]: row["description"] for row in reader
                }
    return _cache["descriptions"]


def predict(note_text: str, model_name: str = "encounter",
            threshold: float = 0.5) -> list[dict]:
    bundle = _load_bundle(model_name)
    pipe = bundle["pipeline"]
    label_names = bundle["label_names"]
    descriptions = _load_descriptions()

    proba = pipe.predict_proba([note_text])[0]

    results = []
    for code, prob in zip(label_names, proba):
        if prob >= threshold:
            results.append({
                "code": code,
                "confidence": round(float(prob), 4),
                "description": descriptions.get(code, ""),
            })

    return sorted(results, key=lambda x: -x["confidence"])


def available_models() -> list[str]:
    return [
        name for name, fname in MODEL_BUNDLES.items()
        if (MODELS_DIR / fname).exists()
    ]
