"""
Offline / batch inference using the saved pipeline (no API needed).

Examples:
    # single JSON record
    python -m src.predict --json '{"age":63,"sex":1,"cp":3,"trestbps":145,
        "chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,
        "slope":0,"ca":0,"thal":1}'

    # a CSV file of records (same columns as the features)
    python -m src.predict --csv data/new_patients.csv
"""
from __future__ import annotations

import argparse
import json
import sys

import joblib
import pandas as pd

from src import config

LABELS = {0: "no_disease", 1: "disease"}


def load_model(path=None):
    return joblib.load(path or config.MODEL_PATH)


def predict_frame(model, X: pd.DataFrame) -> pd.DataFrame:
    X = X[config.ALL_FEATURES].copy()
    proba = model.predict_proba(X)
    preds = proba.argmax(axis=1)
    out = X.copy()
    out["prediction"] = preds
    out["label"] = [LABELS[p] for p in preds]
    out["confidence"] = proba.max(axis=1).round(4)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Heart-disease inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--json", help="single JSON record")
    group.add_argument("--csv", help="path to CSV of records")
    args = parser.parse_args(argv)

    model = load_model()

    if args.json:
        record = json.loads(args.json)
        X = pd.DataFrame([record])
    else:
        X = pd.read_csv(args.csv)

    result = predict_frame(model, X)
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
