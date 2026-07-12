"""
Train and compare classifiers for heart-disease prediction.

Covers assignment tasks 2, 3 and 4:
  * Two models (Logistic Regression + Random Forest) with hyper-parameter tuning
  * Cross-validation + accuracy / precision / recall / F1 / ROC-AUC
  * MLflow experiment tracking (params, metrics, plots, model artifacts)
  * Saves the best FULL pipeline (preprocessing + model) with joblib

Run:  python -m src.train
Then: mlflow ui   (open http://127.0.0.1:5000)
"""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend for CI / servers
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.data_preprocessing import (
    build_preprocessor,
    load_data,
    split_features_target,
)


# --------------------------------------------------------------------------- #
# Model zoo – add / edit models here to personalise your experiments.
# Each entry: (estimator, param_grid for GridSearchCV over the "model" step)
# --------------------------------------------------------------------------- #
def get_model_grid():
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=config.RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5],
            },
        ),
    }


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def _save_plots(pipeline, X_test, y_test, name) -> list:
    """Create ROC curve + confusion matrix, return saved file paths."""
    paths = []

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.set_title(f"ROC Curve – {name}")
    roc_path = config.FIGURES_DIR / f"roc_{name}.png"
    fig.tight_layout(); fig.savefig(roc_path, dpi=120); plt.close(fig)
    paths.append(roc_path)

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {name}")
    cm_path = config.FIGURES_DIR / f"cm_{name}.png"
    fig.tight_layout(); fig.savefig(cm_path, dpi=120); plt.close(fig)
    paths.append(cm_path)

    return paths


def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    df = load_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y,
    )

    results = {}
    for name, (estimator, grid) in get_model_grid().items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline(
                steps=[("preprocessor", build_preprocessor()), ("model", estimator)]
            )
            search = GridSearchCV(
                pipe, grid, cv=config.CV_FOLDS,
                scoring="roc_auc", n_jobs=-1, refit=True,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_

            # Cross-validated ROC-AUC on the training folds (reported metric)
            cv_auc = cross_val_score(
                best, X_train, y_train, cv=config.CV_FOLDS, scoring="roc_auc"
            )

            y_pred = best.predict(X_test)
            y_proba = best.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, y_pred, y_proba)
            metrics["cv_roc_auc_mean"] = float(np.mean(cv_auc))
            metrics["cv_roc_auc_std"] = float(np.std(cv_auc))

            # ---- MLflow logging ----
            mlflow.log_params(search.best_params_)
            mlflow.log_param("model_type", name)
            mlflow.log_param("cv_folds", config.CV_FOLDS)
            mlflow.log_metrics(metrics)

            for p in _save_plots(best, X_test, y_test, name):
                mlflow.log_artifact(str(p), artifact_path="plots")

            mlflow.sklearn.log_model(
                best, name="model", serialization_format="cloudpickle"
            )

            results[name] = {"estimator": best, "metrics": metrics}
            print(f"[{name}] test metrics: "
                  + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    # --------------------------------------------------------------------- #
    # Select best model by test ROC-AUC and persist the FULL pipeline
    # --------------------------------------------------------------------- #
    best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    best_pipeline = results[best_name]["estimator"]
    joblib.dump(best_pipeline, config.MODEL_PATH)

    metadata = {
        "best_model": best_name,
        "metrics": results[best_name]["metrics"],
        "features": config.ALL_FEATURES,
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "sklearn_version": __import__("sklearn").__version__,
        "python_version": platform.python_version(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nBest model: {best_name} "
          f"(ROC-AUC={results[best_name]['metrics']['roc_auc']:.3f})")
    print(f"Saved pipeline -> {config.MODEL_PATH}")
    print(f"Saved metadata -> {config.METADATA_PATH}")


if __name__ == "__main__":
    main()
