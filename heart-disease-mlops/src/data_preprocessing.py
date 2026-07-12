"""
Data loading + the reusable preprocessing pipeline.

Design choices worth noting in your report:
  * We keep the raw '?'-style missing values handling here so the same code
    works on both the cleaned CSV and the raw UCI file.
  * Preprocessing is an sklearn ColumnTransformer wrapped in a Pipeline, so the
    *identical* transformation is applied at training and at inference time
    (this is what guarantees reproducibility – FAQ: "preprocessing pipelines
    must be reusable during inference").
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def load_data(path=None) -> pd.DataFrame:
    """Load the heart-disease CSV and coerce it into a clean, typed frame."""
    path = path or config.RAW_DATA_PATH
    df = pd.read_csv(path)
    # Strip a possible UTF-8 BOM and whitespace from column names
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    # The raw UCI file encodes missing values as '?'. Coerce everything numeric.
    df = df.replace("?", pd.NA)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Some UCI versions have target 0-4; collapse to binary (any disease = 1).
    if config.TARGET in df.columns:
        df[config.TARGET] = (df[config.TARGET] > 0).astype(int)

    return df


def split_features_target(df: pd.DataFrame):
    """Return (X, y) using only the configured feature columns."""
    missing = [c for c in config.ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    X = df[config.ALL_FEATURES].copy()
    y = df[config.TARGET].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: impute + scale numerics, impute + one-hot categoricals."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, config.NUMERIC_FEATURES),
            ("cat", categorical_pipe, config.CATEGORICAL_FEATURES),
        ]
    )
