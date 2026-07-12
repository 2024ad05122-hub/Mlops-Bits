"""Unit tests for data loading + preprocessing (assignment task 5)."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_preprocessing import (
    build_preprocessor,
    load_data,
    split_features_target,
)


@pytest.fixture(scope="module")
def df():
    return load_data()


def test_load_data_not_empty(df):
    assert len(df) > 0
    assert config.TARGET in df.columns


def test_all_features_present(df):
    for feat in config.ALL_FEATURES:
        assert feat in df.columns, f"missing feature {feat}"


def test_target_is_binary(df):
    assert set(df[config.TARGET].unique()).issubset({0, 1})


def test_split_features_target_shapes(df):
    X, y = split_features_target(df)
    assert X.shape[0] == y.shape[0]
    assert list(X.columns) == config.ALL_FEATURES


def test_preprocessor_output_is_numeric(df):
    X, y = split_features_target(df)
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    arr = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    assert arr.shape[0] == X.shape[0]
    assert np.isfinite(arr).all(), "preprocessed features contain NaN/inf"


def test_preprocessor_handles_missing_values():
    # inject a NaN and confirm the pipeline imputes it without error
    X = pd.DataFrame(
        [{f: 1.0 for f in config.ALL_FEATURES}]
        + [{f: (np.nan if f == "chol" else 2.0) for f in config.ALL_FEATURES}]
    )
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    arr = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    assert np.isfinite(arr).all()
