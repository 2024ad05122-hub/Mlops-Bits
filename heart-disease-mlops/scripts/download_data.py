"""
Download the Heart Disease dataset.

Primary source: UCI ML Repository via the official `ucimlrepo` package
(dataset id = 45). Falls back to a public GitHub mirror if UCI is unreachable.

Usage:
    python scripts/download_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "heart.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

MIRROR = (
    "https://raw.githubusercontent.com/kb22/"
    "Heart-Disease-Prediction/master/dataset.csv"
)


def from_ucimlrepo() -> pd.DataFrame | None:
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=45)               # Heart Disease
        X = ds.data.features
        y = ds.data.targets
        df = pd.concat([X, y], axis=1)
        df.columns = COLUMNS
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] ucimlrepo failed ({exc}); trying GitHub mirror...")
        return None


def from_mirror() -> pd.DataFrame:
    df = pd.read_csv(MIRROR)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    return df


def main() -> int:
    df = from_ucimlrepo()
    if df is None:
        df = from_mirror()
    # Collapse any 0-4 target to binary
    df["target"] = (pd.to_numeric(df["target"], errors="coerce") > 0).astype(int)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows, {df.shape[1]} columns -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
