"""
Exploratory Data Analysis for the Heart Disease dataset (assignment task 1).

Produces professional visualisations and saves them to reports/figures/:
    * class_balance.png       – target distribution
    * histograms.png          – distribution of numeric features
    * correlation_heatmap.png – feature correlation matrix
    * missing_values.png      – missing-value counts per column
    * feature_by_target.png   – key numeric features split by target

Run:  python -m src.eda
Tip:  This is a script for reproducibility; you can also paste these cells
      into a Jupyter notebook (notebooks/eda.ipynb) and personalise the
      commentary for your report.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.data_preprocessing import load_data

sns.set_theme(style="whitegrid")


def plot_class_balance(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    order = sorted(df[config.TARGET].unique())
    sns.countplot(x=config.TARGET, data=df, order=order, ax=ax, hue=config.TARGET,
                  palette="Set2", legend=False)
    ax.set_title("Class Balance (0 = no disease, 1 = disease)")
    ax.set_xlabel("target"); ax.set_ylabel("count")
    _save(fig, "class_balance.png")


def plot_histograms(df):
    n = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(1, len(n), figsize=(4 * len(n), 3.5))
    for ax, col in zip(axes, n):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#4C72B0")
        ax.set_title(col)
    fig.suptitle("Numeric Feature Distributions", y=1.03)
    _save(fig, "histograms.png")


def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap")
    _save(fig, "correlation_heatmap.png")


def plot_missing(df):
    missing = df.isnull().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=missing.index, y=missing.values, ax=ax, color="#C44E52")
    ax.set_title(f"Missing Values per Column (total = {int(missing.sum())})")
    ax.set_ylabel("missing count")
    plt.xticks(rotation=45, ha="right")
    _save(fig, "missing_values.png")


def plot_feature_by_target(df):
    feats = config.NUMERIC_FEATURES[:4]
    fig, axes = plt.subplots(1, len(feats), figsize=(4 * len(feats), 3.5))
    for ax, col in zip(axes, feats):
        sns.boxplot(x=config.TARGET, y=col, data=df, ax=ax, hue=config.TARGET,
                    palette="Set3", legend=False)
        ax.set_title(f"{col} by target")
    fig.suptitle("Feature Relationship with Target", y=1.03)
    _save(fig, "feature_by_target.png")


def _save(fig, name):
    fig.tight_layout()
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main():
    df = load_data()
    print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nSummary statistics:\n", df.describe().T.round(2))
    print("\nMissing values:\n", df.isnull().sum())
    print("\nGenerating figures...")
    plot_class_balance(df)
    plot_histograms(df)
    plot_correlation(df)
    plot_missing(df)
    plot_feature_by_target(df)
    print("EDA complete.")


if __name__ == "__main__":
    main()
