"""
src/data.py
-----------
Data loading, auditing and splitting utilities.

All random_state arguments default to 42 for reproducibility.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "creditcard.csv"


# ─── Loading ──────────────────────────────────────────────────────────────────
def load_raw(path: pathlib.Path = RAW_CSV) -> pd.DataFrame:
    """Load the raw creditcard.csv and return a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place creditcard.csv inside the data/ folder."
        )
    return pd.read_csv(path)


# ─── Auditing ─────────────────────────────────────────────────────────────────
def audit(df: pd.DataFrame) -> dict:
    """
    Run a basic data-quality audit.

    Returns a dict with shape, dtypes summary, missing-value counts,
    duplicate-row count, and class balance.
    """
    n_rows, n_cols = df.shape
    n_missing = df.isnull().sum()
    n_missing_total = int(n_missing.sum())
    n_duplicates = int(df.duplicated().sum())

    class_counts = df["Class"].value_counts().to_dict()
    fraud_rate = class_counts.get(1, 0) / n_rows

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_per_column": n_missing[n_missing > 0].to_dict(),
        "total_missing": n_missing_total,
        "duplicate_rows": n_duplicates,
        "class_counts": class_counts,
        "fraud_rate_pct": round(fraud_rate * 100, 4),
    }


def print_audit(info: dict) -> None:
    """Pretty-print the audit dict."""
    print("=" * 55)
    print("DATA AUDIT")
    print("=" * 55)
    print(f"  Rows        : {info['n_rows']:,}")
    print(f"  Columns     : {info['n_cols']}")
    print(f"  Missing vals: {info['total_missing']}")
    print(f"  Duplicates  : {info['duplicate_rows']}")
    print(f"  Genuine     : {info['class_counts'].get(0, 0):,}")
    print(f"  Fraud       : {info['class_counts'].get(1, 0):,}")
    print(f"  Fraud rate  : {info['fraud_rate_pct']} %")
    print("=" * 55)


# ─── Splitting ────────────────────────────────────────────────────────────────
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows (same feature values AND same label)."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} duplicate rows ({before} -> {after}).")
    return df


def stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70/15/15 split preserving class balance in every subset.

    Returns (train_df, val_df, test_df).
    The test set is returned but should NOT be touched until final evaluation.
    """
    # First split off the test set
    remaining_size = val_size + test_size  # fraction of full dataset
    X_temp, X_test, y_temp, y_test = train_test_split(
        df.drop(columns=["Class"]),
        df["Class"],
        test_size=test_size,
        stratify=df["Class"],
        random_state=random_state,
    )

    # From the remaining, split off validation
    val_fraction_of_remaining = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_fraction_of_remaining,
        stratify=y_temp,
        random_state=random_state,
    )

    train_df = X_train.copy()
    train_df["Class"] = y_train.values

    val_df = X_val.copy()
    val_df["Class"] = y_val.values

    test_df = X_test.copy()
    test_df["Class"] = y_test.values

    print(
        f"Split sizes  -> train: {len(train_df):,}  "
        f"val: {len(val_df):,}  "
        f"test: {len(test_df):,}"
    )
    for name, subset in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rate = subset["Class"].mean() * 100
        print(f"  {name} fraud rate: {rate:.4f} %")

    return train_df, val_df, test_df


def time_ordered_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological (time-ordered) split — robustness check only.

    Splits by the 'Time' column so earlier transactions are always in the
    training set. This checks whether results degrade under temporal ordering
    (distribution shift is possible over the two-day window).
    """
    df_sorted = df.sort_values("Time").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * (1.0 - val_frac - test_frac))
    val_end = int(n * (1.0 - test_frac))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    print("Time-ordered split sizes:")
    for name, subset in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rate = subset["Class"].mean() * 100
        print(f"  {name}: {len(subset):,} rows  fraud rate: {rate:.4f} %")

    return train_df, val_df, test_df
