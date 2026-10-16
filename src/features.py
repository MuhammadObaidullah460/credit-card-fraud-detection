"""
src/features.py
---------------
Feature engineering and preprocessing pipeline builders.

Design principles:
- Scalers are fitted ONLY on training data and applied to validation/test.
- All transformations are wrapped in sklearn Pipelines / ColumnTransformers
  so that when used inside cross-validation no leakage can occur.
- V1-V28 are already PCA-transformed and roughly standardized; we leave them
  as-is.  Only 'Amount' and 'Time' require scaling.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer


# ─── Column groups ────────────────────────────────────────────────────────────
V_COLS = [f"V{i}" for i in range(1, 29)]   # PCA features — no scaling needed
SCALE_COLS = ["Amount", "Time"]              # raw scale; need standardization
FEATURE_COLS = SCALE_COLS + V_COLS          # full feature list (30 features)
TARGET_COL = "Class"


def get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return (X, y) from a DataFrame that contains the target column.

    Parameters
    ----------
    df : pd.DataFrame
        A split (train / val / test) that still contains the 'Class' column.
    feature_cols : list[str] | None
        Which columns to use as features. Defaults to FEATURE_COLS.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series  (dtype int)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int)
    return X, y


def build_preprocessor(scaler_type: str = "standard") -> ColumnTransformer:
    """
    Build a ColumnTransformer that scales Amount and Time only.

    Parameters
    ----------
    scaler_type : str
        'standard' → StandardScaler  (zero mean, unit variance)
        'robust'   → RobustScaler    (median / IQR — less sensitive to outliers)

    Returns
    -------
    ColumnTransformer that passes V1-V28 through unchanged.
    """
    if scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", scaler, SCALE_COLS),
        ],
        remainder="passthrough",   # V1-V28 pass through unchanged
    )
    return preprocessor


def build_pipeline(classifier, scaler_type: str = "standard") -> Pipeline:
    """
    Wrap a classifier inside a preprocessing + classification Pipeline.

    The Pipeline ensures the scaler is never fitted on validation / test data
    even when used inside cross_val_score or cross_validate.

    Parameters
    ----------
    classifier : sklearn-compatible estimator
    scaler_type : str  'standard' or 'robust'

    Returns
    -------
    sklearn Pipeline
    """
    preprocessor = build_preprocessor(scaler_type=scaler_type)
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipe


def add_hour_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive 'Hour' from 'Time' (seconds since the first transaction in the
    dataset).  Used only for EDA — not added to model features to avoid
    redundancy with the already-present Time column.
    """
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600) % 24
    return df
