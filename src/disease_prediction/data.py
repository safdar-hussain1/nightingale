"""Data loading and validation."""

from __future__ import annotations

import pandas as pd

from .config import FEATURES, ID_COLUMN, TARGET


def load_train(path) -> tuple[pd.DataFrame, pd.Series]:
    """Load training data and return (features, target)."""
    df = pd.read_csv(path)
    validate(df, require_target=True)
    return df[FEATURES], df[TARGET]


def load_test(path) -> tuple[pd.DataFrame, pd.Series]:
    """Load test data and return (features, patient ids)."""
    df = pd.read_csv(path)
    validate(df, require_target=False)
    return df[FEATURES], df[ID_COLUMN]


def validate(df: pd.DataFrame, require_target: bool) -> None:
    """Fail fast on schema problems instead of producing silent garbage."""
    missing_cols = [c for c in FEATURES + [ID_COLUMN] if c not in df.columns]
    if require_target and TARGET not in df.columns:
        missing_cols.append(TARGET)
    if missing_cols:
        raise ValueError(f"Input data is missing columns: {missing_cols}")
    if df[FEATURES].isna().any().any():
        raise ValueError("Feature columns contain missing values")
    if require_target and not df[TARGET].isin([0, 1]).all():
        raise ValueError(f"{TARGET} must be binary (0/1)")
