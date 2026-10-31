"""
Shared feature engineering utilities for training and inference.

This module defines the canonical feature columns used by the trained
model, provides training-time feature construction functions, and
inference-time input validation.

Used by:
  - src/predict.py          (inference)
  - api/main.py             (API input schema)
  - tests/test_features.py  (automated tests)
"""

import os
import pandas as pd
import numpy as np

# ─── Canonical feature and target definitions ────────────────────────
# These must match exactly what the trained model expects.

FEATURE_COLUMNS = [
    "energy_current_hour",
    "energy_previous_hour",
    "energy_two_hours_ago",
    "target_hour",
    "target_day_of_week",
    "target_is_weekend",
]

TARGET_COLUMN = "target_next_hour"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "gradient_boosting.pkl")


# ─── Data validation (training pipeline) ─────────────────────────────

def validate_data(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    group_col=None,
) -> pd.DataFrame:
    """Validate and sort time-series data without mixing meters."""

    required = [timestamp_col, target_col]
    if group_col:
        required.append(group_col)

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(
        df[timestamp_col], errors="coerce"
    )
    df[target_col] = pd.to_numeric(
        df[target_col], errors="coerce"
    )

    df = df.dropna(subset=required)

    duplicate_cols = (
        [group_col, timestamp_col] if group_col else [timestamp_col]
    )
    if df.duplicated(duplicate_cols).any():
        raise ValueError(
            f"Duplicate observations found for {duplicate_cols}"
        )

    sort_cols = (
        [group_col, timestamp_col] if group_col else [timestamp_col]
    )
    return df.sort_values(sort_cols).reset_index(drop=True)


# ─── Feature construction (training pipeline) ────────────────────────

def add_time_features(
    df: pd.DataFrame,
    timestamp_col: str,
    offset_hours: int = 1,
) -> pd.DataFrame:
    """Create calendar features for the target prediction hour.

    Features describe the hour being predicted, which is
    ``offset_hours`` ahead of the current timestamp.
    """

    df = df.copy()
    target_time = df[timestamp_col] + pd.Timedelta(hours=offset_hours)

    df["target_hour"] = target_time.dt.hour
    df["target_day_of_week"] = target_time.dt.dayofweek
    df["target_is_weekend"] = (
        target_time.dt.dayofweek >= 5
    ).astype(int)

    return df


def add_history_features(
    df: pd.DataFrame,
    target_col: str,
    group_col=None,
) -> pd.DataFrame:
    """Create energy history features using past observations only."""

    df = df.copy()

    if group_col:
        grouped = df.groupby(group_col, sort=False)[target_col]
        df["energy_current_hour"] = grouped.shift(0)
        df["energy_previous_hour"] = grouped.shift(1)
        df["energy_two_hours_ago"] = grouped.shift(2)
    else:
        df["energy_current_hour"] = df[target_col]
        df["energy_previous_hour"] = df[target_col].shift(1)
        df["energy_two_hours_ago"] = df[target_col].shift(2)

    return df


def chronological_split(
    df: pd.DataFrame,
    timestamp_col: str,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
):
    """Split by shared timestamps, keeping all meters at each time together."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "Training and validation ratios must sum to less than 1"
        )

    df = df.sort_values(timestamp_col).reset_index(drop=True)
    timestamps = sorted(df[timestamp_col].dropna().unique())

    if len(timestamps) < 3:
        raise ValueError("Not enough unique timestamps to split data")

    train_end = int(len(timestamps) * train_ratio)
    validation_end = int(
        len(timestamps) * (train_ratio + validation_ratio)
    )

    train_boundary = timestamps[train_end]
    validation_boundary = timestamps[validation_end]

    train_df = df[df[timestamp_col] < train_boundary].copy()
    validation_df = df[
        (df[timestamp_col] >= train_boundary)
        & (df[timestamp_col] < validation_boundary)
    ].copy()
    test_df = df[df[timestamp_col] >= validation_boundary].copy()

    return train_df, validation_df, test_df


# ─── Inference utilities ─────────────────────────────────────────────

def validate_inference_input(data: dict) -> dict:
    """Validate and clean a single prediction input.

    Args:
        data: Dictionary with keys matching FEATURE_COLUMNS.

    Returns:
        Cleaned dictionary ready for feature vector preparation.

    Raises:
        ValueError: If required features are missing or invalid.
    """
    missing = [col for col in FEATURE_COLUMNS if col not in data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    cleaned = {}
    for col in FEATURE_COLUMNS:
        value = data[col]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            raise ValueError(f"Feature '{col}' cannot be null or NaN")

        try:
            cleaned[col] = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Feature '{col}' must be numeric, got: {type(value).__name__}"
            )

    # Range validation for calendar features.
    if not (0 <= cleaned["target_hour"] <= 23):
        raise ValueError("target_hour must be between 0 and 23")
    if not (0 <= cleaned["target_day_of_week"] <= 6):
        raise ValueError("target_day_of_week must be between 0 and 6")
    if cleaned["target_is_weekend"] not in (0.0, 1.0):
        raise ValueError("target_is_weekend must be 0 or 1")

    # Energy values should be non-negative.
    for col in [
        "energy_current_hour",
        "energy_previous_hour",
        "energy_two_hours_ago",
    ]:
        if cleaned[col] < 0:
            raise ValueError(f"{col} must be non-negative")

    return cleaned


def prepare_feature_vector(data: dict) -> pd.DataFrame:
    """Convert validated input to a model-ready DataFrame.

    Args:
        data: Dictionary with keys matching FEATURE_COLUMNS
              (already validated).

    Returns:
        Single-row DataFrame with columns in the correct order.
    """
    return pd.DataFrame([{col: data[col] for col in FEATURE_COLUMNS}])
