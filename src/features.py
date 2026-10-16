import pandas as pd


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


def add_time_features(
    df: pd.DataFrame,
    timestamp_col: str,
) -> pd.DataFrame:
    """Create calendar features from a timestamp."""

    df = df.copy()
    timestamp = df[timestamp_col]

    df["hour"] = timestamp.dt.hour
    df["day_of_week"] = timestamp.dt.dayofweek
    df["month"] = timestamp.dt.month
    df["is_weekend"] = (
        timestamp.dt.dayofweek >= 5
    ).astype(int)

    return df


def add_history_features(
    df: pd.DataFrame,
    target_col: str,
    group_col=None,
) -> pd.DataFrame:
    """Create lag and rolling features using past observations only."""

    df = df.copy()

    if group_col:
        grouped = df.groupby(group_col, sort=False)[target_col]
        df["lag_1"] = grouped.shift(1)
        df["lag_2"] = grouped.shift(2)
        df["rolling_mean_3"] = grouped.transform(
            lambda values: values.shift(1).rolling(3).mean()
        )
    else:
        df["lag_1"] = df[target_col].shift(1)
        df["lag_2"] = df[target_col].shift(2)
        df["rolling_mean_3"] = (
            df[target_col].shift(1).rolling(3).mean()
        )

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
