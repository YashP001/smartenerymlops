
import pandas as pd


def validate_data(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Validate the minimum columns needed for time-series forecasting."""

    required = [timestamp_col, target_col]
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

    df = df.dropna(subset=[timestamp_col, target_col])
    df = df.sort_values(timestamp_col)
    df = df.drop_duplicates(subset=[timestamp_col])

    return df.reset_index(drop=True)


def add_time_features(
    df: pd.DataFrame,
    timestamp_col: str,
) -> pd.DataFrame:
    """Create general calendar features from a timestamp."""

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
) -> pd.DataFrame:
    """Create features using only previous observations."""

    df = df.copy()

    df["lag_1"] = df[target_col].shift(1)
    df["lag_2"] = df[target_col].shift(2)

    # Shift before rolling so the current value is excluded.
    df["rolling_mean_3"] = (
        df[target_col].shift(1).rolling(3).mean()
    )

    return df
