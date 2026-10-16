import pandas as pd

INPUT_PATH = "data/hourly_energy.csv"
OUTPUT_PATH = "data/model_dataset.csv"

df = pd.read_csv(INPUT_PATH, parse_dates=["hour"])
df = df.sort_values(["meter", "hour"]).reset_index(drop=True)

# Only use consecutive hours to create prediction examples.
previous_hour = df.groupby("meter")["hour"].shift(1)
consecutive = (
    (df["hour"] - previous_hour) == pd.Timedelta(hours=1)
)

# Current-hour features and next-hour prediction target.
grouped = df.groupby("meter", group_keys=False)

df["energy_current_hour"] = grouped["energy_kWh"].shift(0)
df["energy_previous_hour"] = grouped["energy_kWh"].shift(1)
df["energy_two_hours_ago"] = grouped["energy_kWh"].shift(2)

df["target_next_hour"] = grouped["energy_kWh"].shift(-1)
next_hour = grouped["hour"].shift(-1)

# The target must be exactly one hour after the current row.
valid_target = (
    next_hour - df["hour"] == pd.Timedelta(hours=1)
)

# Calendar features describe the hour being predicted.
target_time = df["hour"] + pd.Timedelta(hours=1)
df["target_hour"] = target_time.dt.hour
df["target_day_of_week"] = target_time.dt.dayofweek
df["target_is_weekend"] = (
    target_time.dt.dayofweek >= 5
).astype(int)

# Keep rows with sufficient history and a valid next-hour target.
df = df[
    consecutive
    & valid_target
    & df["energy_previous_hour"].notna()
    & df["energy_two_hours_ago"].notna()
].copy()

df.to_csv(OUTPUT_PATH, index=False)

print("Prepared prediction examples:", len(df))
print("Meters:", df["meter"].nunique())
print("Output:", OUTPUT_PATH)
print("\nPreview:")
print(
    df[
        [
            "meter",
            "hour",
            "energy_current_hour",
            "energy_previous_hour",
            "energy_two_hours_ago",
            "target_next_hour",
        ]
    ].head().to_string(index=False)
)
