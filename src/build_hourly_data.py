import os
import pandas as pd

RAW_PATH = (
    "data/raw/SM Cleaned Data MH2021.csv"
    if os.path.exists("data/raw/SM Cleaned Data MH2021.csv")
    else "data/SM Cleaned Data MH2021.csv"
)
OUTPUT_PATH = "data/hourly_energy.csv"

df = pd.read_csv(
    RAW_PATH,
    usecols=["x_Timestamp", "meter", "t_kWh"],
)

df["x_Timestamp"] = pd.to_datetime(
    df["x_Timestamp"], errors="coerce"
)
df["t_kWh"] = pd.to_numeric(df["t_kWh"], errors="coerce")

df = df.dropna(subset=["x_Timestamp", "meter", "t_kWh"])
df = df.sort_values(["meter", "x_Timestamp"])

# Assign each reading to its clock hour.
df["hour"] = df["x_Timestamp"].dt.floor("h")

# A valid hour must contain exactly 20 readings at the
# expected three-minute timestamps.
df["minute"] = df["x_Timestamp"].dt.minute
df["second"] = df["x_Timestamp"].dt.second

hourly = (
    df.groupby(["meter", "hour"], as_index=False)
    .agg(
        readings=("t_kWh", "size"),
        unique_timestamps=("x_Timestamp", "nunique"),
        energy_kWh=("t_kWh", "sum"),
        minute_values=("minute", list),
        second_values=("second", list),
    )
)

def valid_hour(row):
    expected_minutes = list(range(0, 60, 3))
    return (
        row["readings"] == 20
        and row["unique_timestamps"] == 20
        and sorted(row["minute_values"]) == expected_minutes
        and all(s == 0 for s in row["second_values"])
    )

hourly["is_complete"] = hourly.apply(valid_hour, axis=1)

complete = hourly[hourly["is_complete"]].copy()
incomplete = hourly[~hourly["is_complete"]].copy()

complete = complete[["meter", "hour", "readings", "energy_kWh"]]
complete = complete.sort_values(["meter", "hour"])

print("Observed meter-hours:", len(hourly))
print("Verified complete hours:", len(complete))
print("Rejected hours:", len(incomplete))

complete.to_csv(OUTPUT_PATH, index=False)
print("\nSaved:", OUTPUT_PATH)
print("\nFirst 5 verified hours:")
print(complete.head().to_string(index=False))
