
import pandas as pd

raw_path = "data/SM Cleaned Data MH2021.csv"
aggregated_path = "data/SM Cleaned Data MH Aggregated.csv"

# Load both datasets
raw = pd.read_csv(raw_path)
aggregated = pd.read_csv(aggregated_path)

# Extract the date from each raw timestamp
raw["Date"] = pd.to_datetime(raw["x_Timestamp"]).dt.strftime("%Y-%m-%d")
aggregated["Date"] = pd.to_datetime(aggregated["Date"]).dt.strftime("%Y-%m-%d")

# Calculate daily energy from the raw readings
daily = (
    raw.groupby(["meter", "Date"], as_index=False)["t_kWh"]
    .sum()
    .rename(columns={"t_kWh": "raw_daily_kWh"})
)

# Compare only matching meter-date combinations
comparison = daily.merge(
    aggregated.rename(columns={"t_kWh": "official_daily_kWh"}),
    on=["meter", "Date"],
    how="inner",
)

comparison["abs_diff"] = (
    comparison["raw_daily_kWh"] - comparison["official_daily_kWh"]
).abs()

print("Matching meter-days:", len(comparison))

if not comparison.empty:
    print("Mean absolute difference:", comparison["abs_diff"].mean())
    print("Maximum absolute difference:", comparison["abs_diff"].max())
    print(
        "Matches within 0.001 kWh:",
        (comparison["abs_diff"] <= 0.001).sum(),
    )
    print("\nFirst 10 comparisons:")
    print(comparison.head(10).to_string(index=False))
else:
    print("No matching meter-date records were found.")
