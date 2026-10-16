import pandas as pd

INPUT_PATH = "data/model_dataset.csv"

df = pd.read_csv(INPUT_PATH, parse_dates=["hour"])
df = df.sort_values(["hour", "meter"]).reset_index(drop=True)

# Each row predicts energy in the next hour.
df["target_time"] = df["hour"] + pd.Timedelta(hours=1)

# Split using chronological target-time boundaries.
target_times = sorted(df["target_time"].unique())
n_times = len(target_times)

train_end = int(n_times * 0.70)
validation_end = int(n_times * 0.85)

train_boundary = target_times[train_end]
test_boundary = target_times[validation_end]

train = df[df["target_time"] < train_boundary].copy()

validation = df[
    (df["target_time"] >= train_boundary)
    & (df["target_time"] < test_boundary)
].copy()

test = df[df["target_time"] >= test_boundary].copy()

for name, part in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:
    part.drop(columns=["target_time"]).to_csv(
        f"data/{name}.csv", index=False
    )

print("Training target boundary:", train_boundary)
print("Test target boundary:", test_boundary)

for name, part in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test),
]:
    print(f"\n{name}: {len(part)} rows")
    print(
        "Target range:",
        part["target_time"].min(),
        "to",
        part["target_time"].max(),
    )
