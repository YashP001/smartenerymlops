import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/validation.csv"

FEATURES = [
    "energy_current_hour",
    "energy_previous_hour",
    "energy_two_hours_ago",
    "target_hour",
    "target_day_of_week",
    "target_is_weekend",
]
TARGET = "target_next_hour"

train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VAL_PATH)

# Check that the required columns exist.
required = FEATURES + [TARGET]
for name, df in [("train", train), ("validation", val)]:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} dataset is missing columns: {missing}")

X_train = train[FEATURES]
y_train = train[TARGET]
X_val = val[FEATURES]
y_val = val[TARGET]

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42,
    ),
}

results = []

# Persistence baseline: next-hour energy equals current-hour energy.
baseline_predictions = val["energy_current_hour"]
results.append({
    "Model": "Persistence Baseline",
    "MAE": mean_absolute_error(y_val, baseline_predictions),
    "RMSE": np.sqrt(mean_squared_error(y_val, baseline_predictions)),
    "R2": r2_score(y_val, baseline_predictions),
})

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_val, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_val, predictions)),
        "R2": r2_score(y_val, predictions),
    })

results_df = pd.DataFrame(results).sort_values("MAE")
results_df.to_csv("reports/validation_results.csv", index=False)

print("\nValidation results (sorted by MAE):")
print(results_df.to_string(index=False))
print("\nSaved results to reports/validation_results.csv")
