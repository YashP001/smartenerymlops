import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "energy_current_hour",
    "energy_previous_hour",
    "energy_two_hours_ago",
    "target_hour",
    "target_day_of_week",
    "target_is_weekend",
]
TARGET = "target_next_hour"

train = pd.read_csv("data/train.csv")
validation = pd.read_csv("data/validation.csv")
test = pd.read_csv("data/test.csv")

# After selecting the model using validation results,
# refit on training + validation data.
development = pd.concat([train, validation], ignore_index=True)

model = GradientBoostingRegressor(random_state=42)
model.fit(development[FEATURES], development[TARGET])

predictions = model.predict(test[FEATURES])
actual = test[TARGET]

print("Final test results")
print("Test rows:", len(test))
print("MAE:", mean_absolute_error(actual, predictions))
print("RMSE:", np.sqrt(mean_squared_error(actual, predictions)))
print("R2:", r2_score(actual, predictions))

# Save test predictions for later analysis.
output = test[["meter", "hour", TARGET]].copy()
output["predicted_next_hour"] = predictions
output.to_csv("reports/test_predictions.csv", index=False)

print("\nSaved predictions to reports/test_predictions.csv")
