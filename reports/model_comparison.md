# Model Comparison Report

## Objective

Compare candidate machine learning models for next-hour energy consumption prediction using the CEEW Mathura 2021 smart meter dataset.

## Models Evaluated

| Model | Description |
|---|---|
| **Persistence Baseline** | Predicts next-hour energy = current-hour energy. No learning. |
| **Linear Regression** | Standard ordinary least squares regression. |
| **Random Forest** | Ensemble of 100 decision trees (`n_estimators=100`, `random_state=42`). |
| **Gradient Boosting** | Sequential boosting with default learning rate (`random_state=42`). |

## Evaluation Methodology

- **Split strategy:** Chronological 70/15/15 (Train/Validation/Test).
- **Training set:** 18,696 rows
- **Validation set:** 4,085 rows
- **Test set:** 4,120 rows
- **Metrics:** MAE (kWh), RMSE (kWh), R²
- **No random shuffling** — time-series ordering preserved to prevent data leakage.

## Input Features

| Feature | Description |
|---|---|
| `energy_current_hour` | Energy consumption in hour *t* |
| `energy_previous_hour` | Energy consumption in hour *t-1* |
| `energy_two_hours_ago` | Energy consumption in hour *t-2* |
| `target_hour` | Hour of day for the prediction target (0–23) |
| `target_day_of_week` | Day of week for target (0=Mon, 6=Sun) |
| `target_is_weekend` | Binary flag (1 if weekend) |

**Target:** `target_next_hour` — energy consumption in hour *t+1*.

## Validation Set Results

| Model | MAE (kWh) | RMSE (kWh) | R² |
|---|---:|---:|---:|
| **Gradient Boosting** | **0.0712** | **0.1423** | **0.3397** |
| Random Forest | 0.0734 | 0.1472 | 0.2942 |
| Persistence Baseline | 0.0778 | 0.1827 | -0.0885 |
| Linear Regression | 0.0821 | 0.1554 | 0.2125 |

## Final Test Set Results (Gradient Boosting)

After refitting on training + validation data:

- **Test MAE:** 0.0739 kWh
- **Test RMSE:** 0.1512 kWh
- **Test R²:** 0.3723

## Model Selection Justification

**Gradient Boosting Regressor** was selected because:

1. **Lowest validation MAE** (0.0712 kWh) across all candidates.
2. **Highest R²** (0.3397) indicating best variance explanation.
3. **Consistent performance** from validation to test set with minimal degradation.
4. **Meaningful improvement** over the persistence baseline (8.5% MAE reduction).

## Experiment Tracking

- **MLflow experiment:** `smart-home-energy-advisor`
- All model parameters, metrics, and artifacts logged per run.
- Selected model registered in MLflow Model Registry.
- W&B dashboards available for interactive metric comparison.

## Limitations

- R² of ~0.37 indicates moderate predictive power; room for improvement.
- Model uses only historical energy and calendar features (no weather, no voltage/current).
- Performance may vary across individual meters.

## Registered Model

- **Registry name:** `smart-home-energy-advisor`
- **Model type:** `GradientBoostingRegressor`
- **Saved artifact:** `models/gradient_boosting.pkl`
