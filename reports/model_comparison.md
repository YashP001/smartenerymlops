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

- **MLflow experiment:** `smart-home-energy-advisor` (stored in persistent SQLite DB `mlflow.db`).
- All model parameters, metrics, signatures, and artifacts logged per run.
- Selected model registered in MLflow Model Registry (`smart-home-energy-advisor` Version 1) with idempotency protection.
- W&B dashboards available for interactive metric comparison and model evaluation tables.

### Weights & Biases (W&B) Cloud Synchronization

When training runs without a configured `WANDB_API_KEY`, W&B executes in **offline mode** (`WANDB_MODE=offline`) and saves all run data, metrics, and evaluation tables locally under `wandb/offline-run-<timestamp>-<id>`.

To synchronize local offline runs to your personal online W&B project:

1. **Obtain your W&B API Key:**
   - Log in to [Weights & Biases](https://wandb.ai) and retrieve your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize).

2. **Authenticate locally (never commit your API key):**
   ```bash
   wandb login
   ```
   *Alternatively, set the environment variable for your local shell session only:*
   - **PowerShell:** `$env:WANDB_API_KEY="your_api_key_here"`
   - **Bash:** `export WANDB_API_KEY="your_api_key_here"`

3. **Synchronize the offline experiment run:**
   ```bash
   # Sync a specific run:
   wandb sync wandb/offline-run-20261031_230321-rugud7li

   # Or sync all offline runs:
   wandb sync --include-offline wandb/
   ```

4. **Inspect your online dashboard:**
   - Navigate to `https://wandb.ai/<your-username>/smart-home-energy-advisor` to inspect the comparative validation metrics (MAE, RMSE, $R^2$) and the `model_comparison` summary table.

## Limitations

- R² of ~0.37 indicates moderate predictive power; room for improvement.
- Model uses only historical energy and calendar features (no weather, no voltage/current).
- Performance may vary across individual meters.

## Registered Model

- **Registry name:** `smart-home-energy-advisor`
- **Model type:** `GradientBoostingRegressor`
- **Saved artifact:** `models/gradient_boosting.pkl`
