# Model Performance Monitoring Guide

## Overview

The **Model Performance Monitoring Engine** (`src/monitoring/performance_monitor.py`) continuously evaluates live inference accuracy by pairing operational predictions with delayed actual smart-meter feedback.

It tracks key forecasting metrics (MAE, RMSE, R², and prediction bias), isolates evaluation by model version and time windows, and compares observed performance against empirical historical test baselines to detect degradation.

---

## Architecture & Data Flow

```
                      +-----------------------------------+
                      |   Inference API (api/main.py)     |
                      |   - prediction_id                 |
                      |   - 6 model features              |
                      |   - predicted_energy_kwh          |
                      |   - model_version                 |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------------------------+
                      |   Predictions Table (SQLite)      |
                      +-----------------+-----------------+
                                        |
                      +-----------------+-----------------+
                      |   Smart-Meter Feedback API        |
                      |   - /feedback (prediction_id,     |
                      |     actual_energy_kwh, timestamp) |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------------------------+
                      |   Feedback Table (SQLite)         |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------------------------+
                      |   v_monitoring_records (SQL View) |
                      |   - Joins on prediction_id        |
                      |   - Filters actual_energy IS NOT  |
                      |     NULL                          |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------------------------+
                      |   Performance Monitor Engine      |
                      |   (src/monitoring/                |
                      |    performance_monitor.py)        |
                      |   - MAE, RMSE, R², Bias           |
                      |   - Window & version filtering    |
                      |   - Degradation assessment        |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------------------------+
                      |   Reports (reports/performance/)  |
                      |   - performance_report_*.json     |
                      +-----------------------------------+
```

---

## How Predictions and Feedback are Paired

1. **Prediction Ingestion:**
   When a prediction request arrives at `POST /predict`, a unique UUID `prediction_id` is generated, and the input features, prediction, timestamp, latency, and `model_version` are recorded in the `predictions` table.
2. **Delayed Feedback Ingestion:**
   When the smart meter completes the subsequent hour's measurement, `POST /feedback` records the ground truth `actual_energy_kwh` against the corresponding `prediction_id`.
3. **Relational View (`v_monitoring_records`):**
   The database maintains a pre-indexed left outer join view. The performance monitor queries this view with `WHERE actual_energy_kwh IS NOT NULL`, ensuring that only paired observations with verified ground truth are evaluated.

---

## Core Performance Metrics Explained

| Metric | Formula | Plain Language Description |
|:---|:---:|:---|
| **MAE**<br>(Mean Absolute Error) | $\frac{1}{N}\sum \|y_i - \hat{y}_i\|$ | The average magnitude of prediction errors in kilowatt-hours (kWh). Lower is better. |
| **RMSE**<br>(Root Mean Squared Error) | $\sqrt{\frac{1}{N}\sum (y_i - \hat{y}_i)^2}$ | Standard deviation of residuals. Penalizes large forecast spikes more heavily than MAE. Lower is better. |
| **R²**<br>(Coefficient of Determination) | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Proportion of target variance explained by the model. $1.0$ is perfect; $0.0$ indicates performance no better than predicting the mean. Higher is better. |
| **Mean Prediction Bias** | $\frac{1}{N}\sum (\hat{y}_i - y_i)$ | Directional error (predicted minus actual). Positive indicates systematic over-forecasting; negative indicates systematic under-forecasting. Near $0.0$ is ideal. |

### Why Standard MAPE is Avoided
Standard Mean Absolute Percentage Error ($\text{MAPE} = \frac{1}{N}\sum |\frac{y - \hat{y}}{y}|$) divides by actual consumption $y$. In smart-meter telemetry, unoccupied households, zero-consumption circuits, or net-metering events frequently record $0.0$ kWh. This causes division by zero or extreme distortions. Hence, monitoring relies on scale-appropriate absolute metrics (MAE, RMSE, Bias) and R².

### Edge Case Handling in R²
- **Sample Count $< 2$:** Variance is mathematically undefined; R² is recorded as `None`.
- **Constant Target ($\sum (y_i - \bar{y})^2 = 0$):**
  - If the model perfectly predicts the constant value, $R^2 = 1.0$.
  - If the model deviates from the constant value, $R^2 = 0.0$.

---

## Model Version Separation & Evaluation Windows

### 1. Model Version Separation
Production environments may deploy newer model iterations (e.g. `GradientBoostingRegressor_v2` vs `RandomForestRegressor`). The monitor filters explicitly by `model_version`. Performance from distinct model revisions is **never** pooled together, ensuring that each model's operational health is evaluated independently.

### 2. Configurable Evaluation Windows
- `start_time` and `end_time` parameters support ISO-8601 timestamps (`2026-11-01T00:00:00Z`).
- Date filters are evaluated in SQLite via `datetime(prediction_timestamp)`, ensuring reliable comparisons across timezone formats.
- **Time Breakdown:** Automatically groups observations by day (`YYYY-MM-DD`) to show performance evolution across chronological intervals.

---

## Performance Degradation Detection

### Empirical Baseline Selection
Degradation thresholds are anchored directly to the final test set metrics achieved by `GradientBoostingRegressor` on historical data (`reports/phase1_report.md`):
- **Test Baseline MAE:** `0.0739 kWh`
- **Test Baseline RMSE:** `0.1512 kWh`
- **Test Baseline R²:** `0.3723`
- **Test Baseline Bias:** `0.0000 kWh`

### Configurable Degradation Rules

| Parameter | Default | Degradation Condition | Diagnostic Action |
|:---|:---:|:---|:---|
| `min_samples` | `30` | $N < 30$ | Flags `"insufficient_data"`. Suppresses premature false alarms on small feedback batches. |
| `max_mae_increase_pct` | `25%` | $\text{MAE} > 0.0739 \times 1.25 = 0.0924$ | Flags degradation: error magnitude has grown by $> 25\%$. |
| `max_rmse_increase_pct` | `25%` | $\text{RMSE} > 0.1512 \times 1.25 = 0.1890$ | Flags degradation: variance of large errors has grown by $> 25\%$. |
| `min_r2_threshold` | `0.10` | $R^2 < 0.10$ | Flags degradation: model is explaining almost none of the target variance. |
| `max_abs_bias` | `0.05 kWh` | $|\text{Bias}| > 0.05$ | Flags degradation: systematic over- or under-prediction bias. |

### Status Taxonomy
- **`healthy`:** Evaluated samples $\ge 30$, and all metrics remain within permissible baseline limits.
- **`degraded`:** Evaluated samples $\ge 30$, and at least one metric breaches its configured degradation threshold. Clear diagnostic reasons are provided.
- **`insufficient_data`:** Paired feedback records $< 30$. Alerts are withheld until sufficient statistical sample size is reached.
- **`no_feedback_data`:** No paired feedback records found matching the requested window or version.

---

## Report Generation & Storage

Generated performance audit manifests are saved under `reports/performance/`:
- **File Name:** `reports/performance/performance_report_<model_version>.json`
- **Git Isolation:** All `.json` report files are ignored via `.gitignore`, preserving only `reports/performance/.gitkeep`.

### Sample Report Schema

```json
{
  "evaluation_timestamp": "2026-11-01T11:45:00+00:00",
  "model_version": "GradientBoostingRegressor",
  "evaluation_window": {
    "requested_start": null,
    "requested_end": null,
    "first_record_timestamp": "2026-11-01T00:00:00Z",
    "last_record_timestamp": "2026-11-02T15:00:00Z"
  },
  "total_eligible_feedback_records": 40,
  "evaluated_predictions": 40,
  "status": "healthy",
  "is_degraded": false,
  "metrics": {
    "mae": 0.02,
    "rmse": 0.02,
    "r2": 0.9412,
    "bias": 0.0,
    "sample_count": 40
  },
  "degradation_assessment": {
    "status": "healthy",
    "is_degraded": false,
    "sample_count": 40,
    "min_samples_required": 30,
    "reasons": [
      "All metrics within acceptable baseline thresholds."
    ]
  },
  "thresholds": {
    "max_mae_increase_pct": 0.25,
    "max_rmse_increase_pct": 0.25,
    "min_r2_threshold": 0.1,
    "max_abs_bias": 0.05,
    "min_samples": 30
  },
  "time_breakdown": [
    {
      "bucket": "2026-11-01",
      "sample_count": 24,
      "mae": 0.02,
      "rmse": 0.02,
      "r2": 0.9412,
      "bias": 0.0
    }
  ]
}
```

---

## CLI & Python API Usage

### Command Line Interface

```bash
# Evaluate overall performance across all models
.venv\Scripts\python.exe src/monitoring/performance_monitor.py

# Evaluate a specific model version
.venv\Scripts\python.exe src/monitoring/performance_monitor.py \
  --model-version GradientBoostingRegressor

# Evaluate within a specific time window
.venv\Scripts\python.exe src/monitoring/performance_monitor.py \
  --start-time 2026-11-01T00:00:00Z \
  --end-time 2026-11-02T00:00:00Z
```

> [!NOTE]
> **Timezone Normalization:** Window boundaries support ISO-8601 strings with timezone offsets (e.g. `2026-11-01T12:00:00+05:30` or `2026-11-01T06:30:00Z`). Timezone-aware strings are automatically normalized to UTC before querying SQLite. Naive timestamp strings without offset information are assumed to represent UTC.

### Python API

```python
from src.monitoring.performance_monitor import evaluate_model_performance, generate_performance_report

# 1. Run in-memory evaluation
summary = evaluate_model_performance(
    model_version="GradientBoostingRegressor",
    min_samples=30,
)

if summary["status"] == "degraded":
    print("ALERT: Model degradation detected!")
    for reason in summary["degradation_assessment"]["reasons"]:
        print(f"  - {reason}")

# 2. Persist JSON report
report_path, report_data = generate_performance_report(
    model_version="GradientBoostingRegressor",
    output_dir="reports/performance",
)
print(f"Performance report saved to: {report_path}")
```

---

## Limitations Caused by Delayed Feedback

1. **Feedback Latency:** Smart-meter readings are only available after the physical hour concludes and telemetry data is transmitted and parsed. Consequently, performance monitoring operates with an unavoidable delay (typically 1–2 hours).
2. **Incomplete Feedback:** If communication drops occur between meters and the backend, feedback may arrive out of order or be delayed. The engine handles missing feedback gracefully by ignoring un-paired predictions, but monitoring sample size may fluctuate during network outages.
3. **Decoupled from Drift:** Performance degradation measures ground-truth loss ($y - \hat{y}$), whereas data drift measures feature distribution shifts ($P(X)$). A model may suffer feature drift while maintaining acceptable accuracy (e.g., if features shift within robust sub-regions), or suffer performance degradation without obvious input drift (e.g., unobserved covariate shift). Both monitors should be used collaboratively.
