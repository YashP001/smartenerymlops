# Prediction & Feedback Logging Guide (Phase 3.1)

This document describes the operational inference logging and ground-truth feedback mechanism implemented in Phase 3.1 for the **Smart Home Energy Advisor** MLOps project.

---

## 1. Overview & Architecture

In production smart-meter operations, machine learning predictions occur in real time ($t$), while the actual energy consumption is recorded by the physical meter at the end of the hourly window ($t+1$). 

To track model performance and detect drift without impacting real-time inference latency:
1. **Asynchronous Prediction Logging:** When `POST /predict` executes, prediction features, output, and latency are returned immediately to the client, while FastAPI `BackgroundTasks` persists the audit record to a dedicated SQLite monitoring database.
2. **Prediction Identifier:** A unique UUID (`prediction_id`) is returned in both the JSON response body and the `X-Prediction-ID` response header.
3. **Delayed Feedback Ingestion:** When actual meter readings arrive, the `POST /feedback` endpoint pairs the ground truth with the logged prediction.

```
+----------------+      POST /predict       +-------------------+
|  Smart Meter / | -----------------------> | FastAPI Service   |
|  Client App    | <----------------------- | (api/main.py)     |
+----------------+   returns prediction_id  +-------------------+
                                                      |
                                             (Background Task)
                                                      v
                                            +-------------------+
                                            | SQLite Monitoring |
                                            |   (predictions)   |
                                            +-------------------+
                                                      ^
+----------------+      POST /feedback                |
| Meter Feedback | -----------------------------------+
| Ingestion Job  |      (actual_energy_kwh)
+----------------+
```

---

## 2. SQLite Database Schema

The database is stored locally at `data/monitoring/predictions.db` (gitignored to prevent repository bloat).

### Table: `predictions`
Stores raw feature inputs and model inference outputs:

| Column | Type | Description |
| :--- | :--- | :--- |
| `prediction_id` | `TEXT PRIMARY KEY` | UUID assigned at inference time |
| `timestamp` | `TEXT NOT NULL` | ISO 8601 UTC timestamp of the prediction |
| `energy_current_hour` | `REAL NOT NULL` | Current hour consumption (kWh) |
| `energy_previous_hour` | `REAL NOT NULL` | 1-hour lag consumption (kWh) |
| `energy_two_hours_ago` | `REAL NOT NULL` | 2-hour lag consumption (kWh) |
| `target_hour` | `INTEGER NOT NULL` | Target hour of day (0–23) |
| `target_day_of_week` | `INTEGER NOT NULL` | Day of week (0=Mon, 6=Sun) |
| `target_is_weekend` | `INTEGER NOT NULL` | Weekend binary flag (0 or 1) |
| `predicted_energy_kwh` | `REAL NOT NULL` | Model prediction (kWh) |
| `model_version` | `TEXT NOT NULL` | Model name/version (e.g. `GradientBoostingRegressor`) |
| `latency_ms` | `REAL` | Execution latency in milliseconds |

### Table: `feedback`
Stores verified actual meter readings:

| Column | Type | Description |
| :--- | :--- | :--- |
| `prediction_id` | `TEXT PRIMARY KEY` | Foreign key referencing `predictions(prediction_id)` |
| `actual_energy_kwh` | `REAL NOT NULL` | Observed energy consumption ($\ge 0$) |
| `feedback_timestamp` | `TEXT NOT NULL` | ISO 8601 UTC timestamp of feedback receipt |

### View: `v_monitoring_records`
Joined view providing instant actual-versus-predicted error analysis:
- `error = actual_energy_kwh - predicted_energy_kwh`
- `absolute_error = ABS(actual_energy_kwh - predicted_energy_kwh)`

---

## 3. API Endpoints

### 1. `POST /predict`
Predicts next-hour consumption and issues a tracking ID.

**Request:**
```json
{
  "energy_current_hour": 0.05,
  "energy_previous_hour": 0.04,
  "energy_two_hours_ago": 0.06,
  "target_hour": 14,
  "target_day_of_week": 2,
  "target_is_weekend": 0
}
```

**Response (`200 OK`):**
```json
{
  "predicted_energy_kwh": 0.06106,
  "model": "GradientBoostingRegressor",
  "features_used": [
    "energy_current_hour",
    "energy_previous_hour",
    "energy_two_hours_ago",
    "target_hour",
    "target_day_of_week",
    "target_is_weekend"
  ],
  "prediction_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"
}
```
*Response Header:* `X-Prediction-ID: f47ac10b-58cc-4372-a567-0e02b2c3d479`

### 2. `POST /feedback`
Records actual ground truth associated with an earlier prediction.

**Request:**
```json
{
  "prediction_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "actual_energy_kwh": 0.064
}
```

**Response (`200 OK`):**
```json
{
  "status": "success",
  "prediction_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "actual_energy_kwh": 0.064,
  "feedback_timestamp": "2026-11-01T04:30:00.000000+00:00"
}
```

**Error Codes:**
- `404 Not Found`: If `prediction_id` does not exist.
- `409 Conflict`: If feedback for this `prediction_id` has already been recorded.
- `422 Unprocessable Entity`: If `actual_energy_kwh` is negative.

*(Alternative URL: `POST /predictions/{prediction_id}/feedback` with body `{"actual_energy_kwh": 0.064}` is also supported).*

---

## 4. How to Test & Verify

### Running the Test Suite
Run the monitoring tests:
```powershell
.venv\Scripts\python.exe -m pytest tests/test_monitoring.py -v
```

Run the entire project test suite:
```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

### Manual Verification via PowerShell
1. **Start the API locally:**
   ```powershell
   .venv\Scripts\python.exe -m uvicorn api.main:app --port 8000
   ```

2. **Send a prediction request:**
   ```powershell
   $body = @{
       energy_current_hour = 0.05
       energy_previous_hour = 0.04
       energy_two_hours_ago = 0.06
       target_hour = 14
       target_day_of_week = 2
       target_is_weekend = 0
   } | ConvertTo-Json

   $res = Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body
   $id = $res.prediction_id
   Write-Host "Received Prediction ID:" $id
   ```

3. **Send matching feedback:**
   ```powershell
   $feedback = @{
       prediction_id = $id
       actual_energy_kwh = 0.062
   } | ConvertTo-Json

   Invoke-RestMethod -Uri "http://127.0.0.1:8000/feedback" -Method Post -ContentType "application/json" -Body $feedback
   ```
