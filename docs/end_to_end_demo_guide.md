# End-to-End MLOps Demonstration Guide

**Phase 3.7 Deliverable**
**AI-Assisted Smart Home Energy Advisor MLOps Pipeline**

---

## 1. Executive Summary & Objective

This guide provides a reproducible, end-to-end walkthrough of the **Smart Home Energy Advisor** MLOps system. The platform predicts next-hour household electricity consumption from smart-meter telemetry and manages the complete post-deployment operational lifecycle:

1. **Inference Serving & Tracking:** FastAPI REST service with asynchronous prediction logging to SQLite.
2. **Ground-Truth Ingestion:** Delayed actual consumption feedback ingestion and automated record pairing.
3. **Production Simulation:** Deterministic simulation of normal household consumption, afternoon HVAC cooling spikes (covariate shift), and sensor decay (calibration bias).
4. **Drift Detection:** SciPy Kolmogorov-Smirnov statistical tests, Cohen's $d$ effect sizes, and interactive Evidently AI reports.
5. **Performance Monitoring:** Automated calculation of MAE, RMSE, $R^2$, and mean bias against empirical test baselines (with a $\ge 30$ sample statistical validity threshold).
6. **Alert Management:** State machine handling alert triggering, duplicate suppression (cooldown), and automatic resolution upon distribution recovery.
7. **Observability Exposition:** Prometheus scraping endpoint (`/metrics`) and pre-configured Grafana dashboards visualizing inference performance, drift share, and alert status.

---

## 2. System Architecture

The following diagram illustrates the component architecture and data flow across the operational lifecycle:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                SYSTEM ARCHITECTURE OVERVIEW                             │
│                                                                                         │
│   ┌─────────────────────┐                                                               │
│   │ Smart Meter Clients │                                                               │
│   └──────────┬──────────┘                                                               │
│              │ 1. POST /predict                                                         │
│              ▼                                                                          │
│   ┌─────────────────────────────────────────┐         ┌─────────────────────────────┐   │
│   │           FastAPI Service               │         │   MLflow Model Registry     │   │
│   │             (api/main.py)               │◄────────┤ (models:/smart-home-.../1)  │   │
│   │                                         │         └─────────────────────────────┘   │
│   │ • Asynchronous Background Logging       │                                           │
│   │ • POST /feedback                        │                                           │
│   │ • GET /health                           │                                           │
│   │ • GET /metrics (Prometheus Exposition)  │                                           │
│   └──────────┬──────────────────────────────┘                                           │
│              │                                                                          │
│              │ Read / Write                                                             │
│              ▼                                                                          │
│   ┌─────────────────────────────────────────┐                                           │
│   │      SQLite Monitoring Database         │                                           │
│   │       (data/monitoring/predictions.db)  │                                           │
│   │                                         │                                           │
│   │ • predictions                           │                                           │
│   │ • feedback                              │                                           │
│   │ • v_monitoring_records (paired view)    │                                           │
│   │ • alerts                                │                                           │
│   │ • monitoring_runs                       │                                           │
│   │ • monitoring_execution_failures        │                                           │
│   └──────────▲──────────────────────────────┘                                           │
│              │                                                                          │
│              │ Orchestrates & Logs                                                      │
│              │                                                                          │
│   ┌──────────┴──────────────────────────────┐         ┌─────────────────────────────┐   │
│   │       Monitoring Orchestrator           │         │     Evidently AI Engine     │   │
│   │ (src/monitoring/monitor_orchestrator.py)├────────►┤   & SciPy Statistical Tests │   │
│   │                                         │         └──────────────┬──────────────┘   │
│   │ • Drift Detector (drift_detector.py)    │                        │                  │
│   │ • Performance Monitor (perf_monitor.py) │                        ▼                  │
│   │ • Alert Manager (alert_manager.py)      │         ┌─────────────────────────────┐   │
│   └─────────────────────────────────────────┘         │  Interactive HTML & Reports │   │
│                                                       │       (reports/drift/)      │   │
│                                                       └─────────────────────────────┘   │
│                                                                                         │
│   ┌─────────────────────────────────────────┐         ┌─────────────────────────────┐   │
│   │               Prometheus                │         │           Grafana           │   │
│   │        (Scrapes api:8000/metrics)       ├────────►│  (Interactive Dashboard)   │   │
│   └─────────────────────────────────────────┘         └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Real Historical Data vs. Simulated Production Data vs. Demonstration Feedback

To maintain absolute scientific integrity, the project strictly distinguishes between genuine customer measurements, simulated production scenarios, and post-deployment feedback:

| Data Type | Source / Path | Description | Usage |
| :--- | :--- | :--- | :--- |
| **Real Historical Baseline** | `data/train.csv`, `data/test.csv` | Empirical hourly smart-meter readings collected from 50 households in Mathura, Uttar Pradesh (CEEW 2021 study). | Model training, validation, baseline performance metrics, and reference distributions for drift detection. |
| **Simulated Production Data** | `data/simulated/` | Synthetic smart-meter observations generated by `src/simulation.py` with explicit metadata flags. | Evaluating data drift scenarios (`normal`, `hvac_shift`, `sensor_drift`) without altering historical datasets. |
| **Authentic Held-Out Feedback (Default)** | `data/test.csv` (held-out rows) | Genuine historical smart-meter actuals (`target_next_hour`) matching the identical input features sent to `/predict`. | Evaluating genuine operational generalization against the Phase 1 test baseline. |
| **Synthetic Demonstration Feedback (Opt-in)** | Derived in-memory ($\pm 0.015$ kWh noise) | Model predictions perturbed with Gaussian noise (`actual = pred + Normal(0, 0.015)`). | Validating software pipeline mechanics (record pairing, alert evaluation) without claiming authentic predictive accuracy. |

---

## 4. Prerequisites & Environment Setup

### 4.1 Prerequisites
- **Operating System:** Windows 10/11, Linux, or macOS
- **Python:** Version 3.10 to 3.13
- **Virtual Environment:** Recommended (`.venv`)
- **Docker & Docker Compose (Optional):** Required only if running containerized Prometheus and Grafana instances. Local demonstration scripts run 100% out-of-the-box using standard Python.

### 4.2 Installation

```bash
# Clone the repository
git clone https://github.com/YashP001/smartenerymlops.git
cd smartenerymlops

# Activate Python virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1
# Or on Linux/macOS: source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 5. Running the End-to-End Demonstration

The demonstration workflow connects all existing components and can be executed either **automatically in a single command** or **interactively step-by-step**.

### 5.1 Automated One-Command Execution (Recommended)

Run the self-contained demonstration script in either feedback mode:

```bash
# Mode A: Authentic held-out test evaluation (Default)
.venv\Scripts\python.exe scripts/run_end_to_end_demo.py --samples 35

# Mode B: Synthetic demonstration feedback (Explicit opt-in for pipeline validation)
.venv\Scripts\python.exe scripts/run_end_to_end_demo.py --samples 35 --feedback-mode synthetic
```

#### Available CLI Arguments:
- `--samples <int>`: Number of inference and feedback samples to generate (default: `35`, minimum: `30`).
- `--feedback-mode {authentic,synthetic}`: Feedback generation mode (default: `authentic`).
  - `authentic`: Reads `data/test.csv`, samples rows using `--seed`, uses features for prediction, and supplies genuine `target_next_hour` actuals.
  - `synthetic`: Derives feedback values from model predictions with Gaussian noise (`actual = pred + Normal(0, 0.015)`).
- `--api-url <url>`: Connect to an external running FastAPI service (e.g. `http://localhost:8000`). If omitted, executes in-process via FastAPI TestClient.
- `--db-path <path>`: Path to demonstration SQLite database (defaults to isolated `reports/demo/demo_predictions.db`).
- `--demo-dir <path>`: Directory for saving generated reports and manifests (defaults to `reports/demo`).
- `--seed <int>`: Deterministic random seed (default: `42`).
- `--skip-drift-scenarios`: Flag to skip additional HVAC shift and sensor drift scenarios to shorten run time.

---

## 6. Step-by-Step Demonstration Workflow

For examiners or operators wanting to understand the internal mechanisms, the demonstration executes the following 10 stages:

### Step 1: Model Verification & Registry Availability
- The script calls `load_model()` from `src.predict`.
- It attempts to resolve `models:/smart-home-energy-advisor/1` via the MLflow Model Registry, falling back gracefully to the validated local model artifact (`models/best_model.joblib` or `models/gradient_boosting.pkl`).
- **Model Lineage & Checkpoint Architecture:**
  - **Phase 1 Historical Baseline Checkpoint:**
    - Model architecture: `GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)`.
    - Trained on combined `train + validation` splits ($N=22,781$ rows).
    - Evaluated on all $N=4,120$ held-out chronological test rows (`data/test.csv`).
    - Empirical baseline metrics: $\text{MAE} = 0.0739\text{ kWh}$, $\text{RMSE} = 0.1512\text{ kWh}$, $R^2 = 0.3723$, and $\text{Mean Bias} = +0.001534\text{ kWh}$.
  - **MLflow Registry Version 1 Checkpoint (`models:/smart-home-energy-advisor/1`):**
    - Identical model architecture and hyperparameter configuration.
    - Trained on `train.csv` only ($N=18,696$ rows) during experiment tracking.
    - Full test set performance ($N=4,120$ rows): $\text{MAE} = 0.0738\text{ kWh}$, $\text{RMSE} = 0.1510\text{ kWh}$, $R^2 = 0.3736$, and $\text{Mean Bias} = +0.002273\text{ kWh}$.
  - **Important Lineage Distinction:** These are two distinct training checkpoints of the same architecture, not identical serialized artifacts. Both exhibit virtually identical generalization. The Phase 3.7 demonstration samples 35 rows from the identical held-out `data/test.csv` population using seed 42 to evaluate operational performance.
- Confirms the model class (`GradientBoostingRegressor`) and validates the canonical 6-feature schema:
  - `energy_current_hour` (kWh)
  - `energy_previous_hour` (kWh)
  - `energy_two_hours_ago` (kWh)
  - `target_hour` (0–23)
  - `target_day_of_week` (0–6)
  - `target_is_weekend` (0 or 1)

### Step 2: FastAPI Service Health Check
- Sends a `GET /health` request.
- Asserts HTTP 200 and validates response JSON:
  ```json
  {
    "status": "healthy",
    "model_loaded": true
  }
  ```

### Step 3: Prediction Generation & Asynchronous Logging
- Sends $N \ge 35$ sequential inference requests to `POST /predict`.
- Validates the response body (`predicted_energy_kwh`, `prediction_id`, `model`) and response header `X-Prediction-ID`.
- Queries the `predictions` table in SQLite to confirm asynchronous persistence via FastAPI `BackgroundTasks`.

### Step 4: Ground-Truth Feedback Ingestion & Pairing

Depending on the chosen `--feedback-mode`, feedback observations are ingested into SQLite via `POST /feedback`:

- **Mode A: Authentic Held-Out Evaluation (Default):**
  - Reads unseen observations from `data/test.csv`.
  - Uses the identical row's `target_next_hour` as actual ground truth.
  - Sourced completely independently of the model predictions.
- **Mode B: Synthetic Demonstration Feedback (`--feedback-mode synthetic`):**
  - Perturbs model predictions with Gaussian noise (`actual = pred + Normal(0, 0.015)`).
  - Explicitly labeled as synthetic pipeline-validation data.

Example payload:
```json
{
  "prediction_id": "622799dc-...",
  "actual_energy_kwh": 0.1582
}
```
Verifies record insertion in `feedback` table and confirms pairing in `v_monitoring_records` view.

### Step 5: Production Data Simulation (Normal Scenario)
- Generates 200 hours of synthetic smart-meter telemetry using `generate_simulated_data(scenario="normal")`.
- Validates schema, finite values, and non-negativity using `validate_simulated_dataset()`.
- Persists dataset to `data/simulated/demo/simulated_normal_demo.csv` alongside accompanying metadata.

### Step 6: Statistical & Evidently AI Drift Detection
- Runs `run_drift_analysis()` comparing `simulated_normal_demo.csv` against baseline `data/train.csv`.
- Computes two-sample Kolmogorov-Smirnov statistic and Cohen's $d$ effect sizes for all 6 features.
- Generates interactive Evidently HTML report (`reports/demo/drift/drift_report_normal_demo.html`).
- Confirms `drift_share < 0.33` and `dataset_drift_detected == False`.

### Step 7: Model Performance Evaluation & Baseline Comparison

Invokes `evaluate_model_performance(db_path=...)` against paired feedback records in SQLite:

- **Authentic Mode Interpretation (Default):**
  - Metrics (MAE, RMSE, $R^2$, Bias) reflect genuine operational performance on unseen historical test data (`data/test.csv`).
  - Small differences from Phase 1 test baselines (e.g. MAE $\sim 0.0706$ vs. baseline $0.0739$ kWh, Bias $\sim +0.0016$ vs. baseline $+0.0015$ kWh) arise naturally from sampling variation on the evaluated subset ($N=35$ vs. full test set $N=4,120$).
  - Evaluated against Phase 1 empirical baseline thresholds:
    - **MAE:** $\le 0.0739 \times 1.25 = 0.0924$ kWh
    - **RMSE:** $\le 0.1512 \times 1.25 = 0.1890$ kWh
    - **$R^2$:** $\ge 0.10$
    - **Bias:** $|{\text{Bias}}| \le 0.05$ kWh (empirical baseline: $+0.0015$ kWh)
  - Confirms model status is **HEALTHY**.

- **Synthetic Mode Interpretation (`--feedback-mode synthetic`):**
  - Labeled prominently as **Synthetic Pipeline Validation Metrics**.
  - Low MAE ($\sim 0.010$ kWh) is an expected mathematical artifact of the $\pm 0.015$ kWh perturbation, **not** authentic predictive accuracy.
  - Never described as superior performance over the Phase 1 test baseline.

### Step 8: Automated Monitoring Orchestration
- Executes `run_monitoring_cycle()` combining drift detection, performance monitoring, and alert evaluation.
- Persists run metadata to `monitoring_runs` table in SQLite.
- Verifies 0 alerts are triggered under normal conditions.

### Step 9: Prometheus Observability Scraping
- Queries `GET /metrics` and verifies Prometheus exposition format:
  ```promql
  energy_advisor_prediction_requests_total{model_version="GradientBoostingRegressor",status="success"} 35.0
  energy_advisor_drift_detection_runs_total 1.0
  energy_advisor_dataset_drift_share 0.1667
  energy_advisor_performance_evaluation_runs_total 1.0
  energy_advisor_model_mae{model_version="GradientBoostingRegressor"} 0.0102
  energy_advisor_model_rmse{model_version="GradientBoostingRegressor"} 0.0124
  ```

### Step 10: Distributional Shifts & Alert Recovery
- **Scenario A (HVAC Afternoon Shift):** Simulates a heatwave cooling load spike. Drift share reaches $66.7\%$ ($\ge 50\%$), triggering a `[CRITICAL] data_drift` alert in the database.
- **Scenario B (Sensor Calibration Drift):** Simulates sensor degradation. Drift alert is maintained with updated metrics.
- **Scenario C (Automatic Alert Recovery):** Re-evaluates against normal data. The active drift alert is automatically resolved, demonstrating closed-loop alert recovery.

---

## 7. Connecting to Live Services & Observability UI

### 7.1 Running the Live FastAPI Server & Database Alignment

Because an external Uvicorn server executes in an independent process, it does **not** inherit process-level environment variables set by the demonstration script. To ensure predictions and feedback logged by Uvicorn are written to the database inspected by the demonstration script:

**Step 1: In Terminal 1 (PowerShell), set the matching database path and start Uvicorn:**
```powershell
# Set monitoring database path in the server's process space
$env:MONITORING_DB_PATH = "reports/demo/demo_predictions.db"

# Start the FastAPI service
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

**Step 2: In Terminal 2, run the demonstration script targeting the live server:**
```powershell
.venv\Scripts\python.exe scripts/run_end_to_end_demo.py --api-url http://127.0.0.1:8000 --samples 35
```

> **Automatic Database Alignment Verification:**
> The demonstration script verifies that all submitted `prediction_id`s and feedback records actually exist in its target database. If the external server is writing to a different database file, the script halts immediately with an explicit, actionable `RuntimeError` documenting the mismatch and presenting the exact PowerShell commands needed to align them. Alternatively, you can point the client to Uvicorn's active database using `--db-path data/monitoring/predictions.db`.

### 7.2 Accessing Prometheus & Grafana (Docker)

If Docker is available on the deployment host:

```bash
# Start all services (API, Prometheus, Grafana)
docker-compose up -d

# Check service health
docker-compose ps
```

- **FastAPI API:** [http://localhost:8000](http://localhost:8000) (Swagger Docs at `/docs`)
- **Prometheus UI:** [http://localhost:9090](http://localhost:9090)
  - Navigate to **Status -> Targets** to verify `api:8000` is `UP`.
  - Query metrics like `energy_advisor_dataset_drift_share`.
- **Grafana Dashboards:** [http://localhost:3000](http://localhost:3000)
  - Pre-loaded with the **Smart Home Energy Advisor - MLOps Operations** dashboard.
  - Displays real-time API latency, inference volume, drift share gauges, MAE/RMSE tracking, and active alert counters.

---

## 8. Expected Outputs & Verification Checklist

When the demonstration script finishes, it prints a summary checklist:

```text
==============================================================================
  END-TO-END MLOPS DEMONSTRATION COMPLETE
==============================================================================
  [PASS] All operational lifecycle stages executed and verified successfully:
  1. [PASS] Model Verification
  2. [PASS] Service Health Check
  3. [PASS] Prediction Logging
  4. [PASS] Feedback Ingestion
  5. [PASS] Normal Simulation
  6. [PASS] Drift Detection Normal
  7. [PASS] Performance Evaluation
  8. [PASS] Orchestration Cycle Normal
  9. [PASS] Prometheus Observability
  10. [PASS] Drift Scenarios And Recovery

Summary Statistics:
  [INFO] Database: reports\demo\demo_predictions.db
  [INFO] Predictions Logged: 35
  [INFO] Feedback Records Processed: 35
  [INFO] Reports Generated under: reports\demo
  [INFO] Simulated Data stored in: data\simulated\demo
------------------------------------------------------------------------------
```

---

## 9. Troubleshooting & FAQ

### Q1: Why does performance evaluation report `insufficient_data`?
- **Answer:** Statistical validity requires at least 30 paired prediction-feedback records (`DEFAULT_MIN_SAMPLES = 30`). Ensure `--samples` is set to 30 or higher (default is 35).

### Q2: Can the demonstration run without an internet connection?
- **Answer:** Yes. All models, reference datasets (`data/train.csv`), and dependencies operate completely offline. Evidently AI reports are rendered locally.

### Q3: How do I inspect alerts manually in SQLite?
- **Answer:** Open `reports/demo/demo_predictions.db` with any SQLite client:
  ```sql
  SELECT alert_id, alert_type, severity, status, message, created_at, resolved_at
  FROM alerts
  ORDER BY created_at DESC;
  ```

### Q4: Does the demo overwrite training or production data?
- **Answer:** No. All demo databases and simulated datasets are isolated under `reports/demo/` and `data/simulated/demo/`. Existing production databases and training files (`data/train.csv`) remain completely untouched.
