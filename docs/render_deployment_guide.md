# Render Cloud Deployment Guide (Phase 3.8)

**AI-Assisted Smart Home Energy Advisor — Production Cloud Deployment**
**Repository:** [https://github.com/YashP001/smartenerymlops](https://github.com/YashP001/smartenerymlops)
**Configuration File:** `render.yaml`
**Python Runtime Pin:** `.python-version` (`3.11.9`)

---

## 1. Overview & Service Architecture

This guide provides instructions for deploying the **Smart Home Energy Advisor** FastAPI service to [Render](https://render.com) as a Native Python Web Service.

```
┌─────────────────────────────────────────────────────────────┐
│                    RENDER CLOUD PLATFORM                    │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Web Service: smart-home-energy-advisor             │   │
│   │  Runtime: Python 3.11.9 | Region: Singapore         │   │
│   │                                                     │   │
│   │  • Build: pip install -r requirements-inference.txt │   │
│   │  • Start: uvicorn api.main:app --host 0.0.0.0       │   │
│   │           --port $PORT --workers 1                  │   │
│   │                                                     │   │
│   │  ┌───────────────────────────────────────────────┐  │   │
│   │  │ FastAPI Application (api/main.py)             │  │   │
│   │  │                                               │  │   │
│   │  │ • GET  /health   -> Liveness check (200)      │  │   │
│   │  │ • GET  /ready    -> Model & DB readiness (200)│  │   │
│   │  │ • POST /predict  -> Energy inference (200)    │  │   │
│   │  │ • POST /feedback -> Ground-truth pairing (200)│  │   │
│   │  │ • GET  /metrics  -> Prometheus exposition     │  │   │
│   │  │ • GET  /docs     -> Interactive Swagger UI    │  │   │
│   │  └───────────────────────┬───────────────────────┘  │   │
│   │                          │                          │   │
│   │                          ▼                          │   │
│   │             SQLite: predictions.db                  │   │
│   │             (Ephemeral Disk)                        │   │
│   │                                                     │   │
│   │  ┌───────────────────────────────────────────────┐  │   │
│   │  │ Committed Model Checkpoint                    │  │   │
│   │  │ (models/gradient_boosting.pkl - 62 KB)        │  │   │
│   │  └───────────────────────────────────────────────┘  │   │
│   └──────────────────────────▲──────────────────────────┘   │
└──────────────────────────────┼──────────────────────────────┘
                               │ HTTPS / Public Internet
                               ▼
        [External Users / Smart Meter Telemetry / Prometheus Scraper]
```

### Key Architectural Characteristics
1. **Lightweight Runtime Footprint:** Uses `requirements-inference.txt` (only 10 runtime packages; no heavy dev/training libraries like Jupyter, Matplotlib, or DVC).
2. **Offline Local Model Execution:** Loads `models/gradient_boosting.pkl` directly from the Git repository. Operates without an external MLflow server or third-party cloud credentials.
3. **Dedicated Readiness Probe:** Render's health check is routed to `GET /ready`. It returns HTTP 200 only when both the machine learning model is loaded and the SQLite monitoring persistence layer is initialized.
4. **Prometheus Exposition:** The `/metrics` endpoint exposes real-time inference latency, HTTP request rates, and prediction counts for scraping.

---

## 2. Prerequisites

Before deploying, ensure you have:
1. A [Render Account](https://dashboard.render.com).
2. A GitHub account with access to the repository [https://github.com/YashP001/smartenerymlops](https://github.com/YashP001/smartenerymlops).
3. The repository files `render.yaml`, `.python-version`, `requirements-inference.txt`, and `models/gradient_boosting.pkl` present on the `main` branch.

---

## 3. Deployment Methods

### Option A: Deploy via Render Blueprints (Recommended)

Render Blueprints automatically parse `render.yaml` at the root of the repository and configure all settings declaratively.

1. Log in to the [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** in the top navigation and select **Blueprint**.
3. Connect your GitHub account and select the repository **`YashP001/smartenerymlops`**.
4. Choose the **`main`** branch.
5. Render will automatically detect `render.yaml` and display the service specification:
   - **Service Name:** `smart-home-energy-advisor`
   - **Service Type:** Web Service
   - **Runtime:** Python
   - **Region:** Singapore
   - **Plan:** Free
   - **Health Check Path:** `/ready`
6. Click **Apply**. Render will trigger the initial build and deploy.

---

### Option B: Manual Web Service Creation

If you prefer to configure the service manually via the Render UI:

1. In the Render Dashboard, click **New +** and select **Web Service**.
2. Connect the repository **`YashP001/smartenerymlops`** and select the **`main`** branch.
3. Fill in the service configuration:
   - **Name:** `smart-home-energy-advisor`
   - **Region:** `Singapore` (or preferred region)
   - **Runtime:** `Python`
   - **Build Command:**
     ```bash
     pip install --no-cache-dir -r requirements-inference.txt
     ```
   - **Start Command:**
     ```bash
     uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1
     ```
   - **Instance Type / Plan:** `Free`
4. Expand **Advanced** and set:
   - **Health Check Path:** `/ready`
   - **Auto-Deploy:** `Yes`
5. Add the following **Environment Variables**:
   | Key | Value | Description |
   | :--- | :--- | :--- |
   | `PYTHON_VERSION` | `3.11.9` | Matches `.python-version` for reproducible Python build |
   | `PYTHONUNBUFFERED` | `1` | Ensures real-time stdout/stderr log streaming |
   | `MLFLOW_DISABLE_AGENT_HINT` | `1` | Suppresses non-runtime MLflow tracing notices |
   | `MONITORING_DB_PATH` | `data/monitoring/predictions.db` | Sets relative path for SQLite persistence |
6. Click **Create Web Service**.

---

## 4. Monitoring the Initial Deployment

During the build and deployment process:
1. Navigate to the **Logs** tab of your service in the Render dashboard.
2. Confirm the build logs output:
   - Python 3.11.9 environment setup.
   - Successful installation of `requirements-inference.txt`.
   - Build exit code 0.
3. Confirm the application deploy logs show:
   ```text
   Monitoring persistence layer initialized.
   Notice: MLflow registry loading unavailable (...). Falling back to local artifact.
   Loaded model from local fallback: '.../models/gradient_boosting.pkl'
   Model loaded successfully into application state.
   INFO:     Started server process [...]
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://0.0.0.0:10000
   ```
4. Render will issue a probe to `GET /ready`. Once it receives HTTP 200, the status transitions to **Live**.

---

## 5. Live Endpoint Verification

Once the service is Live, replace `<YOUR_RENDER_URL>` (e.g., `https://smart-home-energy-advisor.onrender.com`) with your actual service URL and run the following tests:

### 1. Liveness Check (`GET /health`)
```bash
curl -i https://<YOUR_RENDER_URL>/health
```
**Expected Response:** HTTP 200 OK
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Readiness Check (`GET /ready`)
```bash
curl -i https://<YOUR_RENDER_URL>/ready
```
**Expected Response:** HTTP 200 OK
```json
{
  "status": "ready",
  "model_loaded": true,
  "model_version": "GradientBoostingRegressor",
  "database_initialized": true
}
```

### 3. Interactive Documentation (`GET /docs`)
Open your browser to:
```
https://<YOUR_RENDER_URL>/docs
```
Confirms OpenAPI / Swagger UI interactive documentation is accessible.

### 4. Prediction Request (`POST /predict`)
```bash
curl -X POST https://<YOUR_RENDER_URL>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "energy_current_hour": 0.05,
    "energy_previous_hour": 0.04,
    "energy_two_hours_ago": 0.06,
    "target_hour": 14,
    "target_day_of_week": 2,
    "target_is_weekend": 0
  }'
```
**Expected Response:** HTTP 200 OK with prediction payload and `X-Prediction-ID` header:
```json
{
  "predicted_energy_kwh": 0.06288,
  "model": "GradientBoostingRegressor",
  "features_used": [
    "energy_current_hour",
    "energy_previous_hour",
    "energy_two_hours_ago",
    "target_hour",
    "target_day_of_week",
    "target_is_weekend"
  ],
  "prediction_id": "0c6048b8-0643-4e59-a66f-444d5b1ae15"
}
```

### 5. Ground-Truth Feedback Ingestion (`POST /feedback`)
Use the `prediction_id` returned from the `/predict` call:
```bash
curl -X POST https://<YOUR_RENDER_URL>/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": "0c6048b8-0643-4e59-a66f-444d5b1ae15",
    "actual_energy_kwh": 0.054
  }'
```
**Expected Response:** HTTP 200 OK
```json
{
  "status": "success",
  "prediction_id": "0c6048b8-0643-4e59-a66f-444d5b1ae15",
  "actual_energy_kwh": 0.054,
  "feedback_timestamp": "2026-11-01T10:14:09.801342+00:00"
}
```

### 6. Prometheus Metrics Exposition (`GET /metrics`)
```bash
curl https://<YOUR_RENDER_URL>/metrics
```
**Expected Response:** Plaintext Prometheus metrics output containing counters and gauges:
```text
energy_advisor_http_requests_total{endpoint="/predict",method="POST",status="200"} 1.0
energy_advisor_prediction_requests_total{model_version="GradientBoostingRegressor",status="success"} 1.0
energy_advisor_prediction_latency_seconds_bucket{le="0.05"} 1.0
```

---

## 6. Operational Considerations on Render

### Free Tier Behavior & Cold Starts
- **15-Minute Inactivity Spin-Down:** On Render's Free tier, the web service automatically spins down (sleeps) after 15 minutes of receiving no HTTP traffic.
- **Cold Start Latency:** The next incoming request will wake the container. During a cold start, response latency can take **30 to 50 seconds** while Render provisions the container, runs Python initialization, initializes SQLite, and loads the model into memory. Subsequent requests will respond in milliseconds (< 20 ms).

### Ephemeral Storage vs. Persistent Disks
- **Current Behavior (Ephemeral Storage):** The SQLite database (`data/monitoring/predictions.db`) is written to the container's local ephemeral filesystem.
  - While the service remains running, predictions, feedback, and `/metrics` work seamlessly.
  - When the container spins down, redeploys, or restarts, **all records stored in SQLite are reset**.
- **How to Enable Persistent Storage (Optional Future Upgrade):**
  - Persistent Disks on Render require an upgraded paid instance (e.g. Starter tier at $7/month).
  - To enable persistent storage in `render.yaml` in the future:
    ```yaml
    disk:
      name: monitoring-data
      mountPath: /data
      sizeGB: 1
    ```
    And update the environment variable to point to the mounted disk:
    ```yaml
    envVars:
      - key: MONITORING_DB_PATH
        value: /data/monitoring/predictions.db
    ```
  *(Note: This is an optional infrastructure choice. The application operates fully out-of-the-box on ephemeral storage for demonstration and evaluation).*

---

## 7. Troubleshooting Common Deployment Failures

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **Deploy fails with `Port scan timeout`** | Uvicorn was configured with a hardcoded port (e.g. `8000`) instead of `$PORT`. | Verify the start command in `render.yaml` uses `--port $PORT`. Render dynamically assigns this port at container launch. |
| **Health check fails on `/ready` with HTTP 503** | The model artifact `models/gradient_boosting.pkl` was missing or the database failed initialization. | Check the deployment logs. Ensure `models/gradient_boosting.pkl` is tracked and committed to Git (it is ~62 KB and tracked in the repository). |
| **Build fails during `pip install`** | Memory limit exceeded or wheel compilation failure. | Ensure the build command uses `requirements-inference.txt` (not the full `requirements.txt`). Verify `.python-version` is set to `3.11.9`. |
| **Requests fail with CORS errors from browser** | Browser clients blocked by cross-origin security. | The REST API is designed for server-to-server and curl requests. If a browser frontend is deployed later, enable `CORSMiddleware` in `api/main.py`. |
