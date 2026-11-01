# Observability Guide: Prometheus & Grafana Integration

**Phase 3.6 Deliverable**
**AI-Assisted Smart Home Energy Advisor MLOps Pipeline**

---

## 1. Observability Architecture Overview

The Smart Home Energy Advisor uses an integrated observability architecture combining **Prometheus** (metrics scraping and time-series storage) and **Grafana** (dashboard visualization and alerting).

The system decouples operational inference from scheduled monitoring orchestration while providing a unified metrics exposition endpoint at `GET /metrics`.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                       DOCKER COMPOSE                                   │
│                                                                                        │
│   ┌────────────────────────┐                   ┌───────────────────────────────────┐  │
│   │   FastAPI Service      │                   │            Prometheus             │  │
│   │   (api:8000)           │                   │            (prometheus:9090)      │  │
│   │                        │                   │                                   │  │
│   │  • /health             │  Scrapes /metrics │  • Scrapes api:8000 every 15s     │  │
│   │  • /predict            │ ◄─────────────────┤  • Evaluates PromQL rules         │  │
│   │  • /feedback           │                   │  • 15-day TSDB retention          │  │
│   │  • /metrics            │                   └─────────────────┬─────────────────┘  │
│   └───────────┬────────────┘                                     │                     │
│               │                                                  │ PromQL Queries      │
│               │ Read/Write                                       ▼                     │
│               │                                ┌───────────────────────────────────┐  │
│               ▼                                │             Grafana               │  │
│   ┌────────────────────────┐                   │             (grafana:3000)        │  │
│   │  SQLite Monitoring DB  │                   │                                   │  │
│   │  (predictions.db)      │                   │  • Provisioned Prometheus Source  │  │
│   │                        │                   │  • Auto-loaded MLOps Dashboard    │  │
│   │  • predictions         │                   │  • Anonymous viewer access        │  │
│   │  • feedback            │                   └───────────────────────────────────┘  │
│   │  • alerts              │                                                          │
│   │  • monitoring_runs     │                                                          │
│   │  • failures            │                                                          │
│   └───────────▲────────────┘                                                          │
│               │                                                                        │
│               │ Read/Write                                                             │
│   ┌───────────┴────────────┐                                                          │
│   │ Monitoring Orchestrator│ (Runs via Cron / CLI / Task Scheduler)                   │
│   │ (src.monitoring)       │                                                          │
│   └────────────────────────┘                                                          │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Architecture Key Principles

1. **Unified Metrics Exposition**: All metrics (both synchronous API inference and asynchronous monitoring cycles) are exposed through a single `/metrics` endpoint served by the FastAPI application.
2. **Decoupled Monitoring Synchronization**: The monitoring orchestrator operates out-of-band as a separate CLI or scheduled job. It persists cycle summaries, drift results, performance metrics, and alert states to the shared SQLite database (`predictions.db`).
3. **Dynamic Custom Collector**: When Prometheus scrapes `GET /metrics`, FastAPI executes `MonitoringMetricsCollector`, querying the latest state directly from SQLite. This eliminates the need for Prometheus Pushgateway or error-prone multiprocess shared-memory directories (`PROMETHEUS_MULTIPROC_DIR`).
4. **Zero Fabrication Policy**: If no operational feedback exists or a monitoring cycle has not yet run, metrics such as MAE, RMSE, and drift share are **not emitted**. Panels in Grafana gracefully display `No Data` or `No Feedback Data` rather than fabricated zeroes.
5. **Low-Cardinality Guarantee**: High-cardinality attributes (UUIDs, timestamps, raw meter readings, free-form error strings) are strictly excluded from metric labels. Dynamic endpoint URLs (e.g. `/predictions/{prediction_id}/feedback`) are automatically normalized to route templates.

---

## 2. Metrics Catalog

### API & Inference Metrics

| Metric Name | Type | Labels | Description |
|:---|:---:|:---|:---|
| `energy_advisor_prediction_requests_total` | Counter | `model_version`, `status` | Total prediction requests received, broken down by success (`status="success"`) and failure (`status="error"`). |
| `energy_advisor_prediction_latency_seconds` | Histogram | `model_version`, `le` | Inference latency in seconds. Buckets: `[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5]`. |
| `energy_advisor_http_requests_total` | Counter | `method`, `endpoint`, `status_code` | Total HTTP requests processed by method, normalized endpoint template, and HTTP response status code. |
| `energy_advisor_http_request_duration_seconds` | Histogram | `method`, `endpoint`, `le` | End-to-end HTTP request duration in seconds across all endpoints. |
| `energy_advisor_model_predictions_total` | Counter | `model_version` | Cumulative count of successful predictions generated by the active machine learning model. |
| `energy_advisor_predicted_energy_kwh` | Histogram | `model_version`, `le` | Distribution of predicted hourly electricity consumption in kWh. Buckets: `[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0]`. |

### Model Performance & Monitoring Metrics

| Metric Name | Type | Labels | Description |
|:---|:---:|:---|:---|
| `energy_advisor_drift_detection_runs_total` | Counter | *(none)* | Total data drift detection runs executed by the monitoring orchestrator. |
| `energy_advisor_drift_detection_failures_total` | Counter | *(none)* | Total failed data drift detection runs. |
| `energy_advisor_dataset_drift_share` | Gauge | *(none)* | Current fraction of drifted input features (0.0 to 1.0) from the most recent drift analysis. |
| `energy_advisor_performance_evaluation_runs_total` | Counter | *(none)* | Total model performance evaluations executed against ground-truth smart-meter feedback. |
| `energy_advisor_performance_evaluation_failures_total` | Counter | *(none)* | Total failed model performance evaluations. |
| `energy_advisor_model_mae` | Gauge | `model_version` | Mean Absolute Error (MAE) in kWh evaluated on operational feedback (baseline: `0.0739 kWh`). Only emitted when sufficient feedback samples exist (`>= 30`). |
| `energy_advisor_model_rmse` | Gauge | `model_version` | Root Mean Squared Error (RMSE) in kWh evaluated on operational feedback (baseline: `0.1512 kWh`). Only emitted when sufficient feedback samples exist (`>= 30`). |
| `energy_advisor_active_alerts` | Gauge | `alert_type`, `severity` | Current count of active, unresolved alerts grouped by alert type (`data_drift`, `model_degradation`, `monitoring_execution_failure`) and severity (`INFO`, `WARNING`, `CRITICAL`, `ERROR`). |
| `energy_advisor_monitoring_execution_failures_total` | Counter | `operation` | Cumulative count of monitoring orchestration task execution failures grouped by operation (`drift_detection`, `performance_monitoring`). |

---

## 3. Getting Started with Docker Compose

### Prerequisites
- Docker Engine 24.0+ and Docker Compose v2.20+ (or Docker Desktop).
- Port availability: `8000` (FastAPI), `9090` (Prometheus), `3000` (Grafana).

### Starting Services
Run the following command from the repository root:

```bash
docker compose up -d --build
```

### Verifying Service Health
Check running container status:

```bash
docker compose ps
```

Expected output:
```
NAME                           IMAGE                    COMMAND                  SERVICE      STATUS                    PORTS
smart-home-energy-api          smart-home-energy-api    "uvicorn api.main:ap…"   api          Up (healthy)              0.0.0.0:8000->8000/tcp
smart-home-energy-prometheus   prom/prometheus:v2.53.0  "/bin/prometheus --c…"   prometheus   Up                        0.0.0.0:9090->9090/tcp
smart-home-energy-grafana      grafana/grafana:11.1.0   "/run.sh"                grafana      Up                        0.0.0.0:3000->3000/tcp
```

### Stopping Services
To stop services while preserving database and monitoring volumes:

```bash
docker compose down
```

To stop services and wipe all data volumes:

```bash
docker compose down -v
```

---

## 4. Accessing Dashboards and Endpoints

| Service | URL | Authentication | Notes |
|:---|:---|:---|:---|
| **FastAPI Swagger Docs** | `http://localhost:8000/docs` | None | Interactive OpenAPI exploration |
| **FastAPI Metrics Endpoint** | `http://localhost:8000/metrics` | None | Raw Prometheus exposition text |
| **FastAPI Health Check** | `http://localhost:8000/health` | None | Health status and model load state |
| **Prometheus Web UI** | `http://localhost:9090` | None | Targets, TSDB status, raw PromQL queries |
| **Grafana Dashboard** | `http://localhost:3000` | Anonymous Viewer (or `admin`/`admin`) | Auto-loaded "Smart Home Energy Advisor - MLOps Observability" dashboard |

---

## 5. Interpreting Dashboard Panels

The provisioned Grafana dashboard (`monitoring/grafana/dashboards/smart-home-energy-advisor.json`) is organized into three analytical tiers:

### Tier 1: System Overview & Alert Status (Top Row)
1. **Data Drift Share (Gauge)**:
   - **Range**: `0.0` (0%) to `1.0` (100%).
   - **Green**: `< 0.33` (fewer than 2 of 6 canonical features drifting).
   - **Yellow (Warning)**: `>= 0.33` (2 of 6 features drifting, triggers warning alert).
   - **Red (Critical)**: `>= 0.50` (3 or more of 6 features drifting, triggers critical alert).
2. **Model MAE (Stat Panel)**:
   - **Green**: `< 0.0739 kWh` (healthy baseline accuracy).
   - **Yellow**: `0.0739 – 0.0924 kWh` (slight error increase).
   - **Red**: `> 0.0924 kWh` (+25% degradation threshold exceeded, triggers alert).
   - **Empty State**: Displays `No Feedback Data` when paired feedback is `< 30` samples.
3. **Model RMSE (Stat Panel)**:
   - **Green**: `< 0.1512 kWh` (healthy baseline accuracy).
   - **Yellow**: `0.1512 – 0.1890 kWh` (moderate variance increase).
   - **Red**: `> 0.1890 kWh` (+25% degradation threshold exceeded, triggers alert).
   - **Empty State**: Displays `No Feedback Data` when paired feedback is `< 30` samples.
4. **Total Active Alerts (Stat Panel)**:
   - Displays count of unresolved alerts currently open in SQLite.
   - **Green**: `0` active alerts.
   - **Yellow**: `1 – 2` active alerts.
   - **Red**: `>= 3` active alerts.

### Tier 2: Inference Traffic & Latency (Middle Row)
5. **Prediction Request Volume (Time Series)**:
   - Evaluates `sum by (model_version, status) (rate(energy_advisor_prediction_requests_total[5m]))`.
   - Displays real-time request rates (req/sec) split by success and error.
6. **API Inference Latency (Time Series)**:
   - P95 Latency: `histogram_quantile(0.95, sum(rate(energy_advisor_prediction_latency_seconds_bucket[5m])) by (le))`.
   - P50 Latency: `histogram_quantile(0.50, sum(rate(energy_advisor_prediction_latency_seconds_bucket[5m])) by (le))`.
   - Average Latency: `sum(rate(energy_advisor_prediction_latency_seconds_sum[5m])) / sum(rate(energy_advisor_prediction_latency_seconds_count[5m]))`.
7. **Prediction Success vs Failure Rate (Time Series)**:
   - Displays percentages of successful requests vs HTTP/inference errors over 5-minute rolling windows.

### Tier 3: Model Distribution & Monitoring Deep Dive (Bottom Row)
8. **Model Prediction Output Distribution (Time Series Bars)**:
   - Visualizes consumption predictions across histogram buckets (`<= 0.05 kWh`, `<= 0.10 kWh`, `<= 0.50 kWh`, etc.) to detect prediction distribution anomalies.
9. **Active Alerts by Type & Severity (Horizontal Bar Gauge)**:
   - Breaks down open alerts by classification (`data_drift`, `model_degradation`, `monitoring_execution_failure`) with color coding (WARNING = yellow, CRITICAL = red, ERROR = purple).
10. **Monitoring Execution Failures (Stat Panel)**:
    - Displays cumulative failed orchestrator operations to highlight infrastructure faults.

---

## 6. How the Monitoring Orchestrator Contributes Metrics

The monitoring orchestrator (`src/monitoring/monitor_orchestrator.py`) is decoupled from the FastAPI request/response lifecycle:

1. **Cycle Execution**: An external scheduler (cron, Windows Task Scheduler, or CLI `--interval-seconds`) triggers `run_monitoring_cycle()`.
2. **Analysis**:
   - `run_drift_analysis()` calculates Evidently feature drift against `data/train.csv`.
   - `evaluate_model_performance()` computes MAE, RMSE, $R^2$, and bias from paired records in `v_monitoring_records`.
   - `AlertManager` evaluates threshold rules and resolves recovered conditions.
3. **Persistence**:
   - The cycle summary is written to the `monitoring_runs` table in `predictions.db`.
   - Active and resolved alerts are updated in the `alerts` table.
   - Failures are recorded in `monitoring_execution_failures`.
4. **Scrape-Time Aggregation**:
   - When Prometheus polls `GET /metrics`, the custom `MonitoringMetricsCollector` runs queries against `predictions.db`.
   - It reads run counts, active alert counts, and the latest valid drift share / MAE / RMSE values.
   - If no valid run exists or feedback is insufficient, those metric families are omitted rather than reported as zero.

To manually trigger a monitoring cycle that updates metrics:

```bash
# Evaluate drift on simulated normal scenario and performance on current database
python -m src.monitoring.monitor_orchestrator --prod data/simulated/normal_scenario.csv
```

---

## 7. Troubleshooting Missing Metrics

### Symptom: `GET /metrics` Returns 404 or Connection Refused
- Verify the API container is running: `docker compose ps api`.
- Check API container logs: `docker compose logs api`.
- Verify port 8000 is open: `curl -I http://localhost:8000/health`.

### Symptom: Prometheus Target Status Shows `DOWN`
- Open Prometheus Web UI: `http://localhost:9090/targets`.
- If the endpoint `http://api:8000/metrics` is marked `DOWN`:
  - Check container network: ensure both containers are on `energy_advisor_network`.
  - In Docker Compose, the API service name is `api`, which resolves via Docker DNS.
  - Test connectivity from inside Prometheus container:
    ```bash
    docker compose exec prometheus wget -qO- http://api:8000/metrics
    ```

### Symptom: Grafana Panels Display "No Data"
- **MAE / RMSE Panels**: This is expected behavior when fewer than 30 feedback records exist. Submit 30+ predictions and feedback entries, then run a monitoring cycle:
  ```bash
  python -m src.monitoring.monitor_orchestrator
  ```
- **Drift Share Panel**: This is expected behavior when no drift analysis has been executed yet. Run an orchestrator cycle with `--prod` data.
- **Traffic / Latency Panels**: Send test inference traffic to populate counters:
  ```bash
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"energy_current_hour":0.05,"energy_previous_hour":0.04,"energy_two_hours_ago":0.06,"target_hour":14,"target_day_of_week":2,"target_is_weekend":0}'
  ```

---

## 8. Limitations: Local Demonstration vs Production Deployment

| Aspect | Local Demonstration (Phase 3.6) | Production MLOps Deployment |
|:---|:---|:---|
| **Persistence Engine** | SQLite database with WAL mode on shared volume. | Distributed PostgreSQL or Amazon RDS with connection pooling. |
| **Metrics Storage** | Single-node Prometheus with local volume retention (15 days). | High-availability Prometheus pair with Thanos or Cortex for long-term cloud object storage. |
| **Container Orchestration** | Docker Compose on single host. | Kubernetes (EKS/GKE) with Helm charts and Horizontal Pod Autoscaling (HPA). |
| **Authentication & RBAC** | Anonymous viewer access with default `admin` credentials. | OAuth2 / OIDC SSO (Okta, Azure AD, Keycloak) with role-based dashboard permissions. |
| **Monitoring Scheduling** | Local polling loop or CLI execution. | Kubernetes CronJob or Airflow/Prefect DAG running every hour or midnight UTC. |
| **Alert Routing** | SQLite persistence with stdout logging. | Prometheus Alertmanager routing to PagerDuty, Slack, and email with escalation policies. |
| **TLS / SSL** | Plain HTTP over internal bridge network. | Ingress controller (NGINX/Traefik) terminating TLS with Let's Encrypt certificates. |
