# Automated Monitoring and Alerting Guide (Phase 3.5)

This guide documents the automated monitoring, alert evaluation engine, alert lifecycle management, and orchestration framework for the **AI-Assisted Smart Home Energy Advisor** MLOps project.

---

## 1. System Architecture

The automated monitoring subsystem continuously evaluates operational telemetry and ground truth feedback without intruding on the low-latency FastAPI inference path (`api/main.py`).

```
 +-------------------------------------------------------------------------+
 |                           Production Environment                        |
 |                                                                         |
 |   [POST /predict]   -->  predictions table (SQLite)                     |
 |   [POST /feedback]  -->  feedback table (SQLite)                        |
 +-------------------------------------+-----------------------------------+
                                       |
                                       v
               +-----------------------------------------------+
               |    Monitoring Orchestrator                    |
               |    (src/monitoring/monitor_orchestrator.py)   |
               |    - Invoked via CLI, Cron, or Task Scheduler |
               +-----------------------+-----------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
   +-------------------------------+       +-------------------------------+
   | Data Drift Engine             |       | Performance Monitor Engine    |
   | (src/monitoring/              |       | (src/monitoring/              |
   |  drift_detector.py)           |       |  performance_monitor.py)      |
   | - Reference vs. Production    |       | - Paired feedback evaluation  |
   | - SciPy KS-test & Cohen's d   |       | - MAE, RMSE, R², Bias checks  |
   +---------------+---------------+       +---------------+---------------+
                   |                                       |
                   +-------------------+-------------------+
                                       |
                                       v
               +-----------------------------------------------+
               |    Alert Manager Engine                       |
               |    (src/monitoring/alert_manager.py)          |
               |    - Severity evaluation & threshold gates    |
               |    - Deduplication & Cooldown suppression     |
               |    - Active vs. Resolved state transitions    |
               +-----------------------+-----------------------+
                                       |
                     +-----------------+-----------------+
                     |                                   |
                     v                                   v
     +-------------------------------+   +-------------------------------+
     | Alerts Table (SQLite)         |   | Notification Payloads         |
     | - Persistent alert history    |   | - Plain text summary          |
     | - Active incident tracking    |   | - Rich GitHub Markdown        |
     | - Trigger counts & timestamps |   | - Provider-independent JSON  |
     +-------------------------------+   +-------------------------------+
```

---

## 2. Alert Taxonomy & Categories

The engine standardizes alerts into three distinct categories:

| Category (`alert_type`) | Trigger Condition | Guardrails / Suppressions |
|:---|:---|:---|
| `data_drift` | Triggered when dataset drift share $\ge 50\%$ across canonical model features. | Suppressed if drift share $< 0.50$ (individual feature noise does not declare dataset drift). |
| `model_degradation` | Triggered when paired ground-truth feedback indicates metric degradation past baseline thresholds (MAE/RMSE increase $> 25\%$, $R^2 < 0.10$, or $\|\text{Bias}\| > 0.05$ kWh). | **Strictly suppressed** on `insufficient_data` ($N < 30$) or `no_feedback_data`. |
| `monitoring_execution_failure` | Triggered when an unhandled exception or data ingestion failure occurs during monitoring execution (e.g., corrupt CSV, locked DB). | Categorized distinctly from statistical drift or performance alerts. |

---

## 3. Severity Assignment Rules

Alert severity reflects the operational urgency and magnitude of the detected divergence:

### Data Drift Severity
- **`CRITICAL`:** Dataset drift share $\ge 75\%$ across features, or concept drift detected on `target_next_hour`.
- **`WARNING`:** Dataset drift share between $50\%$ and $75\%$.
- **`INFO`:** Sub-threshold feature shift or alert resolution.

### Model Degradation Severity
- **`CRITICAL`:** Severe degradation where MAE or RMSE increases by $\ge 50\%$ over historical baseline ($0.0739$ kWh and $0.1512$ kWh respectively), or $R^2 < 0.0$ (model performing worse than predicting historical mean).
- **`WARNING`:** Moderate degradation breaching policy thresholds ($25\%$ to $50\%$ increase in MAE/RMSE, $0.0 \le R^2 < 0.10$, or $\|\text{Bias}\| > 0.05$ kWh).
- **`INFO`:** Degradation condition resolved (metrics returned to healthy bounds).

### Execution Failures
- **`ERROR`:** Component failure, database access errors, or schema validation errors.

---

## 4. Alert Lifecycle, Deduplication, and Cooldown

### Deduplication Logic
To prevent alert fatigue and notification flooding, alerts are indexed by unique composite keys:
- **Drift alerts:** `(alert_type="data_drift", model_version=None)`
- **Performance alerts:** `(alert_type="model_degradation", model_version=<version>)`
- **Execution failures:** `(alert_type="monitoring_execution_failure", operation=<operation>)`

### Cooldown Window
When an alert condition is detected:
1. The engine checks the SQLite database for an existing `active` alert matching the deduplication key.
2. If an active alert exists:
   - It checks elapsed time against `cooldown_seconds` (default: `3600` seconds / 1 hour).
   - **Within cooldown:** Increments `trigger_count`, updates `last_triggered_at` and latest metrics in SQLite, but marks notification as `suppressed`.
   - **Cooldown expired:** Increments `trigger_count`, updates timestamps, and issues a recurring `triggered` reminder notification.
3. If no active alert exists:
   - A new unique alert ID (e.g. `alert_drift_a1b2c3d4e5`) is generated with `status="active"`, `trigger_count=1`.

### Resolution Lifecycle
An alert is marked `resolved` only when the underlying condition has demonstrably normalized:
- **Drift Resolution:** When a subsequent monitoring run produces `dataset_drift_detected == False`, the active drift alert transitions to `resolved` with `resolved_at` set to the current UTC timestamp.
- **Performance Resolution:** When paired evaluation reports `status == "healthy"` ($N \ge 30$ and all metrics within baseline limits), the active degradation alert transitions to `resolved`.
- **Failure Auto-Recovery:** When an operation (such as `drift_detection` or `performance_monitoring`) executes successfully in a subsequent cycle, any active `monitoring_execution_failure` alert belonging to that specific operation is automatically resolved. Failures belonging to other components remain untouched, and the original alert record (creation timestamp, trigger count, and alert ID) is fully preserved in history.
- **Guardrail:** If the monitor reports `insufficient_data` ($N < 30$), active degradation alerts are **NOT** resolved, as small sample sizes cannot prove recovery.

---

## 5. Database Schema & Migration Strategy

### SQLite Schema (`src/monitoring/database.py`)

Alert state and incident history are stored in an additive `alerts` table:

```sql
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    status TEXT NOT NULL,
    model_version TEXT,
    message TEXT NOT NULL,
    metrics_json TEXT,
    source_result_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_at TEXT,
    trigger_count INTEGER NOT NULL DEFAULT 1,
    last_triggered_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_type_version ON alerts(alert_type, model_version, status);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);
```

### Migration Strategy
- **100% Non-Breaking & Additive:** Existing operational tables (`predictions`, `feedback`) and view (`v_monitoring_records`) remain completely unchanged.
- **Automatic Migration:** `init_db()` safely creates the `alerts` table and indexes if they do not exist. Existing SQLite files require no manual schema patching or downtime.

---

## 6. Execution & Scheduling

Monitoring is designed to run independently from the web service.

### Manual CLI Execution

```bash
# 1. Run a single complete monitoring cycle (drift + performance)
.venv\Scripts\python.exe src/monitoring/monitor_orchestrator.py \
  --ref data/train.csv \
  --prod data/simulated/normal_scenario.csv \
  --model-version GradientBoostingRegressor

# 2. Run performance monitoring only (using live feedback in database)
.venv\Scripts\python.exe src/monitoring/monitor_orchestrator.py \
  --model-version GradientBoostingRegressor

# 3. Customize cooldown window (e.g., 2 hours / 7200 seconds)
.venv\Scripts\python.exe src/monitoring/monitor_orchestrator.py \
  --prod data/simulated/hvac_consumption_shift.csv \
  --cooldown 7200
```

### Python Programmatic Usage

```python
from src.monitoring.monitor_orchestrator import run_monitoring_cycle
from src.monitoring.alert_manager import AlertManager

# Execute monitoring cycle
summary = run_monitoring_cycle(
    ref_data="data/train.csv",
    prod_data="data/simulated/sensor_drift_bias.csv",
    model_version="GradientBoostingRegressor",
    cooldown_seconds=3600,
)

print(f"Cycle completed: {summary['cycle_id']}")
print(f"Active alerts: {summary['active_alerts_count']}")
for notif in summary["notifications"]:
    print(notif["body"])
```

### Scheduling Options
1. **Windows Task Scheduler / Linux Cron (Recommended):**
   - Schedule the command to run every hour:
     ```bash
     0 * * * * cd /path/to/repo && .venv/bin/python src/monitoring/monitor_orchestrator.py --prod /path/to/latest_batch.csv
     ```
2. **Built-in Daemon Loop:**
   - Run the orchestrator with `--interval-seconds 3600` to execute continuously in the background without external schedulers.

---

## 7. Notification Abstraction

The `NotificationMessage` dataclass produces structured, provider-independent messages ready for downstream ingestion:

- **Plain Text (`to_text()`):** Formatted for console logs, incident tickets, or email bodies.
- **Markdown (`to_markdown()`):** Formatted for Slack webhooks, Microsoft Teams cards, or GitHub PR / issue comments.
- **Dictionary / JSON (`to_dict()`):** Machine-readable payload containing alert IDs, metric snapshots, and recommended actions.

> [!NOTE]
> In Phase 3.5, notifications are structured in-memory and logged to SQLite. No third-party network credentials (Slack webhooks, SMTP servers, PagerDuty APIs) are required or invoked. A dedicated notification integration will connect to this abstraction in future operational phases.

---

## 8. Example Generated Alerts

### Example 1: Data Drift Alert (`WARNING`)
```json
{
  "alert_id": "alert_drift_3a4f89d2c1",
  "alert_type": "data_drift",
  "severity": "WARNING",
  "status": "active",
  "model_version": null,
  "message": "Data drift detected: 4 feature(s) drifted (66.7% share >= 50.0% threshold). Drifted features: [energy_current_hour, energy_previous_hour, energy_two_hours_ago, target_hour].",
  "metrics": {
    "drift_share": 0.6667,
    "drift_share_threshold": 0.50,
    "drifted_features_count": 4,
    "total_features_count": 6
  },
  "created_at": "2026-11-01T12:00:00+00:00",
  "trigger_count": 1
}
```

### Example 2: Model Degradation Alert (`CRITICAL`)
```json
{
  "alert_id": "alert_perf_8f9c1b4e2a",
  "alert_type": "model_degradation",
  "severity": "CRITICAL",
  "status": "active",
  "model_version": "GradientBoostingRegressor",
  "message": "Model performance degraded for 'GradientBoostingRegressor': MAE (0.1150 kWh) increased by 55.6% over baseline (0.0739 kWh), exceeding 25.0% threshold.; R² score (0.0500) dropped below minimum acceptable threshold of 0.10.",
  "metrics": {
    "mae": 0.1150,
    "rmse": 0.2200,
    "r2": 0.0500,
    "bias": 0.0100,
    "sample_count": 50
  },
  "created_at": "2026-11-01T12:00:00+00:00",
  "trigger_count": 1
}
```

### Example 3: Failure Recovery Notification (`INFO`)
```json
{
  "title": "Monitoring Recovery: drift_detection",
  "severity": "INFO",
  "event_type": "resolved",
  "timestamp": "2026-11-01T13:00:00+00:00",
  "body": "Execution for component 'drift_detection' succeeded. Active failure alert resolved.",
  "metrics_summary": "operation=drift_detection, status=recovered",
  "recommended_action": "Normal monitoring operations restored for this component.",
  "alert_id": "alert_fail_7b3e1c9d2f"
}
```
