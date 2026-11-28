# Phase 3 Person 1: continuous monitoring demonstration

## Objective and boundaries

Demonstrate normal operation → HVAC shift → sensor bias → explicit performance
degradation → repeated alerts → recovery, with fresh batches and live monitoring
metrics. This extends the existing implementation, not a replacement pipeline.
It does not train a model, register a version, update MLflow artifacts, change the
six-feature schema, edit historical CSVs, or change Render configuration.

The [implementation report](person1_implementation_report.md) records the initial
audit, measured results, runtime verification, test results, and limitations.

## Architecture and data sources

```text
Presenter's set-scenario command → atomic state.json
                                       ↓ read on EVERY cycle
Independent demo controller → existing simulator → unique production batch CSV
             ↓                                     ↓
Held-out feature replay → /predict or local model   existing drift detector
             ↓                                     ↓
Prediction IDs → actual feedback → existing performance monitor (current IDs only)
                                       ↓
                          existing alert manager
                                       ↓
                  shared dedicated SQLite demonstration store
                                       ↓ queried on each /metrics scrape
                           FastAPI → Prometheus → Grafana
```

- Reference: `data/train.csv`, read only.
- Accuracy source: `data/test.csv`, sampled without replacement within a batch.
  This is a replay of historical held-out observations, not newly measured homes.
- Drift input: the existing `src/simulation.py` generator. A unique CSV is written
  under `reports/demo/continuous/batches/<session>_<cycle>/` for every cycle.
- Healthy and recovery feedback: genuine held-out targets paired with predictions
  on those exact held-out feature vectors.
- Degraded feedback: **SIMULATED** held-out actual target plus `0.4 × severity`
  kWh (0.48 kWh at default severity 1.2). Inputs and predictions are unchanged.
- HVAC/sensor drift accuracy is evaluated on a **separate healthy held-out replay**,
  deliberately isolating input drift from accuracy. Those MAE values do not measure
  accuracy on the synthetic HVAC/sensor population.

The local default checkpoint is `models/gradient_boosting.pkl`, the Phase 1 model
trained on train+validation. Registry version 1 is a distinct train-only checkpoint.
The default demo loads the explicit pickle, never the registry. Native batches record
the model artifact SHA-256 and the historical baseline provenance. API mode records
the existing API model label; that API does not expose an artifact hash/version ID.
Set `ENERGY_MODEL_PATH` as shown below to make its checkpoint unambiguous.
The baseline is fixed to this Phase 1 checkpoint, not inferred from an arbitrary
`--model-path` or remote API. Before changing checkpoints, Person 2 must provide
matching model identity and baseline handling; the store-ID check verifies the
database connection, not the model artifact.

## Repeatability, freshness, and evaluation windows

Default settings: 500 observations, seed 42, severity 1.2, 10-second interval.
Cycle `k` uses seed `42 + k - 1`, so a restart from the same starting scenario and
sequence reproduces observations and metrics. UUIDs and wall-clock timestamps
intentionally differ. Do not expect different sample counts/seeds to have identical
metrics. Small or unrepresentative batches may honestly fail the healthy thresholds.

Every cycle generates a fresh simulated batch and replays a new held-out sample.
Only that batch's explicit prediction IDs are passed to performance monitoring.
The filter combines with model identity and optional time bounds. No timestamp
rounding, rolling-average dilution, deletion of records, or manual alert resolution
is used for recovery. Old degraded feedback stays in SQLite and is excluded from
the new healthy batch's assessment. Sample counts may not exceed held-out rows.

`--interval` is the pause **after** a completed batch, not a guaranteed wall-clock
frequency. Prediction/feedback ingestion adds runtime. One controller should own
the demo store at a time. It runs independently of request handlers. SIGINT/Ctrl+C
or SIGTERM requests shutdown after the current batch, interrupting the interval
wait immediately. All records and artifacts are retained for review.

The older orchestrator CLI still supports rereading a supplied CSV on an interval.
It does not itself produce new data. Use this controller for a changing-data demo.

## Scenarios

| Command value | Input-drift data | Paired accuracy feedback |
|---|---|---|
| `normal` | Existing normal conditioned resampling with small jitter | Genuine held-out actuals |
| `hvac_shift` | Existing increased afternoon/evening load, including lag features | Separate genuine held-out actuals |
| `sensor_drift` | Existing energy calibration bias, noise, and occasional spikes | Separate genuine held-out actuals |
| `model_degradation` | Configurable input scenario; simulated increased target | Simulated held-out actuals + offset |
| `recovery` | Fresh normal simulation | Fresh genuine held-out actuals |

The sensor simulation modifies derived energy measurements. It does not simulate
separate voltage/current channels. HVAC modifies demand; sensor drift modifies the
measurement process. The synthetic generator conditionally samples historical
energy tuples; its hourly timestamps do not make it a physical continuous-household
time-series simulator.

Native mode defaults to `--degradation-input normal` to demonstrate accuracy
degradation without input drift. The live Docker profile selects
`--degradation-input sensor_drift`: the data-drift alert remains active while a
separate model alert appears, and **both resolve together at final recovery**.
Choose `normal`, `hvac_shift`, or `sensor_drift` explicitly; the selected input
scenario is recorded in each batch's provenance. Accuracy still uses the separate
held-out replay, so the two effects are not falsely equated.

## Detection methodology and unchanged thresholds

A numerical feature drifts when either:

1. KS p-value < 0.05 **and** KS distance ≥ 0.15, or
2. KS p-value < 0.05 **and** absolute Cohen's d ≥ 0.20.

Constant/disjoint feature cases use the detector's existing guards. Dataset drift
share is the number of drifted canonical input features divided by six. A share
≥ 0.50 triggers a dataset alert. Statistical significance alone is insufficient:
large samples can make negligible shifts significant, so effect size/distance
guards remain in place. Calendar variables are discrete; the existing generic KS
approach is an educational approximation, not a fully calibrated categorical test.

Target drift is checked separately and excluded from the six-feature denominator.
Input drift concerns P(X); target drift concerns marginal P(y). Concept drift means
P(y|X) changed. Neither input nor marginal target drift alone proves concept drift.
The target-offset demonstration changes the relationship by construction and its
paired errors establish degradation under the simulated scenario.

Standard KS p-values assume independent observations; smart-meter series are
autocorrelated and can have seasonality. The fixed thresholds are demonstration
heuristics, not calibrated production false-alarm guarantees. Block resampling,
seasonal reference windows, and multiple-testing control would need separate
validation before real deployment. A minimum sample count is useful protection,
not proof of sample independence.

| Performance rule | Existing value |
|---|---:|
| Historical baseline MAE | 0.0739 kWh |
| Historical baseline RMSE | 0.1512 kWh |
| Historical baseline R² | 0.3723 |
| Maximum relative MAE / RMSE increase | 25% / 25% |
| Effective MAE / RMSE upper bounds | 0.092375 / 0.189 kWh |
| Minimum R² | 0.10 |
| Maximum absolute mean prediction bias | 0.05 kWh |
| Minimum paired feedback records | 30 |

Bias is mean(prediction − actual), so the positive simulated target offset produces
negative bias. Missing feedback and fewer than 30 pairs never resolve an existing
degradation alert. No threshold is relaxed to make a scenario pass.

## Alert lifecycle

The existing manager handles separate `data_drift`, `model_degradation`, and
`monitoring_execution_failure` categories. Drift is WARNING at 50–75% share and
CRITICAL at ≥75%, or when target drift accompanies dataset drift. Performance
severity follows the existing 50% error-increase/negative-R² critical policy.

Repeated conditions reuse the same active alert ID and increment `trigger_count`.
Notifications are suppressed for the default 3600-second cooldown. The new
`last_notified_at` field measures cooldown from the last emitted notification;
suppressed detections still update `last_triggered_at` without restarting cooldown.
After cooldown, another reminder can be emitted. These are local structured
notifications, not email/Slack delivery.

Fresh no-drift results resolve the drift alert. Healthy paired performance resolves
the matching model degradation alert. Recovery timestamps and resolved alert rows
remain in SQLite. Ingestion faults are recorded as execution failures rather than
statistical drift; a successful later ingestion resolves its own failure alert.

## Windows startup: complete local dashboard stack

Run Docker Desktop, open PowerShell, and use the repository root:

```powershell
Set-Location 'E:\mlops phase 2\smart-home-energy-advisor'
.venv\Scripts\python.exe scripts/run_continuous_monitoring_demo.py set-scenario normal
docker compose up --build -d
docker compose ps
docker compose logs -f monitor
```

The stack shares the Docker volume `monitoring_store` between API and monitor, with
the same container path and the same `MONITORING_DB_PATH`
(`/app/monitoring-store/predictions.db`). State and evidence files use
`reports/demo/continuous` on the host. SQLite stays on Docker's Linux filesystem to
avoid slow per-record I/O through a Windows bind mount. An opaque store ID in the API
`/metrics` header must match SQLite **before any inference writes**. The normal host
`data/monitoring` store is never mounted. Historical CSV mounts are read only. The API
serves the deployed MLflow registry version (v1 until promoted), bootstrapped by the
one-shot `registry-init` service; `/ready` and every prediction report the version and run ID.

The monitoring services opt into `MONITORING_SQLITE_SYNCHRONOUS=NORMAL` with WAL.
Transactions remain consistent, but the latest commits may be lost on a
power failure. The normal application keeps SQLite's default FULL setting. This
is a local presentation performance choice, not a change to production defaults.

This is a mode switch within the same Compose project. The normal and demonstration
stacks reuse service names and ports 8000/9090/3000 and cannot run concurrently as
independent stacks with this configuration. The demo replaces the API's database
mount; the normal host database is retained. Prometheus/Grafana volumes are shared,
so their historical charts can contain observations from both modes under the
existing local labels. Use one mode at a time and a suitable dashboard time range.

To return to the normal application without deleting either database or history:

```powershell
docker compose stop monitor
docker compose -f docker-compose.yml up --build -d api prometheus grafana
```

Do not use `down -v` to switch modes: it removes named volumes. For a completely
separate demo installation, Person 4 must also separate container names, ports,
Compose project/volumes, and metric labels.

Open:

- API readiness: <http://localhost:8000/ready>
- Actual metrics: <http://localhost:8000/metrics>
- Prometheus targets: <http://localhost:9090/targets> (`api:8000` must be UP)
- Grafana: <http://localhost:3000/d/smart-home-energy-advisor>

The existing Compose setup permits anonymous Viewer access; local admin credentials
are `admin` / `admin`. These defaults are for the local academic demo.

## Manual scenario switching during the presentation

In a second PowerShell terminal at the same repository root:

```powershell
.venv\Scripts\python.exe scripts/run_continuous_monitoring_demo.py set-scenario normal
.venv\Scripts\python.exe scripts/run_continuous_monitoring_demo.py set-scenario hvac_shift
.venv\Scripts\python.exe scripts/run_continuous_monitoring_demo.py set-scenario sensor_drift
.venv\Scripts\python.exe scripts/run_continuous_monitoring_demo.py set-scenario model_degradation
# Wait for at least two cycles to show suppressed duplicates and increasing trigger counts.
.venv\Scripts\python.exe scripts/run_continuous_monitoring_demo.py set-scenario recovery
.venv\Scripts\python.exe scripts/run_continuous_monitoring_demo.py status
docker compose exec monitor python scripts/run_continuous_monitoring_demo.py history --db /app/monitoring-store/predictions.db
```

Each command affects the next cycle. Wait for a completed cycle, then a Prometheus
scrape (15 seconds by default) and dashboard refresh. A running cycle finishes with
its original scenario. The state file persists across restarts; explicitly switch
to `normal` when starting a new presentation.

API mode reuses its HTTP connection across prediction/feedback requests. A failed
POST is not automatically retried, since it may already have persisted a record.
Full `status` output and batch provenance explicitly identify simulated feedback;
the short per-cycle log identifies the scenario but does not repeat the full
feedback-source description. Use `status` when explaining the synthetic targets.

Each printed JSON summary includes scenario, cycle, observations, paired count,
drift share/features, MAE/RMSE/R²/bias, performance status, active alerts, new alerts,
reminders, suppressed duplicates, and resolved alerts. Full evidence is in
`latest.json`, `cycles.jsonl`, each batch's `summary.json`/`provenance.json`, and SQLite.
Provenance includes the exact evaluation prediction IDs and explicit feedback labels.

## Native Python and automatic demonstrations

For a no-server demonstration that still writes collector-compatible SQLite data:

```powershell
.venv\Scripts\python.exe scripts/run_continuous_monitoring_demo.py run --automatic --hold-cycles 2 --cycles 20 --interval 10
```

This performs the five scenarios twice. Do not run it concurrently with the Compose
monitor against the same store. To run the automatic workflow against the Docker
API, use a one-off controller with access to its Linux database volume:

```powershell
docker compose stop monitor
docker compose run --rm --no-deps monitor python scripts/run_continuous_monitoring_demo.py run --api-url http://api:8000 --reference /app/data/train.csv --test /app/data/test.csv --db /app/monitoring-store/predictions.db --degradation-input sensor_drift --automatic --cycles 20 --interval 20
```

For native API serving instead of Docker, terminal A:

```powershell
$env:MONITORING_DB_PATH = Join-Path (Get-Location).Path 'reports\demo\continuous\predictions.db'
$env:ENERGY_MODEL_PATH = Join-Path (Get-Location).Path 'models\gradient_boosting.pkl'
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Terminal B runs the controller with `--api-url http://127.0.0.1:8000`. To connect
container Prometheus to a native Windows API, its target must be explicitly changed
to `host.docker.internal:8000` in a separate local configuration, and Uvicorn must
bind an interface reachable from Docker. The supplied Compose demo avoids this
extra networking step by keeping the API on the Compose network.

A native controller cannot directly open the Docker volume's SQLite file. Use the
one-off container above for that API, or use both the native API and native controller
with their shared host database. The store identity guard prevents accidental mixing.

All configurable options:

```powershell
.venv\Scripts\python.exe scripts/run_continuous_monitoring_demo.py run --help
```

Use `--reference`, `--test`, `--db`, `--output-dir`, `--state-file`, `--samples`,
`--seed`, `--severity`, `--degradation-input`, `--model-path`, `--api-url`, `--interval`, `--cooldown`,
`--reports`, `--cycles`, `--automatic`, and `--hold-cycles`. With custom directories,
pass the same state path to `set-scenario`. `--reports` uses the existing optional
Evidently installation for HTML; the slim Docker inference image does not bundle it.
The default fast loop uses the existing statistical detector and JSON summaries.

Graceful container stop:

```powershell
docker compose stop monitor
```

## Dashboard metrics and verification

Prometheus scrapes the actual API; its custom collector queries the shared SQLite
store on every scrape. No process-local drift gauges are assumed. Existing metric
names remain compatible. Four small panels add R², bias, drifted-feature count, and
last successful evaluation to the existing dashboard. Its provisioned datasource
now has the explicit `prometheus` UID already referenced by its panels.
Current-value panels use instant queries, so a missing latest measurement is not
silently replaced with an earlier non-null value from the displayed time range.

Useful queries:

```promql
energy_advisor_dataset_drift_share
energy_advisor_dataset_drift_detected
energy_advisor_drifted_features_count
energy_advisor_model_mae
energy_advisor_model_rmse
energy_advisor_model_r2
energy_advisor_model_bias_kwh
energy_advisor_performance_sample_count
energy_advisor_active_alerts_total
energy_advisor_active_alerts
energy_advisor_drift_detection_runs_total
energy_advisor_performance_evaluation_runs_total
energy_advisor_monitoring_execution_failures_total
energy_advisor_monitoring_last_success_timestamp_seconds
energy_advisor_monitoring_scenario_info
```

The last successful evaluation timestamp signals computation freshness, not that
the model is healthy. MAE/RMSE/R²/bias are omitted when the latest model assessment
has no metrics, rather than silently reusing earlier healthy values. Zero active
alerts is a measured database count; absent error/accuracy measurements are not
fabricated as zero. Scenario labels have five bounded values, never prediction IDs.

## Troubleshooting and limitations

- Docker pipe unavailable: start Docker Desktop and wait until `docker ps` succeeds.
- Port 8000/3000/9090 occupied: identify the service first; do not terminate an
  unrelated process. Use an explicit local port override if needed.
- Store mismatch: stop before sending predictions; align the API/controller bind
  mounts and `MONITORING_DB_PATH`. A remote Render store cannot be a local SQLite
  demo store. This controller intentionally refuses mismatched APIs before POST.
- No graphs yet: wait for one completed batch, then a scrape; check Prometheus UP,
  the datasource UID, API `/metrics`, and the last successful evaluation timestamp.
- Healthy batch unexpectedly degraded: inspect real measured metrics, sampling
  variability, sample size, and checkpoint provenance. Do not alter thresholds.
- Low samples/missing feedback: retain any existing alert; gather sufficient pairs.
- Restart: history is append-only, so the new session begins with prior active
  alerts until fresh evidence resolves them. UUID batch paths prevent overwrites.
- Storage grows with retained evidence; this college demonstration has no automatic
  retention deletion. SQLite and a single controller are local-demo constraints.
- This does not implement automatic retraining, model promotion, external alert
  delivery, or the other team members' full lifecycle dashboard/governance work.

## Test-store isolation

`tests/conftest.py` gives every test a temporary default `MONITORING_DB_PATH`;
individual tests can override it. Earlier API/metrics fixtures could otherwise
append test predictions to the normal development database. Existing records
were retained, and the final full-suite review verified the normal database's
hash and row counts did not change. Run tests from the repository root:

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

## Presentation walkthrough

1. Show `/ready`, Prometheus target UP, and the actual Grafana dashboard.
2. Explain reference data, historical feedback replay, and the separate demo store.
3. Show normal metrics with enough pairs and zero active alerts.
4. Switch to HVAC; explain the changed demand distributions and feature names.
5. Switch to sensor drift; explain energy measurement bias, not voltage channels.
6. Switch to degradation; point out unchanged predictions and simulated targets.
7. Hold the scenario for several cycles; show the same alert ID and higher trigger
   counts, with suppressed notifications rather than duplicate active alerts.
8. Switch to recovery; show fresh-batch metrics and retained resolved alert rows.
9. Repeat the sequence. Quote the measured results in the implementation report,
   and explain sampling/checkpoint/statistical limitations honestly.
