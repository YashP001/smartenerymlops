# Phase 3 Person 1 implementation and verification report

## Initial architecture audit

Audit date: 24 November 2026 (Asia/Kolkata). Starting branch: `main`.
Starting commit: `690976c` (`feat: expand MLflow demonstration artifact package`).
The working tree was clean before this work. No repository `AGENTS.md` was found.
The initial full suite passed: **308 passed, 13 warnings**, 187.93 seconds.

| Component inspected | Existing behavior | Gap addressed |
|---|---|---|
| `src/simulation.py` | Deterministic normal, afternoon HVAC, and derived energy sensor-bias scenarios; metadata and six canonical features | Reused unchanged; invoke again for each cycle with recorded seeds |
| `database.py` | Predictions and feedback joined by prediction ID; alerts and monitoring snapshots persisted in SQLite | Add exact batch provenance, store identity, and additive metadata migration |
| `drift_detector.py` | KS/effect-size checks, six-input drift share, separate target check, optional Evidently reports | Preserve thresholds; correct target-versus-concept terminology |
| `performance_monitor.py` | MAE/RMSE/R²/bias, minimum 30 pairs; version/time filters already exist | Add exact prediction-ID membership to prevent old batches contaminating recovery |
| `alert_manager.py` | Separate drift, accuracy, and execution alerts; recovery and deduplication | Fix cooldown tracking from notification time rather than every repeated detection |
| `monitor_orchestrator.py` | Runs detection/performance/alerts and persists results; CLI interval supported | Pass bounded batch/window metadata; avoid re-evaluating full history when saving a report |
| `metrics.py` | API collector dynamically queries SQLite; drift share, MAE/RMSE, counters, alert severity already exported | Add status/count, R², bias, sample count, timestamps, current scenario, total active alerts |
| `api/main.py` | `/predict` logs predictions asynchronously; `/feedback` joins actuals; `/metrics` exposes collector | Explicit optional local checkpoint path and opaque shared-store header; endpoint semantics preserved |
| `run_end_to_end_demo.py` | Existing isolated one-shot integration demonstration | Preserve unchanged; separate controller supports ongoing fresh batches and manual switching |
| `docker-compose.yml` | API, Prometheus, and Grafana only; regular monitoring store mounted | Preserve base file; opt-in demo override adds independent monitor and separate shared demo store |
| Prometheus config | Scrapes `api:8000/metrics` every 15 seconds | Reuse unchanged with actual running API |
| Grafana provisioning/panels | Existing operational dashboard references datasource UID `prometheus`; datasource did not explicitly set that UID | Set UID; add four small monitoring panels and align display thresholds with existing policy |
| Documentation/tests | Extensive existing simulation, drift, alert, feedback, metrics, and integration coverage | Reuse; add focused tests and an individual demo guide |

Before the change, production CSVs were passed manually to `run_drift_analysis`.
The interval loop reopened the configured CSV, so file contents could refresh if
another producer replaced it, but it did not generate fresh observations itself.
Predictions entered SQLite through the API background task; actuals entered through
feedback endpoints and paired by ID. Performance evaluation defaulted to all paired
history. The orchestrator's `--db` could differ from the API's environment setting;
there was no shared-store handshake. Prometheus collected API-exposed SQLite
snapshots, and Grafana queried Prometheus. Docker initially could not connect to
its engine. Docker Desktop was started successfully during this task; real stack
verification is recorded below rather than inferred from configuration alone.

## Implementation and reused functionality

The new independent controller reads an atomic scenario state file every cycle,
generates a new CSV through the existing simulator, creates prediction IDs through
existing inference/logging functions or the real API, ingests paired feedback, and
calls the existing orchestrator with that batch's IDs. Historical records remain
available. The evaluation bound is explicit batch membership, not an accumulating
window and not a reset database. A new UUID session prevents output overwrites.

Normal/HVAC/sensor input distributions and healthy held-out accuracy are separate
streams. In the degradation scenario, the held-out actuals and synthetic production
target gain `0.4 * severity` kWh; model predictions remain untouched. This explicitly
changes the simulated relationship and demonstrates degradation without claiming
that input drift alone proves it. All synthetic feedback is labeled as such.

The controller defaults to the existing Phase 1 train+validation pickle. The
registry's version 1 is a different train-only checkpoint. No model was retrained,
registered, promoted, or modified. Thresholds and canonical features are unchanged.

The live Docker profile selects `--degradation-input sensor_drift`, retaining the
input-drift condition while adding a separate performance alert. Both alerts then
resolve on the final recovery stage. The default native mode uses normal inputs
during degradation, demonstrating that input drift is not necessary for accuracy
degradation. Both cases have focused lifecycle coverage.

API mode checks an opaque database identity before any POST, then verifies actual
prediction persistence before sending feedback. A mismatched API is rejected.
The default development store and an existing non-demo prediction database are
rejected by the controller. Errors are execution failures, never invented drift.

Actual runtime testing exposed high per-record overhead with a Windows-mounted
SQLite file. The live profile therefore shares a dedicated Linux Docker volume for
SQLite and retains presenter controls/evidence on the Windows host. The earlier
trial database was preserved, not deleted. The controller reuses a standard-library
HTTP connection for prediction/feedback and never automatically retries a POST.

The collector reads the latest model assessment, omitting absent accuracy metrics
rather than retaining older healthy values. New scalar status/freshness values are
omitted before a real observation. Measured alert counts may correctly be zero.

## Exact source and documentation changes

New files:

- `src/monitoring/demo_controller.py`: fresh-batch ingestion, provenance, scenario
  switching, bounded assessments, and interruptible loop.
- `scripts/run_continuous_monitoring_demo.py`: CLI `run`, `set-scenario`, and `status`.
- `docker-compose.demo.yml`: optional isolated live presentation stack.
- `tests/test_continuous_monitoring_demo.py`: portable temporary-store tests.
- `tests/conftest.py`: final-review addition isolating the default monitoring store
  for all tests, including older API/metrics tests.
- `docs/continuous_monitoring_demo_guide.md`: complete startup and presentation guide.
- `docs/person1_implementation_report.md`: this audit and evidence report.

Modified files:

- `src/monitoring/database.py`: additive schema migration, store identity,
  provenance persistence, latest-assessment observability.
  The optional demo-only WAL/NORMAL setting avoids excessive Windows bind-mount
  fsync overhead; default FULL durability remains unchanged.
- `src/monitoring/performance_monitor.py`: optional exact batch membership filter.
- `src/monitoring/monitor_orchestrator.py`: propagate batch/time bounds and metadata;
  persist the same bounded performance assessment; configurable drift report output.
- `src/monitoring/alert_manager.py`: separate notification and detection clocks.
- `src/monitoring/metrics.py`: essential monitoring metric additions.
- `src/monitoring/drift_detector.py`: terminology clarification only.
- `api/main.py`: optional explicit model path and opaque shared-store metrics header.
- `monitoring/grafana/provisioning/datasources/prometheus.yml`: explicit UID.
- `monitoring/grafana/dashboards/smart-home-energy-advisor.json`: four monitoring
  panels and threshold display correction; existing panels retained.
- `README.md`: link to the guide and target-drift terminology correction.

Historical data, the model artifact, MLflow registry/artifacts, Render configuration,
the base Compose file, simulator, and existing one-shot demo were not edited.

## Observed real-model results

Native controller run: 500 observations/pairs per cycle, seed 42, severity 1.2,
two cycles per stage, 10 cycles total, unchanged local Phase 1 pickle. The following
are actual measurements from `reports/demo/person1-validation/cycles.jsonl`, not
invented guarantees. Bias is prediction minus actual, in kWh.

| Scenario (cycle) | Sample count | Input drift share | Drifted features | MAE | RMSE | R² | Bias | Alert status |
|---|---:|---:|---|---:|---:|---:|---:|---|
| Normal (1) | 500 | 0% | None | 0.0740 | 0.1518 | 0.4021 | -0.0031 | None active |
| HVAC shift (3) | 500 | 50% | Current hour, previous hour, two hours ago | 0.0717 | 0.1425 | 0.3372 | +0.0108 | Data drift created |
| Sensor drift (5) | 500 | 50% | Current hour, previous hour, two hours ago | 0.0691 | 0.1555 | 0.3423 | -0.0021 | Existing drift alert reused; notification suppressed |
| Model degradation (7) | 500 | 0% | None | 0.4843 | 0.5162 | -4.6795 | -0.4839 | Separate model alert created; input drift resolved |
| Repeated degradation (8) | 500 | 0% | None | 0.4773 | 0.5052 | -5.3862 | -0.4773 | Same model alert reused; notification suppressed |
| Recovery (9) | 500 | 0% | None | 0.0824 | 0.1695 | 0.3832 | -0.0012 | Model alert resolved; zero active |

The HVAC/sensor MAE values measure the separate genuine held-out replay. They
must not be described as accuracy on shifted synthetic inputs. Degraded actuals
are simulated. Normal held-out actuals are historical measurements. Marginal target
drift is analyzed independently and excluded from input drift share.

## Alert lifecycle, deduplication, and recovery evidence

The first native run created one drift alert, reused it through HVAC/sensor cycles,
resolved it when input distributions normalized in the degradation stage, created
a distinct performance alert, suppressed its repeat, and resolved it on fresh
healthy feedback. No alerts or historical feedback were deleted.

That table describes the isolated native configuration. In the live combined
configuration, input drift deliberately remains active during performance degradation,
so recovery must resolve both distinct alert types, not just the model alert.

Portable lifecycle tests repeat the entire sequence twice and verify persistent
resolved rows, trigger counts, exact feedback pairing, unique fresh batch files,
unchanged historical/model files, and exclusion of old degraded feedback. A clock
test verifies a reminder is emitted after cooldown even when detections continued
during that cooldown. Missing feedback and insufficient samples retain active
degradation alerts. Execution failures are a distinct category with own recovery.

## Runtime dashboard verification

The real local API, monitor, Prometheus, and Grafana are running. `/ready` returns
200, Prometheus reports the `api:8000/metrics` target UP, and Grafana's provisioned
datasource UID is `prometheus`. The runtime verifier queries both Prometheus and
the actual Grafana datasource proxy, not just the API's text exposition.

Two complete live sequences have passed. Grafana's rendered dashboard also
showed 50% drift, MAE 0.4843, RMSE 0.5162, and two distinct CRITICAL alert types.
The new panels visibly showed R² -5.39, bias -0.477 kWh, three drifted features,
and the latest successful evaluation timestamp during repeated degradation.
Current-value panels now use instant queries to avoid retaining historical
non-null measurements when current metrics are missing.

Actual first live repetition (500 pairs per stage; retained sensor drift during
degradation):

| Scenario (cycle) | Sample count | Drift share | Drifted features | MAE | RMSE | R² | Bias | Alert status |
|---|---:|---:|---|---:|---:|---:|---:|---|
| Normal (3) | 500 | 0% | None | 0.0717 | 0.1425 | 0.3372 | +0.0108 | Zero active |
| HVAC shift (4) | 500 | 50% | Current hour, previous hour, two hours ago | 0.0700 | 0.1405 | 0.3971 | +0.0008 | One data alert |
| Sensor drift (5) | 500 | 50% | Current hour, previous hour, two hours ago | 0.0691 | 0.1555 | 0.3423 | -0.0021 | Same data alert; suppressed notification |
| Model degradation (7) | 500 | 50% | Current hour, previous hour, two hours ago | 0.4843 | 0.5162 | -4.6795 | -0.4839 | Two separate active alerts |
| Repeated degradation (8) | 500 | 50% | Current hour, previous hour, two hours ago | 0.4773 | 0.5052 | -5.3862 | -0.4773 | Both alerts reused; notifications suppressed |
| Recovery (9) | 500 | 0% | None | 0.0824 | 0.1695 | 0.3832 | -0.0012 | Both resolved; zero active |

The second repetition also verified normal → HVAC → sensor → degradation → repeated
degradation → recovery. Its degradation MAE was 0.4834 kWh; repeated degradation
was 0.4734 kWh; recovery MAE was 0.0813 kWh, RMSE 0.1679 kWh, R² 0.2994, and
bias -0.0062 kWh. Input drift returned to 0% and both alerts resolved again.
Across both repetitions, the same alert IDs were retained during each ongoing
incident and trigger counts increased; no duplicate active incidents were created.
All 12 verified stage snapshots contained 500 paired records.

Evidence is captured under
`reports/demo/person1-audit/live-dashboard-evidence.json` and
`live-dashboard-verification.log`. Prometheus/Grafana may trail the latest
controller batch by a scrape interval; the verifier checks scenario transitions
and actual values from both services, allowing that normal sampling delay.

After the second recovery, graceful shutdown completed the in-flight healthy
cycle 18 and exited with code **0**. The final rendered dashboard showed **0% drift,
MAE 0.0644, RMSE 0.1405, R² 0.4002, bias +0.0060, zero drifted features, and zero
active alerts**. The native browser view was inspected, not just configuration/API
responses. The API remains ready and the Prometheus target remains UP after the
monitor stops; Grafana continues displaying persisted monitoring results.

The dedicated Docker store retains **14,500 predictions, 14,500 feedback records,
31 monitoring runs, and 10 resolved alerts** across trial and verification sessions.
Two historical execution-failure incidents from runtime debugging were retained
and resolved; their cumulative counters are intentionally not reset to zero.
The final verification session completed 18 fresh batches. Monitoring was stopped
after verification to avoid indefinite background ingestion; the three dashboard
services remain running.

## Test results

- Initial complete suite: **308 passed**, 13 warnings, 187.93 seconds.
- Existing focused monitoring/alert/performance/metrics regression suite:
  **105 passed**, 11 warnings, 39.51 seconds.
- First new focused suite: **10 passed**, 11 warnings, 146.51 seconds.
- Full regression suite with the normal FULL durability default before the final
  transport test: **322 passed**, 13 warnings, 391.24 seconds.
- Final complete suite on the final source, using the demo's optional NORMAL
  durability configuration: **323 passed**, 13 warnings, 428.03 seconds.
  The durability test explicitly unsets the option and verifies FULL remains the
  default, then verifies the opt-in setting. All 15 new portable tests passed.
- Focused closed-idle-connection/no-POST-retry regression: **1 passed**,
  14 deselected, 11 warnings, 20.08 seconds.

Final full-suite command (same test suite as requested, with the demo option):

```powershell
$env:MONITORING_SQLITE_SYNCHRONOUS = 'NORMAL'
.venv\Scripts\python.exe -m pytest tests/ -v
Remove-Item Env:MONITORING_SQLITE_SYNCHRONOUS
```

Warnings concern existing scientific/library behavior (including KS fallback),
not failed tests. Full logs are retained under `reports/demo/person1-audit/`.

Tests use temporary databases/directories and a fixed analytic predictor without
training or external services. The new tests have no Windows-specific path or
MLflow/Render dependency. The existing opt-in local MLflow checks remain unchanged.

## Live commands

See the [guide](continuous_monitoring_demo_guide.md) for all Windows commands and
troubleshooting. Core startup:

```powershell
docker compose -f docker-compose.yml -f docker-compose.demo.yml --profile continuous-monitoring up --build -d
.venv\Scripts\python.exe scripts/run_continuous_monitoring_demo.py set-scenario normal
.venv\Scripts\python.exe scripts/run_continuous_monitoring_demo.py set-scenario hvac_shift
.venv\Scripts\python.exe scripts/run_continuous_monitoring_demo.py set-scenario sensor_drift
.venv\Scripts\python.exe scripts/run_continuous_monitoring_demo.py set-scenario model_degradation
# Wait for repeated completed cycles.
.venv\Scripts\python.exe scripts/run_continuous_monitoring_demo.py set-scenario recovery
```

## Limitations and acceptance status

Statistical thresholds are unchanged heuristics. Standard KS independence assumptions
do not fully hold for autocorrelated smart-meter series. Synthetic hourly samples
are conditional resampling, not a physical continuous-household simulator. The
minimum 30-pair rule does not guarantee statistical independence. Accuracy can vary
with batch size/seed; the controller never forces a healthy label. A single controller
and SQLite store suit the local academic demo, not distributed production service.
The slim Docker runtime omits optional Evidently HTML rendering; native `--reports`
uses it when installed. History is retained, so storage grows without automatic
retention deletion. External notification delivery/retraining are outside scope.

The existing Render deployment was supplied by the user; no remote prediction or
feedback writes were made. This local demonstration does not verify that cloud
deployment or share its SQLite store.

Acceptance criterion: **SATISFIED for the local live demonstration.** The complete
normal → drift → alert → model degradation → repeated monitoring → recovery
sequence passed twice through the actual API, Prometheus, Grafana datasource,
and rendered dashboard. Both distinct alerts resolved on fresh healthy batches,
history remained intact, and graceful termination was verified. This statement
does not claim cloud/Render monitoring verification or full production readiness.

## Final Git and running state

- Branch and HEAD remain `main` / `690976c`.
- Ten tracked files modified; seven new files untracked, listed above (the original
  16-file implementation plus the final-review test-isolation fixture).
- Index is empty: **nothing staged, committed, or pushed** during this task.
- Protected historical datasets, model artifact, MLflow registration/artifacts,
  base Compose configuration, simulator, one-shot demo, and Render configuration
  were not changed. Git comparisons confirmed tracked data/model files unchanged.
- API, Prometheus, and Grafana are running. The monitor is stopped with exit code 0.

To resume fresh monitoring after reviewing the changes:

```powershell
.venv\Scripts\python.exe scripts/run_continuous_monitoring_demo.py set-scenario normal
docker compose -f docker-compose.yml -f docker-compose.demo.yml up -d --no-deps monitor
```

For a fresh build/start, use the full command in the guide. Existing alert and
prediction history is preserved on restart.

## Final implementation and integration review

Reviewed all 16 original pending files against `690976c` on 24 November 2026. The
remote `main` still points to the same commit. This review made documentation
corrections and added one necessary test-isolation fixture, making 17 files total.
Application code, deployment configuration, models, and historical datasets were
not changed again. The commit recommendation below applies to the local academic
demonstration, not unrestricted production deployment.

### 1. Complete Git diff

All ten tracked-file diffs and the original six new files were reviewed, followed
by the new `tests/conftest.py` fixture. The index is empty.
`git diff --check` passes; Git's LF/CRLF conversion notices are not whitespace errors.

### 2. Scope, duplication, paths, credentials, and generated files

The controller reuses existing simulation, prediction, feedback, detection,
performance, alert, and persistence code. Its separate loop supplies fresh batches
and manual controls absent from the one-shot demo. No debugger calls, new secrets,
machine-specific runtime Windows paths, or new project dependencies were found.
Container `/app/...` paths and the documented Windows checkout example are intentional.
Grafana's local `admin` defaults predate this change. Runtime databases, CSVs,
logs, and evidence are ignored; none are among the 17 recommended commit files.

### 3. Demonstration data isolation

Expanded Compose configuration confirms the demo API replaces the normal host
database bind mount with the dedicated `monitoring_demo_data` volume. The monitor
opens the same volume at a different container path and verifies its opaque identity
before POST requests. Evidence/state files stay under ignored `reports/demo/`;
reference/test CSV mounts are read only. The normal database and historical data
remain intact. This is data isolation, not two concurrent application stacks:
service names, ports, and observability volumes are shared between modes.

### 4. Manual scenario switching

`set-scenario` atomically replaces the shared state file. The active controller
reads it before each cycle; scenario changes take effect at the next cycle.
The CLI accepts all five documented scenario names. Existing state persists, so
the guide now resets it to `normal` before startup. Automatic mode intentionally
follows its configured sequence instead of the manual state file.

### 5. Fresh batches while active

Each cycle calls the simulator with an incremented seed, writes a unique batch CSV,
and creates new prediction IDs. The reference/test data are historical replay;
fresh batches do not imply newly measured homes or disjoint held-out rows across
batches. The interval is a delay after processing, not a fixed start-to-start rate.

### 6. Recovery and history preservation

Performance evaluation receives the current batch's exact prediction IDs, combined
with the model filter. Old degraded pairs are excluded, not deleted. Saved live
evidence was rechecked for both repetitions: ongoing alert IDs match, trigger counts
increase, and final healthy feedback resolves both alerts. Historical rows remain.

### 7. Insufficient feedback protection

The alert manager returns without resolving a degradation alert for
`insufficient_data` or `no_feedback_data`. The new test verifies an existing active
alert survives an empty batch and a ten-record batch before healthy recovery.

### 8. Drift versus measured accuracy

HVAC/sensor drift metrics describe the simulator population. Their healthy accuracy
metrics describe a separate held-out feature/actual replay. The guide and comparison
tables explicitly distinguish them. They must not be presented as proof of model
accuracy or degradation on shifted HVAC/sensor inputs.

### 9. Synthetic degradation labels

Degraded actuals are held-out targets plus a controlled offset, with predictions
unchanged. `feedback_source` explicitly says `SIMULATED` in full summaries and
provenance. The short cycle log names the scenario; presenters should show the
`status` output when explaining synthetic ground truth. No synthetic values are
written back to historical datasets.

### 10. Loaded checkpoint and baseline

The running API is configured for `/app/models/gradient_boosting.pkl`. Its SHA-256
and the host artifact's SHA-256 both equal
`6783a9775b66cdb696d3b71b27693572e5a40502b8a8c2e97595475894cb2cf5`.
Read-only evaluation of all 4,120 test rows reproduced MAE **0.0738930282**, RMSE
**0.1511951130**, R² **0.3723265161**, and bias **+0.0015340369**. These match the
rounded monitoring baseline. The artifact has `max_depth=3`; an older README
description saying `5` was corrected without retraining. Custom model/API overrides
do not automatically select a matching baseline or expose a verifiable model ID.

### 11. FastAPI, MLflow, CI/CD, and Render compatibility

Existing endpoint request/response bodies remain intact; `/metrics` adds an opaque
header. With `ENERGY_MODEL_PATH` unset, original registry/fallback loading remains.
Schema migration was independently tested against a database created by the prior
Git version: prediction, feedback, and monitoring rows survived two migrations,
and the store ID stayed stable. MLflow code/artifacts, dependency files, workflow,
Dockerfile, and Render configuration are unchanged. Remote deployment of the new
uncommitted code and a new GitHub Actions execution have not been performed.

### 12. Docker compatibility and mode switching

The demo is opt-in. The base configuration retains its normal database mount and
model-loading behavior. Both modes use the same services and published ports;
the guide now states this explicitly and includes commands to restore normal mode
without removing volumes. Prometheus/Grafana history can contain both modes under
existing labels. Build the API image with the combined configuration before starting
the monitor, which reuses that image with `pull_policy: never`.

The review found that older API and metrics tests did not override the default
monitoring path. The normal development store now contains 522 predictions and
70 feedback records, with its latest writes aligned to earlier full-suite runs.
Those runs were therefore not completely isolated, despite the new demo tests
using temporary stores. No records were deleted or rewritten. The new autouse
fixture sets a unique temporary default path per test and permits tests that
explicitly override/test environment restoration to continue doing so. A full
regression run passed **323 tests, 13 warnings, in 223.12 seconds**. The normal
store's SHA-256 hash and all four table counts were identical before and after:
522 predictions, 70 feedback rows, zero alerts, and zero monitoring runs. This
prevents further test writes; it does not remove prior test traffic from the store.

### 13. Automated test portability

The 15 new tests passed again on Windows: **15 passed, 11 warnings, 137.63 seconds**.
They also passed in a disposable Linux/Python 3.13 container with a read-only
repository mount: **15 passed, 3,609 warnings, 134.32 seconds**. Databases and test
artifacts were temporary; the HTTP transport test used an ephemeral loopback port.
No real MLflow database, Render service, or credentials were required.

After adding the default-store fixture, the existing API and Prometheus metrics
tests also passed in Linux: **51 passed, 134 warnings, 13.49 seconds**. This used
a writable temporary MLflow backend as well as temporary monitoring stores; the
source checkout stayed read only. An earlier audit attempt without relocating
MLflow was stopped because its SQLite initialization retried against the read-only
checkout; that incomplete attempt is not counted as passing validation.

Warning investigation found repeated existing `Timestamp.dayofweek` deprecations
in the unchanged simulator under pandas 3.1.1, plus joblib/NumPy 2.5 shape-setting
deprecations. They did not fail tests. The inference requirements permit these
newer versions, whereas the full CI requirements pin pandas/NumPy. Person 4 should
consider dependency locking separately; no dependency versions were changed merely
to suppress warnings. The prior complete final-source run remains **323 passed**.

### 14. Dependencies for the other team members

- **Person 2:** the demo intentionally selects the Phase 1 pickle and fixed
  baseline. True registry version/run IDs and per-checkpoint baselines must be
  integrated together; a store ID is not model identity. Current model labels
  remain class names. Do not silently promote a different artifact under this baseline.
- **Person 3:** retain datasource UID `prometheus`, current-value instant queries,
  shared-store collection, freshness timestamps, and the actual metric names in
  the guide. Distinguish missing data from zero and cumulative execution failures
  from active incidents. The orchestrator's saved reports now use cycle-specific
  filenames; its separate performance-report CLI retains its original filename.
- **Person 4:** own mode switching, image builds, database-volume persistence,
  ports/service-name conflicts, dependency locking, and cloud integration. Keep
  NORMAL durability opt-in for the demo. The slim image does not include optional
  Evidently HTML rendering. Cloud monitoring still needs a compatible shared store
  and deployment design; local verification does not establish cloud readiness.

### 15. Documentation accuracy and remaining limitations

The guide/report distinguish native isolated degradation from the Docker combined
drift/degradation scenario, document measured rather than guaranteed metrics, and
record the stopped monitor and retained running dashboard services. This review
clarified model-hash availability, corrected README model depth and feedback
provenance, and added mode restoration/startup state instructions.

Remaining bounded limitations: one controller should own a demo store; no automatic
retention, external notifications, or retraining is added. For finite automatic
runs, errors are logged and alerted but the loop does not aggregate failures into
a nonzero final exit code; inspect cycle outcomes rather than exit code alone.
The controller initializes schema before checking ownership of an arbitrary custom
database, so a refused non-demo custom database can receive additive schema metadata
even though no demo predictions are appended. Use dedicated database paths; the
known normal default path is rejected before initialization.

**Review recommendation:** the isolation regression check passed; the 17 files
are safe to commit together for the documented local demonstration. No critical
commit blocker remains. Do not include ignored
runtime evidence or generated data, and do not treat this as production/cloud
approval. No files were staged, committed, or pushed during the review.
