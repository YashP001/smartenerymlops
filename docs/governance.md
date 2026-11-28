# Governance: Smart Home Energy Advisor

This document states who is accountable for the system, which rules apply to its
data, models, monitoring and deployment, **why** each rule exists, and how each rule is
enforced today. Every control is labelled honestly:

- **Code**: the repository enforces it automatically (a test, a check, a refusal).
- **Process**: the team enforces it by following the procedure; nothing stops a violation.
- **Gap**: not implemented. Listed in section 11 with the risk it leaves.

Scope: a student MLOps project that forecasts a household's next-hour electricity use
(kWh) from six numeric features. It is a decision-support demo, not a safety-critical system.

---

## 1. Principles

| Principle | Why it matters for this system |
|---|---|
| **Reproducibility** | A prediction that cannot be traced to a model, a dataset and a code revision cannot be audited or fixed. |
| **Human decision, machine evidence** | Drift and degradation alerts are noisy (feedback labels arrive late, small samples). Automation detects; a named person decides. |
| **Least data** | The model needs six numeric features. Holding more (meter IDs, timestamps tied to a household) adds privacy risk and no accuracy. |
| **Evidence before promotion** | A new model version must earn deployment on a locked test set. Newer is not better. |
| **Be honest about limits** | A governance document that overstates its controls is worse than none. |

---

## 2. Roles and owners

One person may hold several roles; the approver of a model change must not be the person who proposed it.

| Role | Responsibility | Workstream owner |
|---|---|---|
| **Governance & Deployment Owner** | This document; environment, access, incident coordination, final rollback authority | Janvi S Hegde |
| **Model Owner** | Training, registry, version comparison, promote/rollback execution, model card | Simran S Patil |
| **Monitoring Owner** | Drift/performance thresholds, alert behaviour, monitor worker health | Yash Purwar |
| **Dashboard Owner** | Grafana dashboard accuracy, stale/missing-data display | Yash Balotia |
| **Model Approver** | Approves promotion. Any team member who is not the proposer | Rotating |
---

## 3. Data governance

| Rule | Why | Enforced by |
|---|---|---|
| Source data is the CEEW smart-meter dataset (Mathura 2021, 35 meters; plus the supplied aggregated file). Provenance and schema are documented in `data/README.md` and `reports/phase1_report.md`. The dataset's own terms of use apply, and the Phase 1 acquisition/approval record should be linked here: **https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GOCHJH**. | Provenance and permitted use | Process |
| Raw data is versioned with **DVC** (MD5 pointers in `data/raw/*.dvc`, remote in Google Drive); Git holds only pointers. | Exact dataset identity without bloating Git | Code (`.dvc/`) |
| Data passes validation (schema, ranges, gaps, outliers) before modelling. | Garbage in, garbage out; documents known defects (261 timestamp gaps, voltage and frequency outliers) | Code (`src/data_validation.py`, `tests/test_data_validation.py`) |
| **Chronological** train/validation/test splits. | Random splits leak the future into training and overstate accuracy for a forecasting task | Code (`src/split_data.py`) |
| `data/test.csv` is a **locked final holdout**: used once to compare finished candidates, never to tune. | A test set used for tuning stops measuring generalisation | Process, evidenced in `reports/model_comparison_v1_v2.md` (SHA-256 of both datasets recorded) |
| Training and evaluation file hashes are written into every comparison run. | Detects a silently changed dataset | Code (`scripts/train_and_compare_model_versions.py`) |

---

## 4. Privacy

The system follows data minimisation: the prediction API requires six numeric features and does not need direct household identifiers or user-submitted free text.

| Rule | Why | Enforced by |
|---|---|---|
| The prediction API schema defines six numeric features, validated for type and range. The inference validation function returns a cleaned dictionary containing only the canonical model features. | Data minimisation: only the inputs required for prediction should reach the model and monitoring store. | Code (`api/main.py`, `src/features.py::validate_inference_input`) |
| Prediction logging explicitly stores the six feature values, predicted energy, prediction timestamp, model version, MLflow run ID and inference latency. The complete request payload is not intentionally persisted. | Prevents unnecessary request data from entering the monitoring database. | Code (`src/monitoring/database.py::log_prediction`) |
| Feedback accepts an actual energy reading associated with an existing prediction. Non-numeric values, Boolean values, NaN, infinity and negative readings are rejected. Duplicate feedback is also rejected. | Prevents invalid or duplicate ground-truth records from affecting performance monitoring. | Code (`src/monitoring/database.py::log_feedback`) |
| The feedback table stores only the prediction ID, actual energy consumption and feedback timestamp. | Limits the amount of information retained with each feedback record. | Code (schema and insert logic in `src/monitoring/database.py`) |
| No user-submitted free text or direct household identifiers are intentionally collected through the prediction and feedback endpoints. System-generated alert descriptions, monitoring details and operational error messages may be stored. | Avoids collecting unnecessary household information while retaining operational evidence for debugging and monitoring. | Code (API schemas and monitoring database functions) |
| Raw meter-level data containing a meter identifier remains in DVC storage and is not intentionally copied into container images, the monitoring store or the model registry. | Raw data is the most sensitive project artefact. | Code (`.dockerignore`, `docker-compose.yml`, DVC configuration and data mounts) |
| Secrets (`.env`, DVC credentials such as `config.local`) are excluded from Git and container build contexts. | Credentials in Git or images can become publicly accessible. | Code (`.gitignore`, `.dockerignore`, `tests/test_deployment_config.py`) |
| MLflow usage telemetry is disabled through `MLFLOW_DISABLE_TELEMETRY` and `DO_NOT_TRACK`. | Prevents unnecessary usage telemetry from being sent to third parties. | Code (`docker-compose.yml`) |
| Simulated production data is generated from held-out test rows rather than collected from real households. | The demonstration does not require live personal smart-meter data. | Code (`src/simulation.py`) |

---

## 5. Model governance

### Why model versioning exists

A model is the product of **code + data + parameters**. Changing any of these can change its predictions.

Model versioning gives every registered model an identifiable version and MLflow run ID. This allows the team to:

1. **Trace:** Each prediction records `model_version` and `model_run_id`, allowing the team to identify the model that generated a prediction.
2. **Compare:** Candidate models are evaluated using the same locked test dataset, making performance differences measurable.
3. **Roll back:** Previously registered model versions remain available for restoration without retraining.
4. **Audit:** The team can identify which registered model version produced a prediction.

### Versions in this project

| Version | Model | Configuration | Purpose |
|---|---|---|---|
| **v1** | GradientBoostingRegressor | 100 estimators, max depth 3, learning rate 0.1, random state 42 | Original baseline model and default deployed version |
| **v2** | RandomForestRegressor | 100 estimators, random state 42 | Alternative architecture evaluated as a challenger |

On the shared holdout, v2 is slightly worse than v1 on MAE, RMSE and R², as documented in `reports/model_comparison_v1_v2.md`.

**Decision:** The available evaluation evidence does not support promoting v2. Therefore, v1 remains deployed and v2 is retained as a registered candidate. The promotion and rollback commands can be demonstrated as lifecycle drills; demonstrating the mechanism does not imply that v2 is recommended for deployment.

### Approval criteria for promotion

A candidate may be promoted only when all the following conditions are satisfied:

1. It is trained using a recorded code revision and an identified training dataset. Dataset hashes and training-code identity should be available in the MLflow lineage.
2. It is evaluated against the same locked test dataset used for the deployed model. The test dataset must not be used for hyperparameter tuning.
3. It satisfies the project's performance limits:
   - MAE must not exceed the baseline by more than 25%.
   - RMSE must not exceed the baseline by more than 25%.
   - R² must be at least 0.10.
   - Absolute mean prediction bias must not exceed 0.05 kWh.
4. It improves on the deployed version's MAE, or a written justification for accepting a tie is included in the pull request.
5. CI checks pass and a second team member, acting as Model Approver, reviews the comparison report before promotion.

**Why these criteria:** The locked holdout helps prevent evaluation leakage. Performance limits prevent deploying a candidate that would immediately violate the monitoring policy. Comparing against the deployed model avoids unnecessary model changes. Independent review makes promotion deliberate and accountable.

**Enforcement:**
- **Code:** MLflow records model versions, run identities, parameters and dataset/code lineage where available. The comparison workflow evaluates both versions on the shared holdout and records the results. The registry checks that a candidate artifact can be loaded before selection.
- **Process:** Clean-revision requirements, performance acceptance, CI review and second-person approval are governance requirements. They are not all automatically enforced by the promotion command.

### Model registration, selection and promotion

The project provides an explicit model lifecycle through `src/model_registry.py`.

The available operations are:

```bash
docker compose run --rm --no-deps registry-init python -m src.model_registry status
docker compose run --rm --no-deps registry-init python -m src.model_registry select --version 2
docker compose run --rm --no-deps registry-init python -m src.model_registry promote
docker compose run --rm --no-deps registry-init python -m src.model_registry rollback
```

The operations have distinct purposes:

- **Status:** Displays the selected, deployed and previous model versions.
- **Select:** Selects a candidate version after verifying that its registered artifact can be loaded. Selection alone does not change the deployed version.
- **Promote:** Updates the deployed-version pointer to the selected or explicitly specified version.
- **Rollback:** Restores the previously deployed version, provided a previous version is recorded and its artifact can be loaded.

Promotion updates the deployment state in `deployment.json`, stored in the shared `mlflow_store` volume. Registered model artifacts are retained rather than deleted or overwritten.

The `MODEL_VERSION` environment variable overrides the deployment file's deployed-version pointer. It should be used only for a controlled operational reason because it bypasses the normal deployment pointer.

**Audit limitation:** The current deployment state records the selected, deployed and previous versions but does not automatically record the identity of the person performing a promotion or rollback, or the timestamp and reason for that action. Until this is implemented, the team must record these details in the pull request or incident log.

---

## 6. Monitoring governance

The monitoring system provides early warnings about changes in input data and model performance. Monitoring thresholds are governed configuration: changes must be reviewed, justified and documented through the change-management procedure in Section 9.

### Monitoring thresholds

| Signal | Threshold | Meaning |
|---|---|---|
| Feature drift — KS test | p-value < 0.05 and KS statistic ≥ 0.15 | Indicates a statistically and practically meaningful distribution change in a feature. |
| Feature drift — effect size | Absolute Cohen's d ≥ 0.20 with p-value < 0.05 | Identifies a meaningful feature distribution shift. |
| Dataset drift | At least 50% of monitored features drifted | Indicates a broader input-data shift rather than an isolated feature change. |
| MAE degradation | MAE more than 25% above baseline | Indicates a deterioration in prediction error. |
| RMSE degradation | RMSE more than 25% above baseline | Indicates a deterioration in prediction error, with greater sensitivity to large errors. |
| R² degradation | R² < 0.10 | Indicates insufficient explanatory performance under the project's monitoring policy. |
| Prediction bias | Absolute mean bias > 0.05 kWh | Indicates unacceptable systematic overprediction or underprediction. |
| Minimum evidence | At least 30 paired prediction/actual records, as specified by the monitoring policy | Avoids making a performance decision using an insufficient sample. |
| Alert cooldown | 3600 seconds (one hour) | Prevents repeated notifications for the same continuing alert condition. |
| Monitor freshness | Latest monitoring output must be less than 180 seconds old | Makes a stalled monitoring worker detectable. |

The baseline performance values used by the monitoring system are:

- MAE: 0.0739 kWh
- RMSE: 0.1512 kWh
- R²: 0.3723
- Baseline bias: approximately +0.00153 kWh

### Distinguishing drift from performance degradation

**Data drift is not the same as model performance degradation.**

- Data drift indicates that the input feature distributions have changed relative to the reference data.
- Performance degradation indicates that prediction quality has deteriorated when actual energy readings are available for comparison.

A drift alert alone does not establish that the model has failed. It should trigger investigation of possible causes such as sensor faults, seasonal changes or changes in household energy usage.

Performance degradation requires paired predictions and actual energy readings. The monitoring database stores feedback against prediction IDs so that performance can be evaluated when actual values become available.

### Alert severity

The monitoring system distinguishes between detecting degradation and assigning an alert severity.

The performance monitor uses the 25% MAE/RMSE increase, R² < 0.10 and absolute bias > 0.05 kWh thresholds to identify performance concerns.

The alert manager applies a separate severity rule:

- **CRITICAL:** MAE or RMSE increase reaches at least 50% over baseline, or R² falls below 0.0.
- **WARNING:** A degradation condition is detected but does not meet the CRITICAL severity condition.

The alert manager also applies a cooldown of 3600 seconds between repeat notifications.

Therefore, the 25% degradation threshold must not be interpreted as the CRITICAL alert threshold. A performance issue may be detected and classified as WARNING before it reaches CRITICAL severity.

### Monitoring worker health

The monitoring worker has a Docker health check based on the freshness of its latest output file:

`/app/reports/demo/continuous/latest.json`

The health check fails when the output is missing or older than 180 seconds. This makes a stalled monitoring process visible rather than allowing the absence of alerts to be mistaken for healthy operation.

**Enforcement:**
- **Code:** Drift thresholds, performance thresholds, alert cooldown and worker health checks are implemented in the monitoring modules and Docker Compose configuration.
- **Process:** Reviewing drift alerts, deciding whether retraining is necessary, approving threshold changes and deciding whether rollback is justified are team responsibilities.

**Operational rule:** Drift without confirmed performance degradation requires investigation, not automatic rollback. A confirmed degradation event requires review of the evidence and a human decision about rollback or retraining.


---

## 7. Access control

| Asset | Who may do what | Mechanism | Enforced by |
|---|---|---|---|
| Git `main` | Changes should follow the reviewed pull-request workflow | GitHub Actions CI runs on pushes and pull requests targeting `main`; branch protection is not verified as enabled | Process / Gap |
| Model promotion/rollback | Model Owner executes; Model Approver reviews the decision | Requires shell access to the Docker host and access to the shared MLflow volume | Process |
| Grafana dashboards | Anyone with network access may view; anonymous users have Viewer permissions | Anonymous role = Viewer; signup disabled | Code (`docker-compose.yml`) |
| Grafana admin account | Governance Owner manages administrative access | `GRAFANA_ADMIN_PASSWORD` in `.env`; Compose falls back to `admin` if the variable is not set | Process; default-password risk |
| Prediction API | Currently no authentication or rate limit | Published on the configured API host port, default `8000` | Gap |
| Monitoring database | Accessed by the API and monitoring containers through the shared named volume | `monitoring_store`; no direct database host-port mapping | Code (`docker-compose.yml`) |
| MLflow tracking database and artifacts | Accessed by registry-init, API and MLflow UI containers through the shared named volume | `mlflow_store`; no direct database or artifact-directory host-port mapping | Code (`docker-compose.yml`) |
| MLflow UI | Accessible to users with network access to the host | Published on the configured host port, default `5000` (currently mapped to `5001`) | Gap: no authentication configured in this Compose file |
| Prometheus | Accessible to users with network access to the host | Published on the configured host port, default `9090` | Gap: no authentication configured in this Compose file |
| Containers | Run as an unprivileged user | Dockerfile creates `appuser` with UID `10001`; API and monitor runtime identities were verified | Code and runtime verification |

### Network exposure

The Compose port mappings publish the API, Grafana, MLflow UI and Prometheus on all host interfaces rather than restricting access to localhost.

This is acceptable for a controlled classroom demonstration on a trusted network. It is not an appropriate production exposure configuration without network restrictions, authentication and appropriate access controls.

### Access-control limitations

- The prediction API does not implement authentication or rate limiting.
- Grafana anonymous access is read-only, but its administrative password must be changed from the Compose fallback before use on an untrusted network.
- MLflow UI and Prometheus are published without authentication configured in this Compose file.
- Docker named volumes prevent direct host-port access to the underlying SQLite databases and artifacts, but do not make the associated web interfaces private.

These limitations are recorded in Section 11.
---

## 8. Retention

| Data | Retention | Enforced by |
|---|---|---|
| Registered model versions and MLflow runs | Kept indefinitely (small; they are the audit trail) | Process |
| Monitoring DB (predictions, feedback, alert history) | Kept for the life of the demo volume. Reset with `docker compose down -v`. **No automatic deletion exists** | **Gap** |
| Prometheus metrics | 15 days (set explicitly in compose) | Code |
| Evidence reports (`reports/demo/`) | Git-ignored, local only, regenerable | Code (`.gitignore`) |
| Render deployment | Ephemeral: the SQLite DB is lost on every restart or redeploy | Platform behaviour, see runbook |

**Why:** these are numeric, non-identifying records, so long retention carries low privacy
risk, but unbounded growth degrades SQLite and Prometheus. A production policy would add a
scheduled purge (for example, predictions older than 90 days); that is a recorded gap, not a
silent omission.

---

## 9. Change management and audit trail

### Change path

The team's intended change-management path is:

**Branch → Pull Request → CI → Review → Merge to `main`**

The GitHub Actions workflow (`.github/workflows/ci.yml`) implements the following checks:

1. Runs on pushes and pull requests targeting `main`.
2. Installs Python 3.13 and project dependencies.
3. Executes the automated test suite using `pytest`.
4. Validates the Docker Compose configuration.
5. Builds the API Docker image.
6. Builds the monitoring/tools Docker image.
7. Inspects the resulting API image.

The Docker build job depends on successful completion of the test job.

**Enforcement limitation:** The workflow runs CI checks but does not itself enforce branch protection, mandatory pull-request reviews or required status checks before merging. These must be configured separately in GitHub repository settings. Until verified, they remain a governance gap.

Threshold changes, schema changes, model promotions and deployment changes should follow the same pull-request process and include a reason for the change.

### What can be reconstructed after the fact

| Question | Where the answer lives |
|---|---|
| Which model made this prediction? | `predictions` table: `model_version`, `model_run_id` |
| What data and code trained it? | MLflow run parameters/tags: dataset SHA-256, code revision and training-code hash, where recorded |
| How does it compare with the alternative? | `reports/model_comparison_v1_v2.{md,json,csv}` |
| Which alerts fired, how often, and when did they resolve? | Alert history in the monitoring database |
| What ran in CI for a change? | GitHub Actions workflow history |
| Which dataset version existed? | DVC pointer files in Git history |
| Which model version is currently deployed? | `deployment.json` in the shared `mlflow_store` volume |

### Known audit weakness

`deployment.json` records the selected, deployed and previous model versions, but does not automatically record who performed a promotion or rollback, the timestamp, or the reason for the action.

Until this is implemented, the team must record every promotion and rollback in the relevant pull request or incident log.

**Enforcement:**
- **Code:** CI tests, Docker Compose validation and Docker image builds are automated.
- **Process:** Pull-request review, approval of model promotions, recording deployment reasons and maintaining the incident log are team responsibilities.
- **Gap:** Mandatory branch protection and automated deployment actor/timestamp history are not implemented or verified.
---

## 10. Incident response

| Alert | Severity guide | First response | Escalate / decide |
|---|---|---|---|
| `data_drift` only | WARNING (CRITICAL when severe) | Monitoring Owner checks the drifting feature: sensor fault, season change, or genuine shift (HVAC change)? Open the Evidently report. | **No rollback.** The model has not failed. Decide whether to retrain on new data |
| `model_degradation` (with or without drift) | WARNING or CRITICAL | Confirm sample size ≥ 30 and that actuals are valid. Check whether a promotion happened recently | If it follows a promotion: **roll back**. If not: retrain candidate and go through section 5 |
| `monitoring_execution_failure` | ERROR | Treat as **blind**: no verdict is available. Check `docker compose logs monitor`, DB, API reachability | Restore monitoring first; never read silence as health |
| Monitor health check failing | Warning | Restart the worker; confirm `latest.json` is refreshing | Governance Owner if not fixed in 15 minutes |
| API not ready | Critical | `docker compose logs api registry-init`; confirm the registry volume and `deployment.json` | Roll back to previous version if a promotion preceded it |

**Why rollback is manual:** an automatic rollback on a drift alert would punish a model for a
change in the world. A manual, evidence-based decision is slower but correct, and the previous
version is always one command away.

**Close-out:** every incident gets a short note: trigger, evidence, decision, who decided,
version before/after, and what changes to thresholds or process follow.

Demonstrated incident: On 28 November 2026, the hvac_shift scenario triggered a CRITICAL data_drift alert after three of six monitored features drifted. Following restoration of the normal scenario, the alert was subsequently marked resolved in the shared monitoring database. Evidence: evidence/person4/drift-alert-lifecycle.txt.
---

## 11. Known gaps and planned controls

| Gap | Risk | Planned control |
|---|---|---|
| Prediction API has no authentication or rate limit | Anyone with network reach can send requests and write to the monitoring store | API key header and per-key rate limit; do not expose publicly |
| No actor/time record for promotion or rollback | Weak accountability | Append `{actor, timestamp, reason}` to `deployment.json` history |
| No automatic data retention | SQLite grows without bound | Scheduled purge job |
| Branch protection not enforced by the repository | A direct push can bypass review | Enable GitHub required reviews and required CI |
| Monitoring DB on SQLite with a single writer | No concurrency or HA | Move to PostgreSQL for production |
| No automated retraining | Degradation needs a person | Triggered retraining pipeline, still gated by section 5 |
| Free Render hosting is ephemeral | Monitoring history and registry are not preserved in the cloud | Persistent disk or managed database |
| MLflow UI and Prometheus have no authentication configured in the Compose deployment | Users with network access can access operational interfaces and monitoring information | Restrict network access and introduce authentication before shared or production deployment |
| Published service ports are bound to all host interfaces rather than localhost only | Services may be reachable from other devices on the network | Bind to localhost for local demonstrations or apply appropriate firewall and network restrictions |
| Grafana uses a default fallback admin password (`admin`) if `GRAFANA_ADMIN_PASSWORD` is not configured | Default credentials create an administrative access risk | Require a strong password through `.env` before deployment outside a controlled local demonstration |

Deployment details and rehearsal steps: [deployment_runbook.md](deployment_runbook.md).
