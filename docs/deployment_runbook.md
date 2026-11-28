# Deployment runbook

How to start, verify, demonstrate, reset and troubleshoot the full MLOps stack. Written to
be followed from a **fresh clone** by someone who has not seen the project. Governance rules
that sit behind these steps are in [governance.md](governance.md).

## 1. What runs

| Service | Role | Host URL (documented configuration) |
|---|---|---|
| `registry-init` | One-shot: builds the MLflow registry (v1, candidate v2), then exits | none |
| `api` | FastAPI inference service; logs predictions to the shared monitoring DB | http://localhost:8000 (docs at `/docs`) |
| `monitor` | Continuous drift and performance monitoring worker | none (evidence in `reports/demo/continuous/`) |
| `mlflow-ui` | Experiments, runs, registered versions | http://localhost:5001 |
| `prometheus` | Scrapes `api:8000/metrics` every 15 s | http://localhost:9090 |
| `grafana` | The MLOps dashboard (anonymous viewing; admin edits) | http://localhost:3000 |

**Start order** (enforced by health checks): `registry-init` → `api` → `monitor`, `mlflow-ui` →
`prometheus` → `grafana`.

**Shared state** (Docker named volumes): `monitoring_store` (the SQLite monitoring DB used by
both `api` and `monitor`) and `mlflow_store` (tracking DB, model artifacts, and
`deployment.json`, which says which version serves traffic).

## 2. Prerequisites

Ensure the following tools and configurations are available before starting the MLOps deployment.

- **Docker Desktop:** Must be installed and running. On Windows, the WSL2 backend is recommended.
- **Docker Compose:** Version 2 or later.
- **Python:** Version 3.9 or later on the host system, required for running `preflight.py` and `smoke_test.py`. Both scripts use the Python standard library, so no additional packages need to be installed.
- **Git:** Required to clone the repository and preserve Git metadata used for model registry lineage.
- **Host ports:** Ensure ports 8000 (FastAPI), 9090 (Prometheus), 3000 (Grafana), and the configured MLflow host port are available.

**MLflow port configuration:**

The default MLflow host port is 5000. If port 5000 is occupied, configure port 5001 in the `.env` file:

```text
MLFLOW_PORT=5001
```

For the tested macOS configuration, use port 5001 because port 5000 is occupied by macOS Control Center.

Run the preflight script before starting the Docker Compose stack. It checks whether the required host ports are available.

If the project's Docker services are already running, their published ports will appear occupied. The script recognises the project's API on port 8000, but does not independently identify the project's Prometheus, Grafana or MLflow containers.

To perform a fresh preflight check while preserving existing data, first stop the stack:

```bash
docker compose down
```

This stops the containers without deleting named volumes. Then run:

```bash
python3 scripts/preflight.py
```

If a port is occupied by another application, identify the process and configure an alternative host port in `.env` before starting the stack.


## 3. Start from a fresh clone

The following steps help a new team member set up and launch the complete MLOps stack from a clean repository.

### Step 1: Clone the repository

```bash
git clone https://github.com/YashP001/smartenerymlops.git
cd smartenerymlops
git checkout <branch-to-demo>   # Use main once the team has merged
```

### Step 2: Configure environment variables

Create the environment configuration file.

**macOS / Linux (bash):**

```bash
cp .env.example .env
```

**Windows PowerShell:**

```powershell
Copy-Item .env.example .env
```

For the tested macOS configuration, ensure `.env` contains:

```text
MLFLOW_PORT=5001
```

This maps the MLflow UI to host port 5001 instead of 5000, which may already be occupied.

### Step 3: Run the preflight checks

```bash
python3 scripts/preflight.py
```

The script checks Docker availability, required project files, Git metadata, evidence directories and host port availability. Resolve genuine failures before starting the stack.

### Step 4: Build and start the MLOps stack

```bash
docker compose up --build -d
```

The first build may take several minutes because Docker needs to download dependencies and build the application images.

### Step 5: Verify service status

```bash
docker compose ps
```

Wait until the API and monitoring worker report healthy status.

### Step 6: Run the end-to-end smoke test

```bash
python3 scripts/smoke_test.py --mlflow-url http://127.0.0.1:5001
```

The smoke test verifies API readiness, model identity, prediction traceability, Prometheus metrics, shared monitoring storage, MLflow availability, Grafana health and monitoring worker activity.

**Expected result:** `0 failed`.

### Additional information

- Docker Desktop must be running, with Docker Compose v2 available.
- Git must be installed. Clone the repository rather than downloading a ZIP because model registry lineage uses Git metadata.
- Python 3.9+ is required on the host for the preflight and smoke-test scripts.
- `data/train.csv` and `data/test.csv` are included in the repository. Raw meter data managed through DVC is not required to start the stack.
- The registry initialization process trains and registers model versions v1 and v2. The API becomes healthy after initialization, and the monitoring worker becomes healthy after completing its first monitoring cycle.
- The smoke test should be run after the first monitoring cycle has completed.

## 4. Demonstration sequence

Agreed order: **Person 4 starts → Person 2 versioning → Person 3 dashboard → Person 1 induces drift → Person 4 governance and response.**

### 4.1 Person 4: Start and verify (before the audience arrives)

Before starting, ensure Docker Desktop is running and the `.env` file is configured with the required host ports.

# If the Docker Compose stack is already running, do not run preflight yet.
# Either continue with `docker compose ps` and the smoke test,
# or stop the stack first using `docker compose down`.
# This preserves named-volume data.

# Check Docker and project prerequisites
python3 scripts/preflight.py

# Build and start the complete MLOps stack
docker compose up --build -d

# Verify that all services are running
docker compose ps

# Wait until the API and monitoring worker are healthy,
# then perform the end-to-end smoke test
python3 scripts/smoke_test.py --mlflow-url http://127.0.0.1:5001
```

The smoke test must finish with **0 failed checks**.

Say:

"One command starts the API, monitoring worker, MLflow, Prometheus and Grafana. The smoke test verifies the complete pipeline, from model registry and prediction to monitoring and observability."

Set the monitoring scenario to a known healthy baseline before the demonstration:

```bash
docker compose exec monitor python scripts/run_continuous_monitoring_demo.py set-scenario normal
```

Wait for the next monitoring cycle and confirm that the system reports a healthy state before proceeding.

### 4.2 Person 2: versioning (commands run inside the shared stores)

```bash
REG="docker compose run --rm --no-deps registry-init python -m src.model_registry"
$REG status                      # v1 deployed, v2 registered
$REG select --version 2
$REG promote                     # for the drill only, see governance 5
curl http://localhost:8000/ready # now reports version 2 and its MLflow run ID
$REG rollback                    # restore v1
curl http://localhost:8000/ready
```

PowerShell: `$REG = "docker compose run --rm --no-deps registry-init python -m src.model_registry"` then `Invoke-Expression "$REG status"`, or paste the full command each time.

Say: "v2 is a Random Forest challenger. On the same locked test set it is slightly worse, so
the decision is **not to promote it**. We still show the mechanism and the rollback."
Open the comparison in `reports/model_comparison_v1_v2.md` and the run in MLflow (http://localhost:5001).

### 4.3 Person 3: dashboard

Open http://localhost:3000, the **MLOps** folder. Walk: data → validation → training → registry →
deployment → monitoring. Keep the page open while 4.4 runs.

### 4.4 Person 1: induce drift

```bash
MON="docker compose exec monitor python scripts/run_continuous_monitoring_demo.py"

$MON set-scenario normal
$MON status

$MON set-scenario hvac_shift
$MON status

$MON set-scenario normal
$MON status
```

These run inside the monitor container and write the same `state.json` that is bind-mounted at
`reports/demo/continuous/`, so the host needs no Python packages. (If you already have the
project virtualenv, `python scripts/run_continuous_monitoring_demo.py set-scenario <name>` on the
host does the same thing.) PowerShell: `function mon { docker compose exec monitor python scripts/run_continuous_monitoring_demo.py @args }`, then `mon set-scenario hvac_shift`.

Each cycle takes about 10 s plus processing. Switch scenario, wait for the next cycle, show the
result. Evidence: `reports/demo/continuous/latest.json`, `cycles.jsonl`, and Evidently HTML.
Alert history: `docker compose exec monitor python scripts/run_continuous_monitoring_demo.py history --db /app/monitoring-store/predictions.db`.

### 4.5 Person 4: governance and response

Walk governance.md sections 5, 6 and 10 against the demonstrated
alert lifecycle: HVAC shift triggered a CRITICAL data-drift alert
when 3 of 6 features drifted (50% drift share). After switching
back to normal, the alert was resolved. Discuss investigation,
recovery, approval criteria for model promotion, and rollback authority.

### Reset between rehearsals

```bash
docker compose down -v          # removes ALL volumes: monitoring DB, registry, Prometheus, Grafana state
docker compose up --build -d
```

Use `docker compose down` (no `-v`) to stop but keep the data.

## 5. Fresh-clone rehearsal checklist

Perform this checklist on a machine that has never run the project before. Repeat the rehearsal at least twice before the final presentation and record the time taken at each stage.

| # | Step | Pass condition | Time |
|---|---|---|---|
| 1 | Clone the Git repository and enter the project directory | Repository successfully cloned | |
| 2 | Copy `.env.example` to `.env` and configure the required ports | Environment variables configured correctly | |
| 3 | Run `python3 scripts/preflight.py` | Docker and required files validated; no genuine external port conflicts | |
| 4 | Run `docker compose up --build -d` | All services start without restart loops | |
| 5 | Run `docker compose ps` | API and monitoring worker report healthy status | |
| 6 | Run `python3 scripts/smoke_test.py --mlflow-url http://127.0.0.1:5001` | Smoke test completes with 0 failed checks | |
| 7 | Open Grafana, MLflow UI and FastAPI `/docs` | All interfaces load successfully | |
| 8 | Perform model version promotion and rollback demonstration | `/ready` reports version 2 after promotion and version 1 after rollback | |
| 9 | Run the monitoring scenario sequence: normal → hvac_shift → normal | Data-drift alert is triggered during HVAC shift and resolved after recovery | |
| 10 | Perform a complete reset and repeat the deployment | Services start successfully and the demonstration is reproducible | |

### Rehearsal notes

- Allow the monitoring worker to complete its first cycle before running the smoke test.
- Wait for the next monitoring cycle after changing each scenario.
- Verify alert creation and resolution using the shared monitoring database.
- Check the Evidently HTML reports and Grafana dashboard during the monitoring demonstration.
- Record any errors, unexpected behavior and corrective actions.

**Data reset warning:** The command `docker compose down -v` removes persistent Docker volumes, including the monitoring database, MLflow tracking data and monitoring history. Use it only when a complete reset is intended. To stop services while retaining stored data, use:

```bash
docker compose down
```

## 6. Port 8000 is already in use (the "unexpected service")

The preflight script checks whether the configured host ports are available. It recognises the project's API on port 8000, but does not identify the project's own Prometheus, Grafana or MLflow services when they are already running.

Run preflight before starting the stack. If the stack is already running, use `docker compose down` first; this preserves named-volume data.

```powershell
# Windows PowerShell
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select LocalAddress,OwningProcess
Get-Process -Id <OwningProcess>
# Docker container holding it?
docker ps --format "table {{.Names}}\t{{.Ports}}" | findstr 8000
```
```bash
# macOS / Linux
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

Common culprits: a local `uvicorn` run from development (`uvicorn api.main:app` in a terminal),
an older container from a previous `docker run -p 8000:8000`, Jupyter or another dev server,
and (Windows) a port reserved by Hyper-V/WSL: check with
`netsh interface ipv4 show excludedportrange protocol=tcp`.

Fix: stop that program, **or** keep it and move this stack: put `API_PORT=8001` in `.env`, then
`docker compose up -d` and use http://localhost:8001. Prometheus and the monitor reach the API
through the internal name `api:8000`, so nothing else changes. Pass `--api-url http://localhost:8001`
to host-side scripts that call the API.

## 7. Cloud deployment (Render)

The repo contains `render.yaml` (free plan, Singapore, auto-deploy on commit to `main`,
health check `/ready`, start command uses `$PORT`). Render hosts **only the API**.

**Verify (cannot be assumed; check before presenting):**

1. Render dashboard → the service shows **Live**, latest deploy is the intended commit.
2. `curl https://<your-service>.onrender.com/ready` returns `status: ready` (allow 30–50 s on a cold start).
3. `curl -X POST https://<your-service>.onrender.com/predict ...` returns a prediction (payload in `docs/docker_guide.md`).
4. Re-check right before the demo: the free tier sleeps after 15 minutes without traffic.

**Limitations to state out loud:**

| Limitation | Consequence | Why / mitigation |
|---|---|---|
| Free plan filesystem is **ephemeral** | The SQLite monitoring DB and every prediction in it are lost on each restart, redeploy or spin-down | Persistent disk needs a paid plan; use the Docker stack for monitoring demonstrations |
| No MLflow registry on Render | The API serves the bundled `models/gradient_boosting.pkl`, reported as `model_version: "local"` with no run ID | Versioning, promote and rollback are demonstrated on the Docker stack, not on Render |
| No Prometheus, Grafana or monitor worker | Drift and alert demonstrations are not available in the cloud | Same: use Docker |
| Auto-deploy on every commit to `main` | A bad merge goes live without review | Protect `main`; require CI (governance section 11) |
| Public URL, no authentication | Anyone can call `/predict` | Do not put real data through it (governance section 11) |
| Cold start of 30–50 s | First request after idle is slow | Warm it with `/ready` before presenting |

## 8. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `api` never healthy | `registry-init` failed, so no registry | `docker compose logs registry-init` |
| `/ready` says `model_version: "local"` | Registry not reachable, serving fallback | Check `registry-init` logs and that `mlflow_store` is mounted; `docker compose down -v` then up |
| `monitor` restarting | Refuses a DB it does not own, or API unreachable | `docker compose logs monitor`. If predictions were sent before the monitor's first cycle, reset with `down -v` |
| Monitor "unhealthy" right after start | First cycle not finished (up to about 90 s) | Wait; check `reports/demo/continuous/latest.json` updates |
| Grafana panels show "No data" | Prometheus has not scraped yet, or no traffic | Wait 15–30 s; run `smoke_test.py` to generate a prediction; check http://localhost:9090/targets |
| `set-scenario` seems to do nothing | Takes effect on the **next** cycle | Wait one interval (about 10 s) |
| Evidently HTML missing | `--reports` flag removed, or pandas 3 installed | Keep `pandas==2.2.3` pinned (see `tests/test_deployment_config.py`) |
| Build fails fetching packages | No network, or proxy | Retry on a network that allows PyPI and Docker Hub |
| Port permission/bind errors on Windows | Reserved port range | Section 6 |

## 9. Known constraints of this setup

- **Do not send manual predictions before the monitor's first cycle.** The monitor refuses to
  append to a database that already holds predictions it did not create. (The smoke test is
  safe to run after the monitor's first cycle; use `--no-predict` to avoid writing at all.)
- Registry lineage inside Docker records the code revision from the mounted `.git`, but marks
  the working tree as dirty because the container holds only part of the repository.
- The stack is a single-host demonstration. SQLite supports one writer and has no automatic
  retention; see governance section 11.
- **Data reset warning:** `docker compose down -v` removes persistent Docker
  volumes, including the monitoring database, MLflow tracking data and
  monitoring history. Use it only when a complete reset is intended.
  Use `docker compose down` to stop services while retaining stored data.
