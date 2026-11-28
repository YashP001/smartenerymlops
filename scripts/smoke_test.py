"""End-to-end smoke test for the running Docker stack.

Standard library only:  python scripts/smoke_test.py

Verifies, in pipeline order: API ready -> model identity -> prediction
traceability -> shared monitoring store -> Prometheus scraping -> Grafana
dashboard -> monitor worker health. Exit code 0 only when nothing FAILED.

Note: --predict (default) sends ONE real /predict request, which is recorded in
the monitoring store like any other prediction. Use --no-predict to avoid it.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = {
    "energy_current_hour": 0.05, "energy_previous_hour": 0.04, "energy_two_hours_ago": 0.06,
    "target_hour": 14, "target_day_of_week": 2, "target_is_weekend": 0,
}
results = []


def record(status, name, detail=""):
    results.append((status, name, detail))
    print(f"[{status:<4}] {name}" + (f" - {detail}" if detail else ""))


def http(method, url, payload=None, timeout=8):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return None, {}, str(exc)


def wait_ready(url, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        status, _, body = http("GET", url + "/ready")
        if status == 200:
            return json.loads(body)
        time.sleep(3)
    return None


def check_api(api, predict):
    ready = wait_ready(api, 120)
    if not ready:
        record("FAIL", "API /ready", f"not ready at {api}. See: docker compose logs api registry-init")
        return
    record("OK", "API /ready", f"model_loaded={ready.get('model_loaded')} database_initialized={ready.get('database_initialized')}")
    if not ready.get("database_initialized"):
        record("FAIL", "Monitoring DB", "API could not initialise the monitoring store")
    version, run_id = ready.get("model_version"), ready.get("model_run_id")
    if version in (None, "local") or not run_id:
        record("WARN", "Model identity", f"serving local fallback (version={version}). Registry bootstrap did not take effect: docker compose logs registry-init")
    else:
        record("OK", "Model identity", f"registry version {version}, run {run_id}")
    if not predict:
        record("SKIP", "Prediction traceability", "--no-predict")
        return
    status, headers, body = http("POST", api + "/predict", FEATURES)
    if status != 200:
        record("FAIL", "POST /predict", f"HTTP {status}: {body[:120]}")
        return
    out = json.loads(body)
    traced = out.get("model_version") == version and out.get("model_run_id") == run_id and out.get("prediction_id")
    record("OK" if traced else "FAIL", "Prediction traceability",
           f"prediction {out.get('prediction_id')} -> {out.get('model')} v{out.get('model_version')} run {out.get('model_run_id')}")
    status, headers, metrics = http("GET", api + "/metrics")
    record("OK" if status == 200 and "energy_advisor_" in metrics else "FAIL", "API /metrics", "energy_advisor_* series present" if status == 200 else f"HTTP {status}")
    store_id = {k.lower(): v for k, v in headers.items()}.get("x-monitoring-store-id")
    record("OK" if store_id else "FAIL", "Shared monitoring store",
           "API reports a store identity (the monitor verifies the same ID before every batch)" if store_id
           else "API reports no store ID: its monitoring DB is missing or MONITORING_DB_PATH differs from the monitor's")


def check_prometheus(url):
    status, _, body = http("GET", url + "/api/v1/targets")
    if status != 200:
        record("FAIL", "Prometheus", f"unreachable at {url}: {body[:100]}")
        return
    targets = json.loads(body)["data"]["activeTargets"]
    down = [t["scrapeUrl"] for t in targets if t.get("health") != "up"]
    if not targets:
        record("FAIL", "Prometheus targets", "none configured")
    elif down:
        record("WARN", "Prometheus targets", f"not up yet (first scrape can take 15s): {', '.join(down)}")
    else:
        record("OK", "Prometheus targets", f"{len(targets)} target(s) up")


def check_mlflow(url):
    status, _, body = http("GET", url + "/health")
    record("OK" if status == 200 else "FAIL", "MLflow UI", f"reachable at {url}" if status == 200 else f"unreachable at {url}: {body[:100]}")


def check_grafana(url):
    status, _, body = http("GET", url + "/api/health")
    if status != 200:
        record("FAIL", "Grafana", f"unreachable at {url}: {body[:100]}")
        return
    record("OK", "Grafana health", f"database={json.loads(body).get('database')}")
    status, _, body = http("GET", url + "/api/search?type=dash-db")
    boards = json.loads(body) if status == 200 else []
    record("OK" if boards else "FAIL", "Grafana dashboards provisioned",
           ", ".join(b["title"] for b in boards) if boards else f"none visible (HTTP {status}). Check provisioning mounts.")


def check_monitor():
    try:
        done = subprocess.run(["docker", "inspect", "--format", "{{.State.Status}}/{{.State.Health.Status}}", "smart-home-energy-monitor"],
                              capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        record("SKIP", "Monitor worker", "docker CLI unavailable")
        return
    state = done.stdout.strip()
    if done.returncode != 0:
        record("FAIL", "Monitor worker", "container not found. Run: docker compose up -d monitor")
    elif state == "running/healthy":
        record("OK", "Monitor worker", "running and producing fresh cycle output")
    elif state.startswith("running/"):
        record("WARN", "Monitor worker", f"{state} (allow ~90s after start for the first cycle)")
    else:
        record("FAIL", "Monitor worker", f"{state}. See: docker compose logs monitor")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-url", default=f"http://127.0.0.1:{os.getenv('API_PORT', '8000')}")
    parser.add_argument("--prometheus-url", default=f"http://127.0.0.1:{os.getenv('PROMETHEUS_PORT', '9090')}")
    parser.add_argument("--mlflow-url", default=f"http://127.0.0.1:{os.getenv('MLFLOW_PORT', '5000')}")
    parser.add_argument("--grafana-url", default=f"http://127.0.0.1:{os.getenv('GRAFANA_PORT', '3000')}")
    parser.add_argument("--no-predict", dest="predict", action="store_false")
    parser.add_argument("--api-only", action="store_true", help="Skip Prometheus, Grafana and the monitor container (native runs)")
    args = parser.parse_args()
    check_api(args.api_url.rstrip("/"), args.predict)
    if not args.api_only:
        check_prometheus(args.prometheus_url.rstrip("/"))
        check_mlflow(args.mlflow_url.rstrip("/"))
        check_grafana(args.grafana_url.rstrip("/"))
        check_monitor()
    failed = [r for r in results if r[0] == "FAIL"]
    print(f"\n{sum(r[0] == 'OK' for r in results)} ok, {sum(r[0] == 'WARN' for r in results)} warn, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
