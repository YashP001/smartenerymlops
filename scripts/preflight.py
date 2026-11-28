"""Pre-start checks for the Docker stack. Run before `docker compose up`.

Standard library only, so it works on a fresh clone before anything is
installed:  python scripts/preflight.py

Checks Docker/Compose, required files, and whether the host ports are free.
When a port is taken it reports whether the occupant is this project's API
(harmless) or something else (must be stopped, or remapped in .env).
"""

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    "data/train.csv",
    "data/test.csv",
    "models/gradient_boosting.pkl",
    "docker-compose.yml",
    "Dockerfile",
    "monitoring/prometheus/prometheus.yml",
    "monitoring/grafana/dashboards/smart-home-energy-advisor.json",
]
PORTS = [("API_PORT", 8000, "FastAPI"), ("PROMETHEUS_PORT", 9090, "Prometheus"), ("GRAFANA_PORT", 3000, "Grafana"), ("MLFLOW_PORT", 5000, "MLflow UI")]

results = []


def record(status, name, detail=""):
    results.append((status, name, detail))
    print(f"[{status:<4}] {name}" + (f" - {detail}" if detail else ""))


def load_env():
    values = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip('"')
    return {**values, **{k: v for k, v in os.environ.items() if k in values or k.endswith("_PORT")}}


def run(cmd, timeout=20):
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return done.returncode, (done.stdout + done.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.status, response.read(2048).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - any failure means "not our API"
        return None, str(exc)


def who_owns_hint(port):
    system = platform.system()
    if system == "Windows":
        return (
            f"PowerShell: Get-NetTCPConnection -LocalPort {port} | "
            "Select LocalAddress,State,OwningProcess ; then Get-Process -Id <OwningProcess>  "
            f"(or: netstat -ano | findstr :{port}  then  tasklist /FI \"PID eq <pid>\")"
        )
    return f"lsof -nP -iTCP:{port} -sTCP:LISTEN   (or: ss -ltnp 'sport = :{port}')"


def check_docker():
    if not shutil.which("docker"):
        record("FAIL", "Docker CLI", "not found on PATH. Install Docker Desktop (WSL2 backend on Windows).")
        return
    code, out = run(["docker", "compose", "version"])
    record("OK" if code == 0 else "FAIL", "Docker Compose v2", out.splitlines()[0] if out else "")
    code, out = run(["docker", "info", "--format", "{{.ServerVersion}}"])
    if code == 0:
        record("OK", "Docker daemon", f"server {out}")
    else:
        record("FAIL", "Docker daemon", "not reachable. Start Docker Desktop and wait until it says 'running'.")


def check_files():
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).exists()]
    if missing:
        record("FAIL", "Required files", "missing: " + ", ".join(missing)
               + ("  (data files come from the repo; run `git pull` or `dvc pull`)" if any("data/" in m for m in missing) else ""))
    else:
        record("OK", "Required files", f"{len(REQUIRED_FILES)} present")
    if not (ROOT / ".git").exists():
        record("WARN", "Git metadata", "no .git folder: MLflow runs will record code revision 'unknown'. Use `git clone`, not a ZIP download.")
    else:
        record("OK", "Git metadata", "present (code revision will be recorded in MLflow lineage)")
    if (ROOT / "mlflow.db").exists():
        record("INFO", "Host mlflow.db", "ignored by Docker. The stack builds its own registry in the mlflow_store volume.")


def prepare_evidence_dir():
    """Create the bind-mounted evidence folder so the non-root container user can write it."""
    folder = ROOT / "reports" / "demo" / "continuous"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        if platform.system() != "Windows":
            folder.chmod(0o777)  # container runs as uid 10001; folder is git-ignored demo evidence only
        record("OK", "Evidence folder", "reports/demo/continuous ready (state.json, latest.json, Evidently HTML land here)")
    except OSError as exc:
        record("FAIL", "Evidence folder", f"cannot prepare reports/demo/continuous: {exc}")


def check_ports(env):
    for key, default, label in PORTS:
        raw = env.get(key, str(default))
        try:
            port = int(raw)
        except ValueError:
            record("FAIL", f"{label} port", f"{key}={raw!r} is not a number")
            continue
        if not port_in_use(port):
            record("OK", f"{label} port {port}", "free")
            continue
        status, body = fetch(f"http://127.0.0.1:{port}/health") if key == "API_PORT" else (None, "")
        ours = False
        if status == 200:
            try:
                ours = "model_loaded" in json.loads(body)
            except ValueError:
                ours = False
        if ours:
            record("WARN", f"{label} port {port}", "already served by this project's API (a previous run?). "
                   "Run `docker compose down` first, or ignore if you meant to keep it running.")
        else:
            snippet = (body or "").replace("\n", " ")[:80]
            record("FAIL", f"{label} port {port}", "in use by ANOTHER program" + (f" (it answers: {snippet!r})" if snippet else "")
                   + f". Find it: {who_owns_hint(port)}. Stop it, or set {key}=<free port> in .env.")
    if platform.system() == "Windows":
        record("INFO", "Windows reserved ports", "if a free port still fails to bind: netsh interface ipv4 show excludedportrange protocol=tcp")


def main():
    print(f"Preflight for {ROOT} on {platform.system()}\n")
    env = load_env()
    check_docker()
    check_files()
    prepare_evidence_dir()
    check_ports(env)
    failed = [r for r in results if r[0] == "FAIL"]
    warned = [r for r in results if r[0] == "WARN"]
    print(f"\n{len(failed)} failure(s), {len(warned)} warning(s).")
    if failed:
        print("Fix the failures above, then re-run: python scripts/preflight.py")
        return 1
    print("Ready: docker compose up --build -d   then   python scripts/smoke_test.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
