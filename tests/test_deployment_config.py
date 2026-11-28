"""Guards the deployment wiring that no unit test of the app code would catch.

These regressions are silent in practice: the stack starts, but the monitor and
API write to different databases, or the container serves the local fallback
model, or Evidently quietly disappears.
"""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEV_DB = "/app/data/monitoring/predictions.db"


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]


def mounts(service, source):
    return [v.split(":")[1] for v in service.get("volumes", []) if v.split(":")[0] == source]


def test_api_and_monitor_share_one_monitoring_database(compose):
    api_db = compose["api"]["environment"]["MONITORING_DB_PATH"]
    monitor_db = compose["monitor"]["environment"]["MONITORING_DB_PATH"]
    assert api_db == monitor_db
    # The monitor refuses the development DB path, so this must differ from it.
    assert api_db != DEFAULT_DEV_DB
    # Both open the same named volume at the folder that contains that DB.
    api_mount = mounts(compose["api"], "monitoring_store")
    assert api_mount == mounts(compose["monitor"], "monitoring_store") == ["/app/monitoring-store"]
    assert api_db.startswith(api_mount[0] + "/")


def test_api_serves_registry_models_from_the_shared_mlflow_store(compose):
    api, init = compose["api"], compose["registry-init"]
    for key in ("MLFLOW_TRACKING_URI", "MODEL_DEPLOYMENT_FILE"):
        assert api["environment"][key] == init["environment"][key]
    assert mounts(api, "mlflow_store") == mounts(init, "mlflow_store") == ["/app/mlflow-store"]
    assert "ENERGY_MODEL_PATH" not in api["environment"], "would pin the API to the local pickle and disable versioning"
    assert api["depends_on"]["registry-init"]["condition"] == "service_completed_successfully"


def test_startup_order_and_health_gating(compose):
    assert compose["monitor"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert compose["prometheus"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert "healthcheck" in compose["api"] and "healthcheck" in compose["monitor"]
    assert "/ready" in " ".join(compose["api"]["healthcheck"]["test"])


def test_monitor_evidence_is_reachable_from_the_host(compose):
    assert "./reports/demo/continuous:/app/reports/demo/continuous" in compose["monitor"]["volumes"]


def test_host_ports_are_configurable(compose):
    assert compose["api"]["ports"] == ["${API_PORT:-8000}:8000"]
    assert compose["prometheus"]["ports"] == ["${PROMETHEUS_PORT:-9090}:9090"]
    assert compose["grafana"]["ports"] == ["${GRAFANA_PORT:-3000}:3000"]
    assert compose["mlflow-ui"]["ports"] == ["${MLFLOW_PORT:-5000}:5000"]
    # Container-internal URLs must not depend on host port choices.
    assert "http://api:8000" in compose["monitor"]["command"]


def test_default_docker_target_is_the_api():
    stages = re.findall(r"^FROM .* AS (\w+)", (ROOT / "Dockerfile").read_text(), flags=re.M)
    assert stages[-1] == "api"
    assert {"tools", "api"} <= set(stages)


def test_monitor_image_has_the_packages_the_drift_detector_imports():
    text = (ROOT / "requirements-monitor.txt").read_text()
    assert "scipy" in text and "evidently" in text and "requirements-inference.txt" in text


def test_pandas_stays_compatible_with_evidently():
    # pandas 3.x breaks `import evidently`; drift_detector then silently drops Evidently reports.
    pins = (ROOT / "requirements-inference.txt").read_text()
    assert re.search(r"^pandas==2\.", pins, flags=re.M), "pin pandas below 3 while evidently==0.7.x is used"


def test_secrets_and_local_state_stay_out_of_git_and_images():
    assert ".env" in (ROOT / ".gitignore").read_text().splitlines()
    dockerignore = (ROOT / ".dockerignore").read_text()
    assert ".env" in dockerignore and "mlflow.db" in dockerignore


def test_mlflow_ui_reads_the_same_registry_the_api_serves(compose):
    ui, api = compose["mlflow-ui"], compose["api"]
    assert mounts(ui, "mlflow_store") == ["/app/mlflow-store"]
    assert api["environment"]["MLFLOW_TRACKING_URI"] in " ".join(ui["command"]) or \
        "sqlite:////app/mlflow-store/mlflow.db" in ui["command"]
