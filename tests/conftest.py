"""Keep tests away from application databases and the developer's MLflow registry."""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Hermetic MLflow store, set at import so it applies before any test module
# configures MLflow. Without it, tests that load registry models create a
# repo-root mlflow.db as a side effect, and the next run mis-detects it as a
# real local demonstration database (tests/test_demonstration_artifacts.py).
# Developers who want the local-demonstration checks set LOCAL_MLFLOW_TRACKING_URI.
_MLFLOW_TEST_DIR = Path(tempfile.mkdtemp(prefix="pytest-mlflow-"))
atexit.register(shutil.rmtree, _MLFLOW_TEST_DIR, ignore_errors=True)
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(_MLFLOW_TEST_DIR / 'mlflow.db').as_posix()}"
os.environ.pop("MLFLOW_REGISTRY_URI", None)
os.environ.pop("MODEL_VERSION", None)


@pytest.fixture(autouse=True)
def isolated_default_monitoring_database(tmp_path, monkeypatch):
    """Use a fresh store per test; tests may still override the path explicitly."""
    monkeypatch.setenv("MONITORING_DB_PATH", str(tmp_path / "default_monitoring.db"))
