"""Shared, portable MLflow tracking and registry configuration."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Explicitly trust the internal tree storage used by the project's locally
# trained sklearn tree ensembles when MLflow 3 serializes them with skops.
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def default_tracking_uri() -> str:
    """Return a repository-local SQLite URI without URL-encoding spaces."""
    path = (PROJECT_ROOT / "mlflow.db").resolve()
    return f"sqlite:///{path.as_posix()}"


def configure_mlflow() -> str:
    """Apply the same tracking and registry stores across project entry points."""
    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or default_tracking_uri()
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI") or tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    return tracking_uri
