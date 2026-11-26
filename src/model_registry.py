"""MLflow registry operations and an explicit local deployment lifecycle."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.mlflow_config import PROJECT_ROOT, configure_mlflow

DEFAULT_MODEL_NAME = "smart-home-energy-advisor"
DEPLOYMENT_FILE = Path(
    os.getenv("MODEL_DEPLOYMENT_FILE", PROJECT_ROOT / "models" / "deployment.json")
)


def get_best_run(
    experiment_name: str = "smart-home-energy-advisor",
    metric: str = "MAE",
    ascending: bool = True,
):
    """Find the best non-baseline MLflow run by a given metric."""
    configure_mlflow()
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.model_type != 'persistence_baseline'",
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )
    return runs[0] if runs else None


def register_best_model(
    experiment_name: str = "smart-home-energy-advisor",
    model_name: str = DEFAULT_MODEL_NAME,
    metric: str = "MAE",
    run_id: Optional[str] = None,
):
    """Register the selected run idempotently, retaining every prior version."""
    configure_mlflow()
    client = MlflowClient()
    run = client.get_run(run_id) if run_id else get_best_run(experiment_name, metric)
    if run is None:
        return None
    target_run_id = run.info.run_id
    for version in client.search_model_versions(f"name = '{model_name}'"):
        if version.run_id == target_run_id:
            return version
    result = mlflow.register_model(f"runs:/{target_run_id}/model", model_name)
    client.set_model_version_tag(model_name, result.version, "source_run_id", target_run_id)
    return result


def list_model_versions(model_name: str = DEFAULT_MODEL_NAME):
    """Return registered versions newest first."""
    configure_mlflow()
    return sorted(
        MlflowClient().search_model_versions(f"name = '{model_name}'"),
        key=lambda item: int(item.version),
        reverse=True,
    )


def load_registered_model(
    model_name: str = DEFAULT_MODEL_NAME,
    version: Optional[str] = None,
    stage: Optional[str] = None,
):
    """Load a registry artifact and attach its immutable version and run ID."""
    configure_mlflow()
    if version:
        model_uri = f"models:/{model_name}/{version}"
        mv = MlflowClient().get_model_version(model_name, str(version))
    elif stage:
        model_uri = f"models:/{model_name}/{stage}"
        mv = MlflowClient().get_latest_versions(model_name, stages=[stage])[0]
    else:
        versions = list_model_versions(model_name)
        if not versions:
            raise LookupError(f"No registered versions found for '{model_name}'.")
        mv = versions[0]
        model_uri = f"models:/{model_name}/{mv.version}"
    model = mlflow.sklearn.load_model(model_uri)
    model._mlflow_model_name = model_name
    model._mlflow_model_version = str(mv.version)
    model._mlflow_run_id = mv.run_id
    return model


def _read_deployment() -> dict:
    if not DEPLOYMENT_FILE.exists():
        return {
            "model_name": DEFAULT_MODEL_NAME,
            "selected_version": "1",
            "deployed_version": "1",
            "previous_version": None,
        }
    with DEPLOYMENT_FILE.open(encoding="utf-8") as stream:
        data = json.load(stream)
    return {
        "model_name": data.get("model_name", DEFAULT_MODEL_NAME),
        "selected_version": str(data.get("selected_version") or "1"),
        "deployed_version": str(data.get("deployed_version") or "1"),
        "previous_version": (
            str(data["previous_version"]) if data.get("previous_version") else None
        ),
    }


def get_deployed_version() -> str:
    """Return the currently deployed version; MODEL_VERSION overrides the file."""
    return str(os.getenv("MODEL_VERSION") or _read_deployment()["deployed_version"])


def get_deployment_status() -> dict:
    """Return selected, deployed, previous, and run identity information."""
    configure_mlflow()
    state = _read_deployment()
    try:
        mv = MlflowClient().get_model_version(
            state["model_name"], state["deployed_version"]
        )
        state["deployed_run_id"] = mv.run_id
    except Exception:
        state["deployed_run_id"] = None
    return state


def _write_deployment(state: dict) -> None:
    DEPLOYMENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix="deployment-", suffix=".json", dir=str(DEPLOYMENT_FILE.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2)
            stream.write("\n")
        os.replace(temp_path, DEPLOYMENT_FILE)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def select_model_version(version: str, model_name: str = DEFAULT_MODEL_NAME) -> dict:
    """Select a candidate without changing the version serving predictions."""
    configure_mlflow()
    client = MlflowClient()
    candidate = client.get_model_version(model_name, str(version))
    # Check the artifact can be loaded before it can be promoted.
    mlflow.sklearn.load_model(f"models:/{model_name}/{version}")
    state = _read_deployment()
    state.update(model_name=model_name, selected_version=str(candidate.version))
    _write_deployment(state)
    return get_deployment_status()


def promote_selected_model(
    version: Optional[str] = None, model_name: str = DEFAULT_MODEL_NAME
) -> dict:
    """Promote the selected (or explicitly named) validated registry version."""
    state = _read_deployment()
    candidate_version = str(version or state["selected_version"])
    select_model_version(candidate_version, model_name)
    state = _read_deployment()
    old = state["deployed_version"]
    state.update(
        model_name=model_name,
        selected_version=candidate_version,
        deployed_version=candidate_version,
        previous_version=old if old != candidate_version else state["previous_version"],
    )
    _write_deployment(state)
    return get_deployment_status()


def rollback_model(model_name: str = DEFAULT_MODEL_NAME) -> dict:
    """Restore the previous deployed version and retain the current as previous."""
    state = _read_deployment()
    previous = state.get("previous_version")
    if not previous:
        raise ValueError("No previous deployed model version is recorded.")
    configure_mlflow()
    MlflowClient().get_model_version(model_name, previous)
    mlflow.sklearn.load_model(f"models:/{model_name}/{previous}")
    current = state["deployed_version"]
    state.update(
        model_name=model_name,
        selected_version=previous,
        deployed_version=previous,
        previous_version=current,
    )
    _write_deployment(state)
    return get_deployment_status()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage deployed MLflow model versions.")
    parser.add_argument("action", choices=["status", "select", "promote", "rollback"])
    parser.add_argument("--version", help="Registry version for select or promote")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()
    if args.action == "status":
        result = get_deployment_status()
    elif args.action == "select":
        if not args.version:
            parser.error("select requires --version")
        result = select_model_version(args.version, args.model_name)
    elif args.action == "promote":
        result = promote_selected_model(args.version, args.model_name)
    else:
        result = rollback_model(args.model_name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
