"""
MLflow Model Registry management.

Member 2 deliverable: Registers the best model from MLflow
experiments into the Model Registry and manages model versions
as specified in Phase 2 requirements (PDF §4, Member 2).
"""

import mlflow
from mlflow.tracking import MlflowClient


def get_best_run(
    experiment_name: str = "smart-home-energy-advisor",
    metric: str = "MAE",
    ascending: bool = True,
):
    """Find the best MLflow run by a given metric.

    Args:
        experiment_name: Name of the MLflow experiment.
        metric: Metric to sort by.
        ascending: If True, lower metric values are better.

    Returns:
        The best MLflow Run object, or None if not found.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return None

    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.model_type != 'persistence_baseline'",
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )

    if not runs:
        print("No runs found.")
        return None

    return runs[0]


def register_best_model(
    experiment_name: str = "smart-home-energy-advisor",
    model_name: str = "smart-home-energy-advisor",
    metric: str = "MAE",
):
    """Register the best model from experiments into the MLflow
    Model Registry.

    Args:
        experiment_name: Name of the MLflow experiment.
        model_name: Name under which to register the model.
        metric: Metric used to select the best run.

    Returns:
        The registered ModelVersion object, or None on failure.
    """
    best_run = get_best_run(experiment_name, metric)
    if best_run is None:
        return None

    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    print(f"Best run: {run_id}")
    print(f"  Model type: {best_run.data.params.get('model_type')}")
    print(f"  MAE: {best_run.data.metrics.get('MAE'):.4f}")
    print(f"  RMSE: {best_run.data.metrics.get('RMSE'):.4f}")
    print(f"  R2: {best_run.data.metrics.get('R2'):.4f}")

    # Register the model.
    result = mlflow.register_model(model_uri, model_name)
    print(f"\nRegistered model: {model_name}")
    print(f"  Version: {result.version}")

    return result


def load_registered_model(
    model_name: str = "smart-home-energy-advisor",
    version: str = None,
    stage: str = None,
):
    """Load a model from the MLflow Model Registry.

    Args:
        model_name: Registered model name.
        version: Specific version number. Takes precedence over stage.
        stage: Model stage (e.g. 'Production', 'Staging').

    Returns:
        The loaded model object.
    """
    if version:
        model_uri = f"models:/{model_name}/{version}"
    elif stage:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        model_uri = f"models:/{model_name}/latest"

    print(f"Loading model from: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)


if __name__ == "__main__":
    print("MLflow Model Registry")
    print("=" * 50)

    result = register_best_model()

    if result is not None:
        print(f"\nModel '{result.name}' version {result.version} "
              f"registered successfully.")
    else:
        print("\nFailed to register model. Run experiment_tracking.py first.")
