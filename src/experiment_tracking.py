"""
Experiment tracking with MLflow and Weights & Biases.

Member 2 deliverable: Provides functions to track experiments,
log parameters, metrics, and model artifacts to both MLflow
and W&B as specified in the Phase 2 requirements (PDF §4, Member 2).

MLflow is the primary model registry; W&B is used for dashboards
and experiment visualization.
"""

import os
import sys

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Add project root for imports.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from src.features import FEATURE_COLUMNS, TARGET_COLUMN


def evaluate(y_true, predictions):
    """Calculate regression metrics."""
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(mean_squared_error(y_true, predictions) ** 0.5),
        "R2": float(r2_score(y_true, predictions)),
    }


def get_models():
    """Return the regression models to compare."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42,
        ),
    }


def track_experiments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    experiment_name: str = "smart-home-energy-advisor",
    use_wandb: bool = False,
):
    """Run experiment tracking with MLflow and optionally W&B.

    Trains each model, logs parameters, metrics, and model
    artifacts to MLflow. Optionally logs to Weights & Biases
    for dashboard visualization.

    Args:
        train_df: Training data with features and target.
        val_df: Validation data with features and target.
        experiment_name: MLflow experiment name.
        use_wandb: Whether to also log to W&B.

    Returns:
        DataFrame with validation results sorted by MAE.
    """
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]

    # Initialize W&B if requested.
    wandb_run = None
    if use_wandb:
        try:
            import wandb

            if not os.environ.get("WANDB_API_KEY") and "WANDB_MODE" not in os.environ:
                os.environ["WANDB_MODE"] = "offline"

            wandb_run = wandb.init(
                project=experiment_name,
                name="model-comparison",
                reinit=True,
            )
        except ImportError:
            print("Warning: wandb not installed. Skipping W&B logging.")
            use_wandb = False
        except Exception as e:
            print(f"Warning: W&B initialization failed ({e}). Skipping W&B logging.")
            use_wandb = False

    mlflow.set_experiment(experiment_name)

    results = []
    models = get_models()

    # Track the persistence baseline.
    with mlflow.start_run(run_name="Persistence Baseline"):
        predictions = val_df["energy_current_hour"]
        metrics = evaluate(y_val, predictions)

        mlflow.log_param("model_type", "persistence_baseline")
        mlflow.log_param("target", TARGET_COLUMN)
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
        mlflow.log_metrics(metrics)

        if use_wandb:
            import wandb

            wandb.log({"Persistence Baseline": metrics})

        results.append({"Model": "Persistence Baseline", **metrics})

    # Track each sklearn model.
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            predictions = fitted.predict(X_val)
            metrics = evaluate(y_val, predictions)

            mlflow.log_param("model_type", name)
            mlflow.log_param("target", TARGET_COLUMN)
            mlflow.log_param("feature_count", len(FEATURE_COLUMNS))

            if hasattr(fitted, "get_params"):
                params = fitted.get_params()
                for key, value in params.items():
                    if isinstance(value, (str, int, float, bool)):
                        mlflow.log_param(key, value)

            mlflow.log_metrics(metrics)

            input_example = X_train.head(5)
            signature = infer_signature(
                input_example, fitted.predict(input_example)
            )
            mlflow.sklearn.log_model(
                fitted,
                artifact_path="model",
                input_example=input_example,
                signature=signature,
                serialization_format="cloudpickle",
            )

            if use_wandb:
                import wandb

                wandb.log({name: metrics})

            results.append({"Model": name, **metrics})

    results_df = pd.DataFrame(results).sort_values("MAE")

    if wandb_run is not None:
        import wandb

        # Log summary comparison table.
        table = wandb.Table(dataframe=results_df)
        wandb.log({"model_comparison": table})
        wandb.finish()

    print("\nExperiment tracking results (sorted by MAE):")
    print(results_df.to_string(index=False))
    print(f"\nMLflow experiment: {experiment_name}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    return results_df


if __name__ == "__main__":
    TRAIN_PATH = "data/train.csv"
    VALIDATION_PATH = "data/validation.csv"

    try:
        train = pd.read_csv(TRAIN_PATH)
        validation = pd.read_csv(VALIDATION_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run the data pipeline first (split_data.py).")
        exit(1)

    # Set use_wandb=True to also log to Weights & Biases (offline or online).
    results = track_experiments(
        train, validation, use_wandb=True
    )

    os.makedirs("reports", exist_ok=True)
    results.to_csv("reports/validation_results.csv", index=False)
    print("\nSaved results to reports/validation_results.csv")
