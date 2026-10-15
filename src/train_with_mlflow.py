import mlflow
import mlflow.sklearn
import pandas as pd

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

TRAIN_PATH = "data/train.csv"
VALIDATION_PATH = "data/validation.csv"
TARGET = "target_next_hour"

FEATURES = [
    "energy_current_hour",
    "energy_previous_hour",
    "energy_two_hours_ago",
    "target_hour",
    "target_day_of_week",
    "target_is_weekend",
]


def evaluate(y_true, predictions):
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(mean_squared_error(y_true, predictions) ** 0.5),
        "R2": float(r2_score(y_true, predictions)),
    }


def main():
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)

    required = FEATURES + [TARGET]
    for name, frame in [
        ("training", train),
        ("validation", validation),
    ]:
        missing = [column for column in required if column not in frame]
        if missing:
            raise ValueError(
                f"Missing columns in {name} data: {missing}"
            )
        if frame[required].isna().any().any():
            raise ValueError(f"Missing values found in {name} data")

    X_train = train[FEATURES]
    y_train = train[TARGET]
    X_val = validation[FEATURES]
    y_val = validation[TARGET]

    models = {
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

    mlflow.set_experiment("smart-home-energy-advisor")

    results = []

    # Track the persistence baseline as a separate experiment run.
    with mlflow.start_run(run_name="Persistence Baseline"):
        predictions = validation["energy_current_hour"]
        metrics = evaluate(y_val, predictions)

        mlflow.log_param("model_type", "persistence_baseline")
        mlflow.log_param("target", TARGET)
        mlflow.log_param("feature_count", len(FEATURES))
        mlflow.log_metrics(metrics)

        results.append({"Model": "Persistence Baseline", **metrics})

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            predictions = fitted.predict(X_val)
            metrics = evaluate(y_val, predictions)

            mlflow.log_param("model_type", name)
            mlflow.log_param("target", TARGET)
            mlflow.log_param("feature_count", len(FEATURES))

            if hasattr(fitted, "get_params"):
                params = fitted.get_params()
                for key, value in params.items():
                    if isinstance(value, (str, int, float, bool)):
                        mlflow.log_param(key, value)

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                fitted,
                artifact_path="model",
                input_example=X_train.head(5),
            )
            results.append({"Model": name, **metrics})

    results_df = pd.DataFrame(results).sort_values("MAE")
    print("\nMLflow validation results (sorted by MAE):")
    print(results_df.to_string(index=False))
    print("\nExperiment: smart-home-energy-advisor")
    print("MLflow tracking URI:", mlflow.get_tracking_uri())


if __name__ == "__main__":
    main()