
import numpy as np
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


def evaluate_predictions(y_true, predictions):
    """Calculate regression metrics."""
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, predictions)
        ),
        "R2": r2_score(y_true, predictions),
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


def compare_models(
    train_df,
    validation_df,
    feature_cols,
    target_col,
    baseline_col,
):
    """
    Train models and compare them on the same validation period.

    The input data must already be cleaned, chronologically
    split, and free from target leakage.
    """
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = validation_df[feature_cols]
    y_val = validation_df[target_col]

    results = []

    # Persistence baseline: predict using the latest reading
    baseline_predictions = validation_df[baseline_col]

    results.append({
        "Model": "Persistence Baseline",
        **evaluate_predictions(y_val, baseline_predictions),
    })

    fitted_models = {}

    for name, model in get_models().items():
        fitted = clone(model)
        fitted.fit(X_train, y_train)

        predictions = fitted.predict(X_val)

        results.append({
            "Model": name,
            **evaluate_predictions(y_val, predictions),
        })

        fitted_models[name] = fitted

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "MAE", ascending=True
    ).reset_index(drop=True)

    return results_df, fitted_models
