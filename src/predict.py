"""
Model loading and prediction pipeline.

Member 3 deliverable: Implements model loading and prediction
with consistent preprocessing as specified in Phase 2 requirements
(PDF §4, Member 3).
"""

import os
import sys
import joblib

# Add project root for imports.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from src.features import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    validate_inference_input,
    prepare_feature_vector,
)


def verify_model_features(model):
    """Verify that the loaded model accepts exactly the six expected features.

    Raises:
        ValueError: If the model has an incompatible feature schema.
    """
    if hasattr(model, "feature_names_in_"):
        actual = list(model.feature_names_in_)
        if actual != FEATURE_COLUMNS:
            raise ValueError(
                f"Feature schema mismatch. Expected {FEATURE_COLUMNS}, got {actual}"
            )
    elif hasattr(model, "n_features_in_"):
        if model.n_features_in_ != len(FEATURE_COLUMNS):
            raise ValueError(
                f"Feature count mismatch. Expected {len(FEATURE_COLUMNS)}, got {model.n_features_in_}"
            )

    # Verification probe using canonical 6-feature vector
    import pandas as pd
    test_vec = pd.DataFrame([{col: 0.0 for col in FEATURE_COLUMNS}])
    try:
        model.predict(test_vec)
    except Exception as e:
        raise ValueError(f"Loaded model failed feature verification probe: {e}")


def load_model(
    model_path: str = None,
    use_registry: bool = True,
    model_name: str = "smart-home-energy-advisor",
    version: str = "1",
):
    """Load the trained model, attempting MLflow Model Registry first, then local fallback.

    Args:
        model_path: Optional explicit local path. Defaults to models/gradient_boosting.pkl.
        use_registry: Whether to attempt loading from MLflow Model Registry first.
        model_name: MLflow registered model name.
        version: MLflow registered model version.

    Returns:
        The verified scikit-learn model.

    Raises:
        FileNotFoundError: If both registry and local loading fail.
        ValueError: If the loaded model does not match the 6 canonical features.
    """
    model = None
    registry_error = None

    if use_registry and model_path is None:
        try:
            from src.model_registry import load_registered_model
            model = load_registered_model(model_name=model_name, version=version)
            print(f"Loaded model from MLflow Model Registry: '{model_name}' (version {version})")
        except Exception as e:
            registry_error = e
            print(f"Notice: MLflow registry loading unavailable ({e}). Falling back to local artifact.")

    if model is None:
        path = model_path or MODEL_PATH
        if not os.path.exists(path):
            detail_msg = f" (registry error: {registry_error})" if registry_error else ""
            raise FileNotFoundError(
                f"Model file not found: {path}{detail_msg}\n"
                "Run 'python src/evaluate_model.py' to train and save the model."
            )
        model = joblib.load(path)
        print(f"Loaded model from local fallback: '{path}'")

    verify_model_features(model)
    return model


def predict(model, input_data: dict) -> float:
    """Make a single prediction using the trained model.

    Args:
        model: A fitted scikit-learn regressor.
        input_data: Dictionary with keys matching FEATURE_COLUMNS.

    Returns:
        Predicted energy consumption for the next hour (kWh).

    Raises:
        ValueError: If input data fails validation.
    """
    validated = validate_inference_input(input_data)
    features = prepare_feature_vector(validated)
    prediction = model.predict(features)[0]
    return float(prediction)


def predict_batch(model, inputs: list) -> list:
    """Make predictions for a batch of inputs.

    Args:
        model: A fitted scikit-learn regressor.
        inputs: List of dictionaries, each with keys matching
            FEATURE_COLUMNS.

    Returns:
        List of predicted energy values.
    """
    return [predict(model, inp) for inp in inputs]


if __name__ == "__main__":
    model = load_model()
    print(f"Model loaded: {type(model).__name__}")
    print(f"Expected features: {FEATURE_COLUMNS}")

    # Example prediction.
    sample = {
        "energy_current_hour": 0.05,
        "energy_previous_hour": 0.04,
        "energy_two_hours_ago": 0.06,
        "target_hour": 14,
        "target_day_of_week": 2,
        "target_is_weekend": 0,
    }

    result = predict(model, sample)
    print(f"\nSample prediction: {result:.6f} kWh")
