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


def load_model(model_path: str = None):
    """Load the trained model from disk.

    Args:
        model_path: Path to the model file. Defaults to
            ``models/gradient_boosting.pkl`` relative to project root.

    Returns:
        The loaded scikit-learn model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    path = model_path or MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run 'python src/evaluate_model.py' to train and save the model."
        )
    return joblib.load(path)


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
