"""
Tests for model loading and prediction.

Member 3 deliverable: Tests for src/predict.py as specified in
Phase 2 requirements (PDF §4, Member 3).
"""

import os
import sys
import pytest

# Add project root for imports.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.features import MODEL_PATH, FEATURE_COLUMNS
from src.predict import load_model, predict


# ─── Model loading tests ─────────────────────────────────────────────

class TestModelLoading:
    """Tests for model loading functionality."""

    def test_model_file_exists(self):
        """The trained model artifact must exist."""
        assert os.path.exists(MODEL_PATH), (
            f"Model file not found at {MODEL_PATH}. "
            "Run 'python src/evaluate_model.py' first."
        )

    def test_load_model_returns_object(self):
        """load_model should return a fitted model."""
        model = load_model()
        assert model is not None

    def test_model_has_predict(self):
        """The loaded model must have a predict method."""
        model = load_model()
        assert hasattr(model, "predict")

    def test_load_nonexistent_model_raises(self):
        """Loading from a bad path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent/path/model.pkl")


# ─── Prediction tests ────────────────────────────────────────────────

class TestPrediction:
    """Tests for the prediction pipeline."""

    @pytest.fixture
    def model(self):
        return load_model()

    @pytest.fixture
    def valid_input(self):
        return {
            "energy_current_hour": 0.05,
            "energy_previous_hour": 0.04,
            "energy_two_hours_ago": 0.06,
            "target_hour": 14,
            "target_day_of_week": 2,
            "target_is_weekend": 0,
        }

    def test_prediction_returns_float(self, model, valid_input):
        """Predictions should be Python floats."""
        result = predict(model, valid_input)
        assert isinstance(result, float)

    def test_prediction_is_non_negative(self, model, valid_input):
        """Energy predictions should be non-negative."""
        result = predict(model, valid_input)
        assert result >= 0, f"Prediction was negative: {result}"

    def test_prediction_reasonable_range(self, model, valid_input):
        """Prediction should be in a reasonable kWh range."""
        result = predict(model, valid_input)
        # Based on dataset: max 3-min reading is ~0.232 kWh,
        # so hourly max should be well under 5 kWh.
        assert result < 5.0, f"Prediction seems too high: {result}"

    def test_different_inputs_different_outputs(self, model):
        """Different inputs should generally produce different outputs."""
        input1 = {
            "energy_current_hour": 0.01,
            "energy_previous_hour": 0.01,
            "energy_two_hours_ago": 0.01,
            "target_hour": 3,
            "target_day_of_week": 0,
            "target_is_weekend": 0,
        }
        input2 = {
            "energy_current_hour": 0.20,
            "energy_previous_hour": 0.18,
            "energy_two_hours_ago": 0.15,
            "target_hour": 18,
            "target_day_of_week": 5,
            "target_is_weekend": 1,
        }
        result1 = predict(model, input1)
        result2 = predict(model, input2)
        assert result1 != result2

    def test_invalid_input_raises(self, model):
        """Prediction with missing features should raise ValueError."""
        with pytest.raises(ValueError):
            predict(model, {"energy_current_hour": 0.05})

    def test_prediction_with_zero_energy(self, model):
        """Model should handle zero energy inputs gracefully."""
        zero_input = {
            "energy_current_hour": 0.0,
            "energy_previous_hour": 0.0,
            "energy_two_hours_ago": 0.0,
            "target_hour": 0,
            "target_day_of_week": 0,
            "target_is_weekend": 0,
        }
        result = predict(model, zero_input)
        assert isinstance(result, float)
