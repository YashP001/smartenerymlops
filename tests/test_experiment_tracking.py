"""
Unit and integration tests for Member 2 deliverables:
Experiment Tracking, Model Performance Dashboards, and Model Registry.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import mlflow

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.experiment_tracking import evaluate, get_models, track_experiments
from src.model_registry import get_best_run, register_best_model, load_registered_model


class TestEvaluationMetrics:
    """Tests for regression metric calculation."""

    def test_evaluate_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = evaluate(y_true, predictions)

        assert metrics["MAE"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["RMSE"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["R2"] == pytest.approx(1.0, abs=1e-6)

    def test_evaluate_known_error(self):
        y_true = np.array([1.0, 2.0, 3.0])
        predictions = np.array([2.0, 3.0, 4.0])  # +1 error on each
        metrics = evaluate(y_true, predictions)

        assert metrics["MAE"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["RMSE"] == pytest.approx(1.0, abs=1e-6)

    def test_evaluate_returns_float_dict(self):
        y_true = np.array([0.5, 1.2])
        predictions = np.array([0.6, 1.1])
        metrics = evaluate(y_true, predictions)

        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "R2" in metrics
        for val in metrics.values():
            assert isinstance(val, float)


class TestModelDefinitions:
    """Tests for model dictionary definitions."""

    def test_get_models_keys(self):
        models = get_models()
        assert "Linear Regression" in models
        assert "Random Forest" in models
        assert "Gradient Boosting" in models

    def test_model_hyperparameters(self):
        models = get_models()
        rf = models["Random Forest"]
        gb = models["Gradient Boosting"]

        assert rf.n_estimators == 100
        assert rf.random_state == 42
        assert gb.random_state == 42


class TestExperimentTracking:
    """Tests for MLflow tracking pipeline."""

    def test_validation_results_file_exists(self):
        path = "reports/validation_results.csv"
        assert os.path.exists(path), f"{path} not found. Run experiment_tracking.py first."

    def test_validation_results_content(self):
        df = pd.read_csv("reports/validation_results.csv")
        assert len(df) == 4
        assert set(df["Model"]) == {
            "Gradient Boosting",
            "Random Forest",
            "Persistence Baseline",
            "Linear Regression",
        }
        assert list(df.columns) == ["Model", "MAE", "RMSE", "R2"]
        # Best model should be top row
        assert df.iloc[0]["Model"] == "Gradient Boosting"

    def test_gradient_boosting_beats_persistence_baseline(self):
        df = pd.read_csv("reports/validation_results.csv").set_index("Model")
        gb_mae = df.loc["Gradient Boosting", "MAE"]
        baseline_mae = df.loc["Persistence Baseline", "MAE"]
        assert gb_mae < baseline_mae, "Gradient Boosting should outperform persistence baseline"

    def test_mlflow_experiment_exists(self):
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("smart-home-energy-advisor")
        assert experiment is not None
        assert experiment.name == "smart-home-energy-advisor"

    def test_mlflow_runs_logged(self):
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("smart-home-energy-advisor")
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) >= 4

        model_types = {r.data.params.get("model_type") for r in runs}
        assert "persistence_baseline" in model_types
        assert "Gradient Boosting" in model_types


class TestModelRegistry:
    """Tests for MLflow model registry."""

    def test_get_best_run(self):
        best_run = get_best_run("smart-home-energy-advisor", metric="MAE")
        assert best_run is not None
        assert best_run.data.params.get("model_type") == "Gradient Boosting"

    def test_load_registered_model(self):
        model = load_registered_model("smart-home-energy-advisor", version="1")
        assert model is not None
        assert hasattr(model, "predict")

    def test_registered_model_prediction(self):
        model = load_registered_model("smart-home-energy-advisor", version="1")
        sample_input = pd.DataFrame([{
            "energy_current_hour": 0.5,
            "energy_previous_hour": 0.4,
            "energy_two_hours_ago": 0.3,
            "target_hour": 14,
            "target_day_of_week": 2,
            "target_is_weekend": 0,
        }])
        pred = model.predict(sample_input)
        assert len(pred) == 1
        assert float(pred[0]) >= 0.0

    def test_register_best_model_idempotent(self):
        """Calling register_best_model multiple times for the same run reuses version."""
        v_first = register_best_model("smart-home-energy-advisor", "smart-home-energy-advisor")
        v_second = register_best_model("smart-home-energy-advisor", "smart-home-energy-advisor")
        assert v_first is not None and v_second is not None
        assert str(v_first.version) == str(v_second.version) == "1"

    def test_register_different_candidate_creates_new_version(self):
        """Registering a genuinely different candidate run creates a new version."""
        client = mlflow.tracking.MlflowClient()
        test_model = "test-model-candidate-switch"
        run_gb = "e9922ecbb3af4c029aebf2f19b755694"
        run_rf = "88bbd825d67c474da7c1b071591d559c"

        try:
            # Register candidate 1
            v1 = register_best_model(run_id=run_gb, model_name=test_model)
            assert str(v1.version) == "1"

            # Repeat candidate 1 (must be idempotent)
            v1_repeat = register_best_model(run_id=run_gb, model_name=test_model)
            assert str(v1_repeat.version) == "1"

            # Register candidate 2 (genuinely different run)
            v2 = register_best_model(run_id=run_rf, model_name=test_model)
            assert str(v2.version) == "2"

            # Verify version 1 was preserved without modification
            v1_check = client.get_model_version(test_model, "1")
            assert v1_check.run_id == run_gb
        finally:
            try:
                client.delete_registered_model(test_model)
            except Exception:
                pass
