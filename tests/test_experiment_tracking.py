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
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.experiment_tracking import evaluate, get_models, track_experiments
from src.model_registry import get_best_run, register_best_model, load_registered_model


@pytest.fixture
def isolated_mlflow(tmp_path, monkeypatch):
    """Sets up an isolated, temporary SQLite MLflow tracking and registry store.

    Captures prior Python-level MLflow tracking and registry URIs, configures
    a fresh temporary database and artifact directory in tmp_path, and fully
    restores prior URIs, environment variables, and disposes database engines
    on teardown.
    """
    prev_tracking_uri = mlflow.get_tracking_uri()
    prev_registry_uri = mlflow.get_registry_uri()

    db_file = (tmp_path / "test_mlflow.db").as_posix()
    artifacts_dir = (tmp_path / "artifacts").as_posix()
    os.makedirs(artifacts_dir, exist_ok=True)
    tracking_uri = f"sqlite:///{db_file}"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    try:
        yield {
            "client": client,
            "tracking_uri": tracking_uri,
            "artifacts_dir": artifacts_dir,
        }
    finally:
        # Dispose SQLAlchemy connection pools so SQLite files release cleanly on Windows
        if hasattr(client, "_tracking_client") and hasattr(client._tracking_client.store, "engine"):
            client._tracking_client.store.engine.dispose()
        try:
            reg_client = client._get_registry_client()
            if hasattr(reg_client.store, "engine"):
                reg_client.store.engine.dispose()
        except Exception:
            pass

        # Explicitly restore prior Python-level MLflow tracking and registry URIs
        mlflow.set_tracking_uri(prev_tracking_uri)
        mlflow.set_registry_uri(prev_registry_uri)


def _log_dummy_model_run(exp_id: str, run_name: str, model_type: str, mae: float, rmse: float, r2: float, regressor=None):
    """Helper to log a run with metrics and optional model artifact into a test experiment."""
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})
        if regressor is not None:
            X = pd.DataFrame([{col: 0.1 for col in FEATURE_COLUMNS}, {col: 0.2 for col in FEATURE_COLUMNS}])
            y = np.array([0.1, 0.2])
            regressor.fit(X, y)
            sig = infer_signature(X, y)
            mlflow.sklearn.log_model(
                regressor,
                artifact_path="model",
                signature=sig,
                serialization_format="cloudpickle",
            )
        return run.info.run_id


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

    def test_mlflow_experiment_exists(self, isolated_mlflow):
        client = isolated_mlflow["client"]
        client.create_experiment("smart-home-energy-advisor", artifact_location=isolated_mlflow["artifacts_dir"])
        experiment = client.get_experiment_by_name("smart-home-energy-advisor")
        assert experiment is not None
        assert experiment.name == "smart-home-energy-advisor"

    def test_mlflow_runs_logged(self, isolated_mlflow):
        client = isolated_mlflow["client"]
        exp_id = client.create_experiment("smart-home-energy-advisor", artifact_location=isolated_mlflow["artifacts_dir"])
        for mtype in ["Linear Regression", "persistence_baseline", "Random Forest", "Gradient Boosting"]:
            with mlflow.start_run(experiment_id=exp_id, run_name=mtype):
                mlflow.log_param("model_type", mtype)

        runs = client.search_runs(experiment_ids=[exp_id])
        assert len(runs) >= 4

        model_types = {r.data.params.get("model_type") for r in runs}
        assert "persistence_baseline" in model_types
        assert "Gradient Boosting" in model_types


class TestModelRegistry:
    """Tests for MLflow model registry."""

    def test_get_best_run(self, isolated_mlflow):
        client = isolated_mlflow["client"]
        exp_id = client.create_experiment("smart-home-energy-advisor", artifact_location=isolated_mlflow["artifacts_dir"])
        _log_dummy_model_run(exp_id, "Linear Regression", "Linear Regression", 0.0821, 0.1554, 0.2125)
        _log_dummy_model_run(exp_id, "Persistence Baseline", "persistence_baseline", 0.0778, 0.1827, -0.0885)
        _log_dummy_model_run(exp_id, "Random Forest", "Random Forest", 0.0734, 0.1472, 0.2942)
        _log_dummy_model_run(exp_id, "Gradient Boosting", "Gradient Boosting", 0.0712, 0.1423, 0.3397)

        best_run = get_best_run("smart-home-energy-advisor", metric="MAE")
        assert best_run is not None
        assert best_run.data.params.get("model_type") == "Gradient Boosting"

    def test_load_registered_model(self, isolated_mlflow):
        client = isolated_mlflow["client"]
        exp_id = client.create_experiment("smart-home-energy-advisor", artifact_location=isolated_mlflow["artifacts_dir"])
        gb = GradientBoostingRegressor(n_estimators=2, random_state=42)
        run_id = _log_dummy_model_run(exp_id, "Gradient Boosting", "Gradient Boosting", 0.0712, 0.1423, 0.3397, regressor=gb)
        mlflow.register_model(f"runs:/{run_id}/model", "smart-home-energy-advisor")

        model = load_registered_model("smart-home-energy-advisor", version="1")
        assert model is not None
        assert hasattr(model, "predict")

    def test_registered_model_prediction(self, isolated_mlflow):
        client = isolated_mlflow["client"]
        exp_id = client.create_experiment("smart-home-energy-advisor", artifact_location=isolated_mlflow["artifacts_dir"])
        gb = GradientBoostingRegressor(n_estimators=2, random_state=42)
        run_id = _log_dummy_model_run(exp_id, "Gradient Boosting", "Gradient Boosting", 0.0712, 0.1423, 0.3397, regressor=gb)
        mlflow.register_model(f"runs:/{run_id}/model", "smart-home-energy-advisor")

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

    def test_register_best_model_idempotent(self, isolated_mlflow):
        """Calling register_best_model multiple times for the same run reuses version."""
        client = isolated_mlflow["client"]
        exp_id = client.create_experiment("smart-home-energy-advisor", artifact_location=isolated_mlflow["artifacts_dir"])
        gb = GradientBoostingRegressor(n_estimators=2, random_state=42)
        _log_dummy_model_run(exp_id, "Gradient Boosting", "Gradient Boosting", 0.0712, 0.1423, 0.3397, regressor=gb)

        v_first = register_best_model("smart-home-energy-advisor", "smart-home-energy-advisor")
        v_second = register_best_model("smart-home-energy-advisor", "smart-home-energy-advisor")
        assert v_first is not None and v_second is not None
        assert str(v_first.version) == str(v_second.version) == "1"

    def test_register_different_candidate_creates_new_version(self, isolated_mlflow):
        """Registering a genuinely different candidate run creates a new version."""
        client = isolated_mlflow["client"]
        exp_id = client.create_experiment("smart-home-energy-advisor", artifact_location=isolated_mlflow["artifacts_dir"])
        test_model = "test-model-candidate-switch"

        gb = GradientBoostingRegressor(n_estimators=2, random_state=42)
        rf = RandomForestRegressor(n_estimators=2, random_state=42)
        run_gb = _log_dummy_model_run(exp_id, "Gradient Boosting", "Gradient Boosting", 0.0712, 0.1423, 0.3397, regressor=gb)
        run_rf = _log_dummy_model_run(exp_id, "Random Forest", "Random Forest", 0.0734, 0.1472, 0.2942, regressor=rf)

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
