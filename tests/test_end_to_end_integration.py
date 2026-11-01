"""
End-to-End MLOps Integration Tests for Smart Home Energy Advisor (Phase 3.7).

Validates the full operational lifecycle:
1. Prediction generation and SQLite persistence
2. Delayed ground-truth feedback submission and database record pairing
3. Production simulation and statistical/Evidently drift detection
4. Performance monitoring evaluation with sufficient (>= 30) vs insufficient (< 30) feedback
5. Alert creation and resolution lifecycle across operational shifts
6. Prometheus metrics reflection from persistent storage
7. Safe, repeatable execution and idempotency under isolated test environments.
"""

import json
import os
import sqlite3
import uuid
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.predict import load_model, predict
from src.simulation import (
    generate_simulated_data,
    save_simulated_dataset,
    validate_simulated_dataset,
)
from src.monitoring.database import (
    get_connection,
    get_monitoring_stats,
    init_db,
    log_feedback,
    log_prediction,
)
from src.monitoring.drift_detector import (
    DEFAULT_REF_PATH,
    calculate_dataset_drift,
    run_drift_analysis,
)
from src.monitoring.performance_monitor import (
    BASELINE_METRICS,
    evaluate_model_performance,
)
from src.monitoring.alert_manager import AlertManager, AlertType, AlertSeverity
from src.monitoring.monitor_orchestrator import run_monitoring_cycle
from src.monitoring.metrics import MetricsContainer
from scripts.run_end_to_end_demo import run_end_to_end_demo


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Provide a pristine isolated SQLite monitoring database for testing."""
    db_file = str(tmp_path / "test_integration.db")
    monkeypatch.setenv("MONITORING_DB_PATH", db_file)
    init_db(db_file)
    return db_file


@pytest.fixture
def client(isolated_db):
    """Provide a FastAPI TestClient bound to the isolated database."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_feature_payload():
    """Valid canonical feature payload for prediction inference."""
    return {
        "energy_current_hour": 0.052,
        "energy_previous_hour": 0.048,
        "energy_two_hours_ago": 0.061,
        "target_hour": 14,
        "target_day_of_week": 2,
        "target_is_weekend": 0,
    }


class TestPredictionAndFeedbackLifecycle:
    """Validate inference prediction logging and ground-truth feedback ingestion."""

    def test_prediction_generation_and_sqlite_logging(self, client, isolated_db, sample_feature_payload):
        """Making a prediction returns valid inference results and asynchronously logs to SQLite."""
        response = client.post("/predict", json=sample_feature_payload)
        assert response.status_code == 200
        data = response.json()

        assert "predicted_energy_kwh" in data
        assert isinstance(data["predicted_energy_kwh"], float)
        assert data["predicted_energy_kwh"] >= 0.0
        assert "prediction_id" in data
        assert data["model"] == "GradientBoostingRegressor"
        pred_id = data["prediction_id"]

        # Verify header
        assert response.headers.get("X-Prediction-ID") == pred_id

        # Verify SQLite record persistence
        with get_connection(isolated_db) as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE prediction_id = ?;", (pred_id,)
            ).fetchone()
            assert row is not None
            assert row["prediction_id"] == pred_id
            assert abs(row["energy_current_hour"] - 0.052) < 1e-4
            assert abs(row["predicted_energy_kwh"] - data["predicted_energy_kwh"]) < 1e-4
            assert row["model_version"] == "GradientBoostingRegressor"

    def test_feedback_submission_and_record_pairing(self, client, isolated_db, sample_feature_payload):
        """Feedback submitted attaches to the prediction and pairs in v_monitoring_records view."""
        # 1. Predict
        pred_resp = client.post("/predict", json=sample_feature_payload)
        pred_id = pred_resp.json()["prediction_id"]
        pred_kwh = pred_resp.json()["predicted_energy_kwh"]

        # 2. Submit feedback
        actual_val = 0.0540
        fb_resp = client.post("/feedback", json={"prediction_id": pred_id, "actual_energy_kwh": actual_val})
        assert fb_resp.status_code == 200
        fb_data = fb_resp.json()
        assert fb_data["status"] == "success"
        assert fb_data["prediction_id"] == pred_id
        assert abs(fb_data["actual_energy_kwh"] - actual_val) < 1e-4

        # 3. Query paired view
        with get_connection(isolated_db) as conn:
            paired = conn.execute(
                "SELECT * FROM v_monitoring_records WHERE prediction_id = ?;", (pred_id,)
            ).fetchone()
            assert paired is not None
            assert paired["actual_energy_kwh"] == actual_val
            assert paired["predicted_energy_kwh"] == pred_kwh
            assert abs(paired["absolute_error"] - abs(actual_val - pred_kwh)) < 1e-4

    def test_duplicate_feedback_returns_conflict_status(self, client, isolated_db, sample_feature_payload):
        """Submitting feedback for an already-feedbacked prediction returns HTTP 409."""
        pred_resp = client.post("/predict", json=sample_feature_payload)
        pred_id = pred_resp.json()["prediction_id"]

        client.post("/feedback", json={"prediction_id": pred_id, "actual_energy_kwh": 0.050})
        dup_resp = client.post("/feedback", json={"prediction_id": pred_id, "actual_energy_kwh": 0.052})
        assert dup_resp.status_code == 409

    def test_feedback_for_nonexistent_prediction_returns_not_found(self, client, isolated_db):
        """Submitting feedback for an unknown prediction ID returns HTTP 404."""
        unknown_id = str(uuid.uuid4())
        resp = client.post("/feedback", json={"prediction_id": unknown_id, "actual_energy_kwh": 0.050})
        assert resp.status_code == 404


class TestSimulationAndDriftDetectionIntegration:
    """Validate data simulation and statistical drift detection."""

    def test_production_simulation_and_drift_detection_normal(self, tmp_path):
        """Normal simulation dataset generates valid schema and produces low drift share (< 0.33)."""
        sim_df = generate_simulated_data(
            scenario="normal",
            n_samples=150,
            random_state=42,
        )
        val = validate_simulated_dataset(sim_df)
        assert val["passed"] is True

        csv_path, json_path = save_simulated_dataset(
            sim_df,
            scenario="normal_test",
            output_dir=str(tmp_path),
        )
        assert os.path.exists(csv_path)
        assert os.path.exists(json_path)

        drift_res = run_drift_analysis(
            reference_data=DEFAULT_REF_PATH,
            production_data=csv_path,
            output_dir=str(tmp_path / "reports"),
            generate_reports=False,
        )
        assert drift_res["dataset_drift_detected"] is False
        assert drift_res["drift_share"] < 0.33

    def test_production_simulation_and_drift_detection_hvac_shift(self, tmp_path):
        """HVAC shift simulation creates covariate shift that is detected by drift analysis."""
        sim_df = generate_simulated_data(
            scenario="hvac_shift",
            n_samples=150,
            severity=1.2,
            random_state=42,
        )
        val = validate_simulated_dataset(sim_df)
        assert val["passed"] is True

        csv_path, _ = save_simulated_dataset(
            sim_df,
            scenario="hvac_test",
            output_dir=str(tmp_path),
        )

        drift_res = run_drift_analysis(
            reference_data=DEFAULT_REF_PATH,
            production_data=csv_path,
            output_dir=str(tmp_path / "reports"),
            generate_reports=False,
        )
        assert drift_res["drift_share"] >= 0.50
        assert drift_res["dataset_drift_detected"] is True
        assert "energy_current_hour" in drift_res["drifted_features"]

    def test_production_simulation_and_drift_detection_sensor_drift(self, tmp_path):
        """Sensor drift simulation introduces systematic calibration bias detected by drift analysis."""
        sim_df = generate_simulated_data(
            scenario="sensor_drift",
            n_samples=150,
            severity=1.2,
            random_state=42,
        )
        val = validate_simulated_dataset(sim_df)
        assert val["passed"] is True

        csv_path, _ = save_simulated_dataset(
            sim_df,
            scenario="sensor_test",
            output_dir=str(tmp_path),
        )

        drift_res = run_drift_analysis(
            reference_data=DEFAULT_REF_PATH,
            production_data=csv_path,
            output_dir=str(tmp_path / "reports"),
            generate_reports=False,
        )
        assert drift_res["drift_share"] >= 0.50
        assert drift_res["dataset_drift_detected"] is True


class TestPerformanceEvaluationAndDataThresholds:
    """Validate performance evaluation requirements (>= 30 minimum sample threshold)."""

    def test_performance_evaluation_with_insufficient_data(self, client, isolated_db, sample_feature_payload):
        """When fewer than 30 feedback records exist, status is 'insufficient_data' without alert."""
        # Insert 10 paired records
        for i in range(10):
            resp = client.post("/predict", json=sample_feature_payload)
            p_id = resp.json()["prediction_id"]
            client.post("/feedback", json={"prediction_id": p_id, "actual_energy_kwh": 0.052})

        perf_summary = evaluate_model_performance(db_path=isolated_db)
        assert perf_summary["status"] == "insufficient_data"
        assert perf_summary["is_degraded"] is False
        assert perf_summary["total_eligible_feedback_records"] == 10

        # Verify AlertManager does not create an alert for insufficient data
        mgr = AlertManager(db_path=isolated_db)
        alert, notif = mgr.evaluate_performance_result(perf_summary)
        assert alert is None
        assert notif is None

    def test_performance_evaluation_with_sufficient_healthy_feedback(self, client, isolated_db):
        """When >= 30 feedback records exist and track predictions, performance is 'healthy'."""
        for i in range(32):
            varying_payload = {
                "energy_current_hour": round(0.03 + 0.015 * (i % 6), 4),
                "energy_previous_hour": round(0.03 + 0.01 * (i % 5), 4),
                "energy_two_hours_ago": 0.045,
                "target_hour": (8 + i) % 24,
                "target_day_of_week": i % 7,
                "target_is_weekend": 1 if (i % 7) >= 5 else 0,
            }
            resp = client.post("/predict", json=varying_payload)
            p_id = resp.json()["prediction_id"]
            p_val = resp.json()["predicted_energy_kwh"]
            # Feedback closely tracks predictions with slight noise
            client.post("/feedback", json={"prediction_id": p_id, "actual_energy_kwh": round(p_val + 0.003, 4)})

        perf_summary = evaluate_model_performance(db_path=isolated_db)
        assert perf_summary["status"] == "healthy"
        assert perf_summary["is_degraded"] is False
        assert perf_summary["evaluated_predictions"] == 32
        metrics = perf_summary["metrics"]
        assert metrics["mae"] <= BASELINE_METRICS["MAE"] * 1.25
        assert metrics["rmse"] <= BASELINE_METRICS["RMSE"] * 1.25

    def test_performance_evaluation_detects_degradation(self, client, isolated_db):
        """When actual values severely diverge from predictions, degradation is detected."""
        for i in range(32):
            varying_payload = {
                "energy_current_hour": round(0.03 + 0.015 * (i % 6), 4),
                "energy_previous_hour": round(0.03 + 0.01 * (i % 5), 4),
                "energy_two_hours_ago": 0.045,
                "target_hour": (8 + i) % 24,
                "target_day_of_week": i % 7,
                "target_is_weekend": 1 if (i % 7) >= 5 else 0,
            }
            resp = client.post("/predict", json=varying_payload)
            p_id = resp.json()["prediction_id"]
            p_val = resp.json()["predicted_energy_kwh"]
            # Feedback severely degraded (> 3x error)
            client.post("/feedback", json={"prediction_id": p_id, "actual_energy_kwh": round(p_val + 0.35, 4)})

        perf_summary = evaluate_model_performance(db_path=isolated_db)
        assert perf_summary["status"] == "degraded"
        assert perf_summary["is_degraded"] is True


class TestMonitoringOrchestrationAndAlertLifecycle:
    """Validate full orchestration cycle, alert generation, and recovery."""

    def test_orchestration_cycle_alert_lifecycle(self, isolated_db, tmp_path):
        """Verify: normal data -> healthy (0 alerts), shifted data -> CRITICAL drift alert, recovery -> resolved."""
        # Seed 32 varied paired records
        for i in range(32):
            p_id = str(uuid.uuid4())
            curr_e = 0.03 + 0.01 * (i % 6)
            pred_e = curr_e + 0.002
            actual_e = pred_e + 0.003
            log_prediction(
                prediction_id=p_id,
                features={"energy_current_hour": curr_e, "energy_previous_hour": 0.04, "energy_two_hours_ago": 0.06,
                          "target_hour": (8 + i) % 24, "target_day_of_week": i % 7, "target_is_weekend": 1 if (i % 7) >= 5 else 0},
                predicted_energy_kwh=pred_e,
                model_version="GradientBoostingRegressor",
                db_path=isolated_db,
            )
            log_feedback(prediction_id=p_id, actual_energy_kwh=actual_e, db_path=isolated_db)

        # 1. Normal cycle
        sim_normal = generate_simulated_data(scenario="normal", n_samples=100, random_state=42)
        csv_normal, _ = save_simulated_dataset(sim_normal, scenario="norm", output_dir=str(tmp_path))

        mgr = AlertManager(db_path=isolated_db)
        c1 = run_monitoring_cycle(
            ref_data=DEFAULT_REF_PATH,
            prod_data=csv_normal,
            db_path=isolated_db,
            alert_manager=mgr,
        )
        assert c1["drift_status"] == "evaluated"
        assert c1["performance_status"] == "evaluated"
        assert len(c1["alerts_triggered"]) == 0
        assert c1["active_alerts_count"] == 0

        # 2. Shifted cycle (triggers drift alert)
        sim_shift = generate_simulated_data(scenario="hvac_shift", n_samples=100, severity=1.2, random_state=42)
        csv_shift, _ = save_simulated_dataset(sim_shift, scenario="shift", output_dir=str(tmp_path))

        c2 = run_monitoring_cycle(
            ref_data=DEFAULT_REF_PATH,
            prod_data=csv_shift,
            db_path=isolated_db,
            alert_manager=mgr,
        )
        assert len(c2["alerts_triggered"]) == 1
        assert c2["alerts_triggered"][0]["alert_type"] == AlertType.DATA_DRIFT.value
        assert c2["active_alerts_count"] == 1

        # 3. Recovery cycle (resolves active drift alert)
        c3 = run_monitoring_cycle(
            ref_data=DEFAULT_REF_PATH,
            prod_data=csv_normal,
            db_path=isolated_db,
            alert_manager=mgr,
        )
        assert len(c3["alerts_resolved"]) == 1
        assert c3["alerts_resolved"][0]["alert_type"] == AlertType.DATA_DRIFT.value
        assert c3["active_alerts_count"] == 0

    def test_metrics_exposition_reflects_persisted_results(self, client, isolated_db):
        """Prometheus metrics endpoint correctly queries and exposes SQLite monitoring run stats."""
        # Seed 35 paired records so performance is evaluated
        for i in range(35):
            p_id = str(uuid.uuid4())
            log_prediction(
                prediction_id=p_id,
                features={"energy_current_hour": 0.05, "energy_previous_hour": 0.04, "energy_two_hours_ago": 0.06,
                          "target_hour": 14, "target_day_of_week": 2, "target_is_weekend": 0},
                predicted_energy_kwh=0.050,
                model_version="GradientBoostingRegressor",
                db_path=isolated_db,
            )
            log_feedback(prediction_id=p_id, actual_energy_kwh=0.051, db_path=isolated_db)

        # Run cycle
        sim_df = generate_simulated_data(scenario="normal", n_samples=100, random_state=42)
        csv_p, _ = save_simulated_dataset(sim_df, scenario="norm_met", output_dir=os.path.dirname(isolated_db))

        run_monitoring_cycle(
            ref_data=DEFAULT_REF_PATH,
            prod_data=csv_p,
            db_path=isolated_db,
            model_version="GradientBoostingRegressor",
        )

        resp = client.get("/metrics")
        assert resp.status_code == 200
        content = resp.text

        assert "energy_advisor_drift_detection_runs_total 1.0" in content
        assert "energy_advisor_performance_evaluation_runs_total 1.0" in content
        assert "energy_advisor_dataset_drift_share" in content
        assert 'energy_advisor_model_mae{model_version="GradientBoostingRegressor"}' in content
        assert 'energy_advisor_model_rmse{model_version="GradientBoostingRegressor"}' in content


class TestEndToEndProgrammaticDemonstration:
    """Validate programmatic execution of the full end-to-end demonstration script."""

    def test_run_end_to_end_demo_function_completes_successfully(self, tmp_path):
        """run_end_to_end_demo completes all 10 stages in-process without error."""
        demo_db = str(tmp_path / "e2e_demo.db")
        demo_out = str(tmp_path / "reports_demo")

        results = run_end_to_end_demo(
            db_path=demo_db,
            demo_dir=demo_out,
            n_samples=32,
            seed=42,
            skip_drift_scenarios=False,
        )

        assert results is not None
        assert results["n_samples"] == 32
        expected_stages = [
            "model_verification",
            "service_health_check",
            "prediction_logging",
            "feedback_ingestion",
            "normal_simulation",
            "drift_detection_normal",
            "performance_evaluation",
            "orchestration_cycle_normal",
            "prometheus_observability",
            "drift_scenarios_and_recovery",
        ]
        assert results["stages_completed"] == expected_stages

        # Verify database files were created and populated
        assert os.path.exists(demo_db)
        with get_connection(demo_db) as conn:
            p_count = conn.execute("SELECT COUNT(*) FROM predictions;").fetchone()[0]
            f_count = conn.execute("SELECT COUNT(*) FROM feedback;").fetchone()[0]
            m_count = conn.execute("SELECT COUNT(*) FROM monitoring_runs;").fetchone()[0]
            assert p_count == 32
            assert f_count == 32
            assert m_count >= 1

    def test_repeat_execution_idempotency_and_isolation(self, tmp_path):
        """Repeated runs with isolated demo databases do not corrupt records or crash."""
        demo_db = str(tmp_path / "idempotent_demo.db")
        demo_out = str(tmp_path / "reports_idempotent")

        # Run 1
        res1 = run_end_to_end_demo(
            db_path=demo_db,
            demo_dir=demo_out,
            n_samples=30,
            seed=42,
            skip_drift_scenarios=True,
        )
        assert len(res1["stages_completed"]) == 9

        # Run 2 (repeat execution)
        res2 = run_end_to_end_demo(
            db_path=demo_db,
            demo_dir=demo_out,
            n_samples=30,
            seed=42,
            skip_drift_scenarios=True,
        )
        assert len(res2["stages_completed"]) == 9


class TestFeedbackModesAndEnvironmentRestoration:
    """Validate authentic vs synthetic feedback modes, environment safety, and database alignment."""

    def test_default_feedback_mode_uses_held_out_targets_not_noise(self, tmp_path):
        """Default feedback_mode='authentic' reads target_next_hour from data/test.csv matching features."""
        demo_db = str(tmp_path / "authentic_demo.db")
        demo_out = str(tmp_path / "reports_authentic")

        results = run_end_to_end_demo(
            db_path=demo_db,
            demo_dir=demo_out,
            n_samples=32,
            seed=42,
            skip_drift_scenarios=True,
        )

        assert results["feedback_mode"] == "authentic"

        # Read the test data directly to compare
        test_df = pd.read_csv("data/test.csv")
        sample_rows = test_df.sample(n=32, random_state=42).reset_index(drop=True)
        expected_actuals = sorted([round(float(val), 4) for val in sample_rows["target_next_hour"]])

        with get_connection(demo_db) as conn:
            records = conn.execute(
                "SELECT prediction_id, energy_current_hour, predicted_energy_kwh, actual_energy_kwh "
                "FROM v_monitoring_records;"
            ).fetchall()

        assert len(records) == 32
        persisted_actuals = sorted([round(r["actual_energy_kwh"], 4) for r in records])

        # Verify exact match with test.csv targets (order-independent of background task completion)
        assert persisted_actuals == expected_actuals

        # Verify that actual is NOT derived from prediction (errors are not artificially constrained)
        diffs = [abs(r["predicted_energy_kwh"] - r["actual_energy_kwh"]) for r in records]
        assert max(diffs) > 0.05

    def test_held_out_mode_feature_and_target_row_alignment(self, tmp_path):
        """Verify the exact same held-out row supplies both the input features and target."""
        demo_db = str(tmp_path / "row_align.db")
        demo_out = str(tmp_path / "reports_align")

        test_df = pd.read_csv("data/test.csv")
        sample_rows = test_df.sample(n=30, random_state=123).reset_index(drop=True)

        results = run_end_to_end_demo(
            db_path=demo_db,
            demo_dir=demo_out,
            n_samples=30,
            seed=123,
            skip_drift_scenarios=True,
        )

        with get_connection(demo_db) as conn:
            records = conn.execute(
                "SELECT energy_current_hour, energy_previous_hour, energy_two_hours_ago, "
                "target_hour, target_day_of_week, target_is_weekend, actual_energy_kwh "
                "FROM v_monitoring_records;"
            ).fetchall()

        assert len(records) == 30
        for rec in records:
            match = sample_rows[
                (abs(sample_rows["energy_current_hour"] - rec["energy_current_hour"]) < 1e-4) &
                (abs(sample_rows["energy_previous_hour"] - rec["energy_previous_hour"]) < 1e-4) &
                (abs(sample_rows["energy_two_hours_ago"] - rec["energy_two_hours_ago"]) < 1e-4) &
                (sample_rows["target_hour"] == rec["target_hour"]) &
                (sample_rows["target_day_of_week"] == rec["target_day_of_week"]) &
                (sample_rows["target_is_weekend"] == rec["target_is_weekend"])
            ]
            assert len(match) > 0, "Record in database did not match any source test row"
            assert any(abs(m - rec["actual_energy_kwh"]) < 1e-4 for m in match["target_next_hour"]), (
                "Feedback actual_energy_kwh did not match target_next_hour from the corresponding source row"
            )

    def test_synthetic_feedback_mode_opt_in_and_labelled(self, tmp_path):
        """Opt-in synthetic mode derives feedback from prediction + Gaussian noise."""
        demo_db = str(tmp_path / "synth_demo.db")
        demo_out = str(tmp_path / "reports_synth")

        results = run_end_to_end_demo(
            db_path=demo_db,
            demo_dir=demo_out,
            n_samples=32,
            seed=42,
            feedback_mode="synthetic",
            skip_drift_scenarios=True,
        )

        assert results["feedback_mode"] == "synthetic"

        with get_connection(demo_db) as conn:
            records = conn.execute(
                "SELECT predicted_energy_kwh, actual_energy_kwh, absolute_error "
                "FROM v_monitoring_records;"
            ).fetchall()

        # In synthetic mode, absolute error is close to 0 with std dev ~ 0.015
        errors = [r["absolute_error"] for r in records]
        assert np.mean(errors) < 0.03
        assert np.max(errors) < 0.10

    def test_environment_variable_restored_when_originally_unset(self, tmp_path, monkeypatch):
        """Environment variable MONITORING_DB_PATH is restored to unset if originally absent."""
        monkeypatch.delenv("MONITORING_DB_PATH", raising=False)
        assert "MONITORING_DB_PATH" not in os.environ

        demo_db = str(tmp_path / "env_unset.db")
        demo_out = str(tmp_path / "reports_env_unset")

        run_end_to_end_demo(
            db_path=demo_db,
            demo_dir=demo_out,
            n_samples=30,
            seed=42,
            skip_drift_scenarios=True,
        )

        assert "MONITORING_DB_PATH" not in os.environ

    def test_environment_variable_restored_when_originally_set(self, tmp_path, monkeypatch):
        """Environment variable MONITORING_DB_PATH is restored to prior value when originally set."""
        prior_value = "custom_prior_test_path.db"
        monkeypatch.setenv("MONITORING_DB_PATH", prior_value)
        assert os.environ.get("MONITORING_DB_PATH") == prior_value

        demo_db = str(tmp_path / "env_set.db")
        demo_out = str(tmp_path / "reports_env_set")

        run_end_to_end_demo(
            db_path=demo_db,
            demo_dir=demo_out,
            n_samples=30,
            seed=42,
            skip_drift_scenarios=True,
        )

        assert os.environ.get("MONITORING_DB_PATH") == prior_value

    def test_environment_variable_restored_after_failed_execution(self, tmp_path, monkeypatch):
        """Environment variable is safely restored even when run_end_to_end_demo raises an exception."""
        prior_value = "custom_prior_fail_path.db"
        monkeypatch.setenv("MONITORING_DB_PATH", prior_value)

        with pytest.raises(ValueError, match="n_samples must be at least 30"):
            run_end_to_end_demo(n_samples=10)

        assert os.environ.get("MONITORING_DB_PATH") == prior_value

    def test_live_http_database_mismatch_detected(self, tmp_path, monkeypatch):
        """When running in live API mode and server writes elsewhere, client detects mismatch."""
        demo_db = str(tmp_path / "client_target.db")
        demo_out = str(tmp_path / "reports_mismatch")

        from scripts.run_end_to_end_demo import DemoApiClient
        real_post = DemoApiClient.post

        def mock_get(self, path):
            if path == "/health":
                return 200, {"status": "healthy", "model_loaded": True}
            return 200, {}

        def mock_post(self, path, payload):
            if path == "/predict":
                # Return prediction with fresh ID not present in client's demo_db
                return 200, {
                    "predicted_energy_kwh": 0.085,
                    "prediction_id": str(uuid.uuid4()),
                    "model": "GradientBoostingRegressor",
                }
            elif path == "/feedback":
                return 200, {"status": "success", "prediction_id": payload["prediction_id"]}
            return real_post(self, path, payload)

        monkeypatch.setattr(DemoApiClient, "get", mock_get)
        monkeypatch.setattr(DemoApiClient, "post", mock_post)

        with pytest.raises(RuntimeError, match="Database alignment mismatch"):
            run_end_to_end_demo(
                api_url="http://127.0.0.1:8000",
                db_path=demo_db,
                demo_dir=demo_out,
                n_samples=30,
                seed=42,
                skip_drift_scenarios=True,
            )

    def test_held_out_mode_fails_if_insufficient_samples(self, tmp_path):
        """Authentic mode fails clearly if requested samples exceed test dataset size."""
        demo_db = str(tmp_path / "too_many.db")
        demo_out = str(tmp_path / "reports_too_many")

        with pytest.raises(ValueError, match="Requested 999999 samples, but held-out test dataset"):
            run_end_to_end_demo(
                db_path=demo_db,
                demo_dir=demo_out,
                n_samples=999999,
                seed=42,
                skip_drift_scenarios=True,
            )
