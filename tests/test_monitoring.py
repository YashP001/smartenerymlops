"""
Tests for Prediction and Feedback Logging (Phase 3.1).

Verifies SQLite persistence, asynchronous prediction logging,
ground-truth feedback association, idempotency/duplicate detection,
finite/non-negative validation, and background logging resilience.
"""

import math
import os
import sys
import logging
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app
from src.monitoring.database import (
    init_db,
    log_prediction,
    get_prediction,
    log_feedback,
    get_feedback,
    get_recent_monitoring_records,
    get_monitoring_counts,
    PredictionNotFoundError,
    DuplicateFeedbackError,
    DuplicatePredictionError,
    InvalidFeedbackValueError,
)


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """Provide an isolated temporary SQLite database for monitoring tests."""
    db_file = str(tmp_path / "test_monitoring.db")
    monkeypatch.setenv("MONITORING_DB_PATH", db_file)
    init_db(db_file)
    return db_file


@pytest.fixture
def client(test_db):
    """Create a test client with lifespan execution in the isolated test database."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_features():
    """Valid canonical feature vector dictionary."""
    return {
        "energy_current_hour": 0.05,
        "energy_previous_hour": 0.04,
        "energy_two_hours_ago": 0.06,
        "target_hour": 14,
        "target_day_of_week": 2,
        "target_is_weekend": 0,
    }


# ─── 1. Storage & Persistence Layer Tests ─────────────────────────────

class TestDatabasePersistence:
    """Tests for SQLite database initialization and direct CRUD operations."""

    def test_database_initialization(self, test_db):
        """Verifies database file creation and table existence."""
        assert os.path.exists(test_db)
        counts = get_monitoring_counts(test_db)
        assert counts["total_predictions"] == 0
        assert counts["total_feedback"] == 0

    def test_prediction_logging_direct(self, test_db, sample_features):
        """Records a prediction directly and checks all stored attributes."""
        log_prediction(
            prediction_id="test-pred-1",
            features=sample_features,
            predicted_energy_kwh=0.061,
            model_version="GradientBoostingRegressor",
            latency_ms=1.45,
            db_path=test_db,
        )

        record = get_prediction("test-pred-1", db_path=test_db)
        assert record is not None
        assert record["prediction_id"] == "test-pred-1"
        assert record["predicted_energy_kwh"] == 0.061
        assert record["model_version"] == "GradientBoostingRegressor"
        assert record["latency_ms"] == 1.45
        assert record["energy_current_hour"] == 0.05
        assert record["target_hour"] == 14
        assert record["timestamp"] is not None

    def test_duplicate_prediction_id_rejected(self, test_db, sample_features):
        """Inserting a duplicate prediction ID raises DuplicatePredictionError."""
        log_prediction("test-dup-pred", sample_features, 0.05, db_path=test_db)
        with pytest.raises(DuplicatePredictionError):
            log_prediction("test-dup-pred", sample_features, 0.06, db_path=test_db)

    def test_feedback_ingestion_direct(self, test_db, sample_features):
        """Attaches ground-truth actual energy to an existing prediction."""
        log_prediction(
            prediction_id="test-pred-2",
            features=sample_features,
            predicted_energy_kwh=0.050,
            db_path=test_db,
        )

        fb = log_feedback("test-pred-2", actual_energy_kwh=0.054, db_path=test_db)
        assert fb["prediction_id"] == "test-pred-2"
        assert fb["actual_energy_kwh"] == 0.054
        assert fb["feedback_timestamp"] is not None

        stored_fb = get_feedback("test-pred-2", db_path=test_db)
        assert stored_fb is not None
        assert stored_fb["actual_energy_kwh"] == 0.054

    def test_feedback_accepts_zero_and_positive_finite_values(self, test_db, sample_features):
        """Valid finite non-negative values (including zero) are accepted."""
        log_prediction("test-zero", sample_features, 0.05, db_path=test_db)
        fb_zero = log_feedback("test-zero", actual_energy_kwh=0.0, db_path=test_db)
        assert fb_zero["actual_energy_kwh"] == 0.0

        log_prediction("test-positive", sample_features, 0.05, db_path=test_db)
        fb_pos = log_feedback("test-positive", actual_energy_kwh=15.75, db_path=test_db)
        assert fb_pos["actual_energy_kwh"] == 15.75

    def test_joined_monitoring_records_view(self, test_db, sample_features):
        """Verifies joined prediction and feedback records calculation."""
        log_prediction(
            prediction_id="test-pred-3",
            features=sample_features,
            predicted_energy_kwh=0.100,
            db_path=test_db,
        )
        log_feedback("test-pred-3", actual_energy_kwh=0.120, db_path=test_db)

        records = get_recent_monitoring_records(limit=10, db_path=test_db)
        assert len(records) >= 1
        rec = [r for r in records if r["prediction_id"] == "test-pred-3"][0]
        assert rec["actual_energy_kwh"] == 0.120
        assert round(rec["error"], 4) == 0.020
        assert round(rec["absolute_error"], 4) == 0.020

    def test_unknown_prediction_feedback_raises(self, test_db):
        """Submitting feedback for a non-existent prediction raises PredictionNotFoundError."""
        with pytest.raises(PredictionNotFoundError):
            log_feedback("non-existent-id", actual_energy_kwh=0.05, db_path=test_db)

    def test_duplicate_feedback_raises(self, test_db, sample_features):
        """Submitting feedback twice for the same prediction raises DuplicateFeedbackError."""
        log_prediction("test-dup-1", sample_features, 0.05, db_path=test_db)
        log_feedback("test-dup-1", actual_energy_kwh=0.06, db_path=test_db)

        with pytest.raises(DuplicateFeedbackError):
            log_feedback("test-dup-1", actual_energy_kwh=0.07, db_path=test_db)

    @pytest.mark.parametrize("invalid_val", [-0.01, -100.0])
    def test_negative_feedback_value_raises(self, test_db, sample_features, invalid_val):
        """Negative actual energy values raise InvalidFeedbackValueError."""
        log_prediction("test-neg-1", sample_features, 0.05, db_path=test_db)
        with pytest.raises(InvalidFeedbackValueError):
            log_feedback("test-neg-1", actual_energy_kwh=invalid_val, db_path=test_db)

    @pytest.mark.parametrize("non_finite_val", [float("nan"), float("inf"), float("-inf")])
    def test_nan_and_infinite_feedback_values_raise(self, test_db, sample_features, non_finite_val):
        """NaN, +inf, and -inf actual energy values raise InvalidFeedbackValueError."""
        pred_id = f"test-inf-{math.isnan(non_finite_val)}"
        log_prediction(pred_id, sample_features, 0.05, db_path=test_db)
        with pytest.raises(InvalidFeedbackValueError):
            log_feedback(pred_id, actual_energy_kwh=non_finite_val, db_path=test_db)

    @pytest.mark.parametrize("non_numeric_val", ["not_a_float", None, [0.5], {"val": 1}])
    def test_non_numeric_feedback_values_raise(self, test_db, sample_features, non_numeric_val):
        """Non-numeric values raise InvalidFeedbackValueError."""
        log_prediction("test-non-num", sample_features, 0.05, db_path=test_db)
        with pytest.raises(InvalidFeedbackValueError):
            log_feedback("test-non-num", actual_energy_kwh=non_numeric_val, db_path=test_db)


# ─── 2. API Integration Tests ────────────────────────────────────────

class TestAPIIntegration:
    """Tests for prediction logging and feedback endpoints in FastAPI."""

    def test_predict_returns_prediction_id_in_body_and_header(self, client, sample_features):
        """Verifies prediction ID in both response body and X-Prediction-ID header."""
        response = client.post("/predict", json=sample_features)
        assert response.status_code == 200
        data = response.json()

        assert "prediction_id" in data
        assert data["prediction_id"] is not None
        assert "X-Prediction-ID" in response.headers
        assert response.headers["X-Prediction-ID"] == data["prediction_id"]

    def test_consecutive_predictions_have_unique_ids(self, client, sample_features):
        """Verifies that consecutive predictions generate distinct UUIDs."""
        resp1 = client.post("/predict", json=sample_features)
        resp2 = client.post("/predict", json=sample_features)

        id1 = resp1.json()["prediction_id"]
        id2 = resp2.json()["prediction_id"]
        assert id1 != id2

    def test_predict_automatically_persists_to_database(self, client, test_db, sample_features):
        """Verifies that calling /predict writes a record to the monitoring store."""
        resp = client.post("/predict", json=sample_features)
        pred_id = resp.json()["prediction_id"]

        record = get_prediction(pred_id, db_path=test_db)
        assert record is not None
        assert record["prediction_id"] == pred_id
        assert record["predicted_energy_kwh"] == resp.json()["predicted_energy_kwh"]
        assert record["energy_current_hour"] == sample_features["energy_current_hour"]

    def test_successful_feedback_via_post_feedback(self, client, test_db, sample_features):
        """Submits feedback via POST /feedback and checks persistence."""
        resp = client.post("/predict", json=sample_features)
        pred_id = resp.json()["prediction_id"]

        fb_payload = {
            "prediction_id": pred_id,
            "actual_energy_kwh": 0.065,
        }
        fb_resp = client.post("/feedback", json=fb_payload)
        assert fb_resp.status_code == 200
        fb_data = fb_resp.json()
        assert fb_data["status"] == "success"
        assert fb_data["prediction_id"] == pred_id
        assert fb_data["actual_energy_kwh"] == 0.065
        assert "feedback_timestamp" in fb_data

        stored = get_feedback(pred_id, db_path=test_db)
        assert stored is not None
        assert stored["actual_energy_kwh"] == 0.065

    def test_successful_feedback_via_path_param(self, client, test_db, sample_features):
        """Submits feedback via POST /predictions/{id}/feedback."""
        resp = client.post("/predict", json=sample_features)
        pred_id = resp.json()["prediction_id"]

        fb_payload = {"actual_energy_kwh": 0.072}
        fb_resp = client.post(f"/predictions/{pred_id}/feedback", json=fb_payload)
        assert fb_resp.status_code == 200
        assert fb_resp.json()["prediction_id"] == pred_id
        assert fb_resp.json()["actual_energy_kwh"] == 0.072

    def test_feedback_unknown_prediction_returns_404(self, client):
        """Submitting feedback for unknown prediction returns HTTP 404."""
        fb_payload = {
            "prediction_id": "non-existent-uuid-12345",
            "actual_energy_kwh": 0.05,
        }
        fb_resp = client.post("/feedback", json=fb_payload)
        assert fb_resp.status_code == 404
        assert "not found" in fb_resp.json()["detail"].lower()

    def test_duplicate_feedback_returns_409(self, client, sample_features):
        """Submitting duplicate feedback returns HTTP 409 Conflict."""
        resp = client.post("/predict", json=sample_features)
        pred_id = resp.json()["prediction_id"]

        fb_payload = {
            "prediction_id": pred_id,
            "actual_energy_kwh": 0.055,
        }
        resp1 = client.post("/feedback", json=fb_payload)
        assert resp1.status_code == 200

        resp2 = client.post("/feedback", json=fb_payload)
        assert resp2.status_code == 409
        assert "already" in resp2.json()["detail"].lower()

    @pytest.mark.parametrize("invalid_payload", [
        {"actual_energy_kwh": -0.05},
        {"actual_energy_kwh": "invalid_string"},
        {"actual_energy_kwh": None},
    ])
    def test_invalid_feedback_payloads_return_422(self, client, sample_features, invalid_payload):
        """Negative or non-numeric energy values return HTTP 422 Unprocessable Entity."""
        resp = client.post("/predict", json=sample_features)
        pred_id = resp.json()["prediction_id"]

        payload = {"prediction_id": pred_id, **invalid_payload}
        fb_resp = client.post("/feedback", json=payload)
        assert fb_resp.status_code == 422

    def test_logging_failure_does_not_break_prediction_response(self, client, sample_features, caplog):
        """A failure during background logging logs an error but does not fail the /predict response."""
        with patch("api.main.log_prediction", side_effect=RuntimeError("Simulated DB connection failure")):
            with caplog.at_level(logging.ERROR):
                resp = client.post("/predict", json=sample_features)
                assert resp.status_code == 200
                data = resp.json()
                assert "predicted_energy_kwh" in data
                assert "prediction_id" in data
                # Verify logging error was recorded
                assert any("Failed to record prediction" in record.message for record in caplog.records)
