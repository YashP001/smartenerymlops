"""
Tests for the FastAPI inference service.

Member 3 deliverable: Tests for api/main.py as specified in
Phase 2 requirements (PDF §4, Member 3).
"""

import os
import sys
import pytest

# Add project root for imports.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from unittest.mock import patch, MagicMock
import numpy as np
from fastapi.testclient import TestClient
from api.main import app
from src.features import FEATURE_COLUMNS


@pytest.fixture
def client():
    """Create a test client for the FastAPI application with lifespan context."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def valid_payload():
    """A valid prediction request payload."""
    return {
        "energy_current_hour": 0.05,
        "energy_previous_hour": 0.04,
        "energy_two_hours_ago": 0.06,
        "target_hour": 14,
        "target_day_of_week": 2,
        "target_is_weekend": 0,
    }


# ─── Health endpoint tests ───────────────────────────────────────────

class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contains_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_contains_model_loaded(self, client):
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)


# ─── Predict endpoint tests ─────────────────────────────────────────

class TestPredictEndpoint:
    """Tests for the POST /predict endpoint."""

    def test_predict_returns_200(self, client, valid_payload):
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200

    def test_predict_returns_prediction(self, client, valid_payload):
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        assert "predicted_energy_kwh" in data
        assert isinstance(data["predicted_energy_kwh"], float)

    def test_predict_returns_model_name(self, client, valid_payload):
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        assert "model" in data
        assert data["model"] == "GradientBoostingRegressor"

    def test_predict_returns_features_used(self, client, valid_payload):
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        assert "features_used" in data
        assert len(data["features_used"]) == 6

    def test_predict_non_negative(self, client, valid_payload):
        response = client.post("/predict", json=valid_payload)
        data = response.json()
        assert data["predicted_energy_kwh"] >= 0


# ─── Invalid input tests ────────────────────────────────────────────

class TestPredictValidation:
    """Tests for input validation on POST /predict."""

    def test_missing_field_returns_422(self, client):
        """Missing required fields should return 422."""
        incomplete = {
            "energy_current_hour": 0.05,
            "energy_previous_hour": 0.04,
            # Missing other fields.
        }
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_negative_energy_returns_422(self, client):
        """Negative energy values should be rejected."""
        payload = {
            "energy_current_hour": -0.1,
            "energy_previous_hour": 0.04,
            "energy_two_hours_ago": 0.06,
            "target_hour": 14,
            "target_day_of_week": 2,
            "target_is_weekend": 0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_hour_returns_422(self, client):
        """Hour values outside 0-23 should be rejected."""
        payload = {
            "energy_current_hour": 0.05,
            "energy_previous_hour": 0.04,
            "energy_two_hours_ago": 0.06,
            "target_hour": 25,
            "target_day_of_week": 2,
            "target_is_weekend": 0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_day_returns_422(self, client):
        """Day of week outside 0-6 should be rejected."""
        payload = {
            "energy_current_hour": 0.05,
            "energy_previous_hour": 0.04,
            "energy_two_hours_ago": 0.06,
            "target_hour": 14,
            "target_day_of_week": 8,
            "target_is_weekend": 0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_weekend_returns_422(self, client):
        """Weekend flag outside 0-1 should be rejected."""
        payload = {
            "energy_current_hour": 0.05,
            "energy_previous_hour": 0.04,
            "energy_two_hours_ago": 0.06,
            "target_hour": 14,
            "target_day_of_week": 2,
            "target_is_weekend": 3,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client):
        """Empty request body should return 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_string_energy_returns_422(self, client):
        """Non-numeric energy values should be rejected."""
        payload = {
            "energy_current_hour": "not_a_number",
            "energy_previous_hour": 0.04,
            "energy_two_hours_ago": 0.06,
            "target_hour": 14,
            "target_day_of_week": 2,
            "target_is_weekend": 0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


# ─── API docs tests ─────────────────────────────────────────────────

class TestAPIDocs:
    """Tests for API documentation endpoints."""

    def test_docs_available(self, client):
        """Swagger UI docs should be available at /docs."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_available(self, client):
        """OpenAPI schema should be available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "paths" in data
        assert "/health" in data["paths"]
        assert "/predict" in data["paths"]


# ─── Model Registry & Lifespan tests ────────────────────────────────

def _create_mock_regressor(pred_val: float = 0.08):
    """Helper to build a mock scikit-learn regressor compatible with the 6 features."""
    mock = MagicMock()
    mock.feature_names_in_ = FEATURE_COLUMNS
    mock.n_features_in_ = len(FEATURE_COLUMNS)
    mock.predict.return_value = np.array([pred_val])
    return mock


class TestRegistryAndLifecycle:
    """Tests for MLflow Model Registry integration, fallback, and lifespan management."""

    def test_startup_loads_via_mlflow_registry(self, valid_payload):
        """FastAPI loads model from MLflow Model Registry on startup."""
        mock_model = _create_mock_regressor(0.08)
        with patch("src.model_registry.load_registered_model", return_value=mock_model) as mock_reg:
            with TestClient(app) as test_client:
                health = test_client.get("/health")
                assert health.status_code == 200
                assert health.json()["model_loaded"] is True
                assert mock_reg.called

    def test_prediction_using_registered_model(self, valid_payload):
        """Predictions use the registered model loaded during startup."""
        mock_model = _create_mock_regressor(0.08)
        with patch("src.model_registry.load_registered_model", return_value=mock_model):
            with TestClient(app) as test_client:
                res = test_client.post("/predict", json=valid_payload)
                assert res.status_code == 200
                assert res.json()["predicted_energy_kwh"] == 0.08

    def test_fallback_to_local_joblib_when_registry_fails(self, valid_payload):
        """Falls back cleanly to local model artifact when MLflow registry is unreachable."""
        mock_model = _create_mock_regressor(0.05)
        with patch("src.model_registry.load_registered_model", side_effect=Exception("MLflow unavailable")):
            with patch("joblib.load", return_value=mock_model) as mock_joblib:
                with patch("os.path.exists", return_value=True):
                    with TestClient(app) as test_client:
                        res = test_client.post("/predict", json=valid_payload)
                        assert res.status_code == 200
                        assert res.json()["predicted_energy_kwh"] == 0.05
                        assert mock_joblib.called

    def test_clear_failure_when_both_registry_and_local_fail(self, valid_payload):
        """Returns 503 Service Unavailable when both registry and local fallback fail."""
        with patch("src.model_registry.load_registered_model", side_effect=Exception("MLflow down")):
            with patch("os.path.exists", return_value=False):
                with TestClient(app) as test_client:
                    health = test_client.get("/health")
                    assert health.status_code == 200
                    assert health.json()["model_loaded"] is False

                    pred = test_client.post("/predict", json=valid_payload)
                    assert pred.status_code == 503
                    assert "Model not loaded" in pred.json()["detail"]

    def test_model_loaded_only_once_at_startup(self, valid_payload):
        """Ensures load_model is called only once during startup, not per request."""
        mock_model = _create_mock_regressor(0.06)
        with patch("api.main.load_model", return_value=mock_model) as mock_load:
            with TestClient(app) as test_client:
                for _ in range(3):
                    res = test_client.post("/predict", json=valid_payload)
                    assert res.status_code == 200
                assert mock_load.call_count == 1
