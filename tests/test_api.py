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

from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


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
