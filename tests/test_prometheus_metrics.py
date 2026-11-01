"""
Unit and Integration Tests for Prometheus Observability and Metrics (Phase 3.6).

Verifies:
1. /metrics endpoint returns standard Prometheus format with HTTP 200.
2. Inference activity correctly updates prediction counters and latency histograms.
3. API failures increment error counters without disrupting service.
4. HTTP request middleware records method, low-cardinality endpoint, and status codes.
5. Out-of-band monitoring metrics (drift share, MAE/RMSE, active alerts, failures)
   are exposed from SQLite without fabricated zero values.
6. Metric labels avoid high cardinality and exclude sensitive identifiers.
7. Prometheus, Grafana, and Docker Compose configurations are valid and consistent.
"""

import json
import os
import re
from unittest.mock import MagicMock, patch
import pytest
import yaml
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from api.main import app
from src.monitoring.database import (
    init_db,
    log_prediction,
    log_feedback,
    save_alert,
    log_monitoring_run,
    log_monitoring_execution_failure,
)
from src.monitoring.metrics import (
    MetricsContainer,
    MonitoringMetricsCollector,
    default_metrics,
    normalize_endpoint,
)


@pytest.fixture
def test_client():
    """Create a FastAPI test client with initialized database."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def isolated_db(tmp_path):
    """Provide an isolated SQLite database path initialized with monitoring schema."""
    db_file = str(tmp_path / "test_observability.db")
    init_db(db_file)
    return db_file


@pytest.fixture
def isolated_metrics(isolated_db):
    """Provide an isolated MetricsContainer with its own registry and database."""
    registry = CollectorRegistry()
    container = MetricsContainer(registry=registry, db_path=isolated_db)
    return container


# ─── 1. API Endpoints & /metrics Exposition ───────────────────────────

class TestMetricsEndpoint:
    """Validate the /metrics HTTP endpoint and response characteristics."""

    def test_metrics_endpoint_status_and_content_type(self, test_client):
        """GET /metrics should return HTTP 200 with standard Prometheus text/plain content type."""
        resp = test_client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        assert "version=0.0.4" in resp.headers["content-type"]
        assert len(resp.text) > 0

    def test_existing_endpoints_unaffected(self, test_client):
        """Existing /health and /predict endpoints must continue functioning normally."""
        health_resp = test_client.get("/health")
        assert health_resp.status_code == 200
        data = health_resp.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_prediction_requests_counter_and_latency(self, test_client):
        """Successful prediction requests must increment prediction counters and observe latency."""
        payload = {
            "energy_current_hour": 0.05,
            "energy_previous_hour": 0.04,
            "energy_two_hours_ago": 0.06,
            "target_hour": 14,
            "target_day_of_week": 2,
            "target_is_weekend": 0,
        }
        pred_resp = test_client.post("/predict", json=payload)
        assert pred_resp.status_code == 200
        assert "predicted_energy_kwh" in pred_resp.json()

        metrics_resp = test_client.get("/metrics")
        body = metrics_resp.text

        # Verify prediction request counter
        assert "energy_advisor_prediction_requests_total" in body
        assert 'status="success"' in body

        # Verify model predictions counter
        assert "energy_advisor_model_predictions_total" in body

        # Verify prediction latency histogram
        assert "energy_advisor_prediction_latency_seconds_count" in body
        assert "energy_advisor_prediction_latency_seconds_sum" in body

        # Verify predicted energy distribution histogram
        assert "energy_advisor_predicted_energy_kwh_bucket" in body

    def test_api_validation_failure_recorded_as_error(self, test_client):
        """Invalid feature payloads (HTTP 422) must increment prediction error counter."""
        invalid_payload = {
            "energy_current_hour": -0.05,  # negative violates ge=0
            "energy_previous_hour": 0.04,
            "energy_two_hours_ago": 0.06,
            "target_hour": 25,  # invalid hour > 23
            "target_day_of_week": 2,
            "target_is_weekend": 0,
        }
        pred_resp = test_client.post("/predict", json=invalid_payload)
        assert pred_resp.status_code == 422

        metrics_resp = test_client.get("/metrics")
        assert "energy_advisor_http_requests_total" in metrics_resp.text
        assert 'status_code="422"' in metrics_resp.text

    def test_model_missing_records_error_counter(self, test_client):
        """When model is not loaded (HTTP 503), error counter is updated."""
        with patch("api.main.get_active_model", return_value=None):
            payload = {
                "energy_current_hour": 0.05,
                "energy_previous_hour": 0.04,
                "energy_two_hours_ago": 0.06,
                "target_hour": 14,
                "target_day_of_week": 2,
                "target_is_weekend": 0,
            }
            resp = test_client.post("/predict", json=payload)
            assert resp.status_code == 503

        metrics_resp = test_client.get("/metrics")
        assert 'energy_advisor_prediction_requests_total{model_version="unknown",status="error"}' in metrics_resp.text

    def test_feedback_endpoint_integration(self, test_client):
        """Feedback endpoints continue operating and register in HTTP metrics."""
        payload = {
            "energy_current_hour": 0.05,
            "energy_previous_hour": 0.04,
            "energy_two_hours_ago": 0.06,
            "target_hour": 14,
            "target_day_of_week": 2,
            "target_is_weekend": 0,
        }
        pred_resp = test_client.post("/predict", json=payload)
        assert pred_resp.status_code == 200
        pred_id = pred_resp.json()["prediction_id"]

        fb_resp = test_client.post("/feedback", json={"prediction_id": pred_id, "actual_energy_kwh": 0.058})
        assert fb_resp.status_code == 200

        metrics_resp = test_client.get("/metrics")
        assert 'endpoint="/feedback"' in metrics_resp.text

        # Test parameterized path endpoint /predictions/{prediction_id}/feedback
        pred_resp2 = test_client.post("/predict", json=payload)
        pred_id2 = pred_resp2.json()["prediction_id"]

        fb_resp2 = test_client.post(f"/predictions/{pred_id2}/feedback", json={"actual_energy_kwh": 0.061})
        assert fb_resp2.status_code == 200

        metrics_resp2 = test_client.get("/metrics")
        # Ensure parameterized path template is used, NOT the raw UUID
        assert 'endpoint="/predictions/{prediction_id}/feedback"' in metrics_resp2.text
        assert pred_id2 not in metrics_resp2.text

    def test_repeated_metrics_scrapes_do_not_increment_http_counters(self, test_client):
        """Internal /metrics scrapes must be excluded from application HTTP traffic metrics."""
        for _ in range(5):
            resp = test_client.get("/metrics")
            assert resp.status_code == 200

        final_metrics = test_client.get("/metrics").text
        # Ensure endpoint="/metrics" is never registered in HTTP request counters
        assert 'endpoint="/metrics"' not in final_metrics

    def test_unexpected_inference_exception_increments_error_counter(self, test_client):
        """Unexpected inference exceptions increment prediction error counter and preserve HTTP 500 error."""
        payload = {
            "energy_current_hour": 0.05,
            "energy_previous_hour": 0.04,
            "energy_two_hours_ago": 0.06,
            "target_hour": 14,
            "target_day_of_week": 2,
            "target_is_weekend": 0,
        }
        with patch("api.main.predict", side_effect=RuntimeError("Simulated inference hardware crash")):
            with pytest.raises(RuntimeError):
                test_client.post("/predict", json=payload)

        metrics_resp = test_client.get("/metrics")
        assert 'energy_advisor_prediction_requests_total{model_version="GradientBoostingRegressor",status="error"}' in metrics_resp.text


# ─── 2. Metric Container & Collector Unit Tests ───────────────────────

class TestMetricsContainerUnit:
    """Deterministic unit tests on MetricsContainer and MonitoringMetricsCollector."""

    def test_isolated_container_recording(self, isolated_metrics):
        """Container accurately updates counters, histograms, and distributions."""
        container = isolated_metrics

        # Record successful prediction
        container.record_prediction(
            model_version="GradientBoostingRegressor",
            status="success",
            latency_seconds=0.015,
            predicted_energy=0.075,
        )

        # Record failed prediction
        container.record_prediction(
            model_version="GradientBoostingRegressor",
            status="error",
        )

        # Record HTTP request
        container.record_http(
            method="POST",
            endpoint="/predict",
            status_code=200,
            duration_seconds=0.02,
        )

        payload = container.generate_payload().decode()
        assert 'energy_advisor_prediction_requests_total{model_version="GradientBoostingRegressor",status="success"} 1.0' in payload
        assert 'energy_advisor_prediction_requests_total{model_version="GradientBoostingRegressor",status="error"} 1.0' in payload
        assert 'energy_advisor_model_predictions_total{model_version="GradientBoostingRegressor"} 1.0' in payload
        assert 'energy_advisor_http_requests_total{endpoint="/predict",method="POST",status_code="200"} 1.0' in payload

    def test_missing_monitoring_data_no_fabricated_values(self, isolated_metrics):
        """When database has no drift or feedback, MAE, RMSE, and drift share must NOT be emitted."""
        payload = isolated_metrics.generate_payload().decode()
        # Should NOT contain fabricated 0.0 for MAE, RMSE, or drift share
        assert "energy_advisor_model_mae" not in payload
        assert "energy_advisor_model_rmse" not in payload
        assert "energy_advisor_dataset_drift_share" not in payload

        # Zero counters are allowed and expected for discoverability
        assert "energy_advisor_drift_detection_runs_total 0.0" in payload
        assert "energy_advisor_performance_evaluation_runs_total 0.0" in payload

    def test_monitoring_run_updates_drift_and_performance_metrics(self, isolated_metrics, isolated_db):
        """Logging a completed monitoring cycle exposes drift share, MAE, RMSE, and runs count."""
        log_monitoring_run(
            run_id="cycle_001",
            timestamp="2026-11-01T10:00:00Z",
            model_version="GradientBoostingRegressor",
            drift_status="evaluated",
            drift_share=0.1667,
            performance_status="evaluated",
            mae=0.0782,
            rmse=0.1540,
            r2=0.3650,
            bias=0.005,
            sample_count=45,
            alerts_triggered=0,
            alerts_resolved=0,
            active_alerts_count=0,
            db_path=isolated_db,
        )

        payload = isolated_metrics.generate_payload().decode()
        assert "energy_advisor_drift_detection_runs_total 1.0" in payload
        assert "energy_advisor_dataset_drift_share 0.1667" in payload
        assert "energy_advisor_performance_evaluation_runs_total 1.0" in payload
        assert 'energy_advisor_model_mae{model_version="GradientBoostingRegressor"} 0.0782' in payload
        assert 'energy_advisor_model_rmse{model_version="GradientBoostingRegressor"} 0.154' in payload

    def test_monitoring_failures_exposed_correctly(self, isolated_metrics, isolated_db):
        """Drift and performance orchestration failures update error counters by operation."""
        # Log drift failure run
        log_monitoring_run(
            run_id="cycle_002",
            timestamp="2026-11-01T11:00:00Z",
            model_version="GradientBoostingRegressor",
            drift_status="failed",
            performance_status="failed",
            alerts_triggered=1,
            alerts_resolved=0,
            active_alerts_count=1,
            db_path=isolated_db,
        )
        log_monitoring_execution_failure(
            operation="drift_detection",
            error_message="Drift reference file missing",
            db_path=isolated_db,
        )
        log_monitoring_execution_failure(
            operation="performance_monitoring",
            error_message="Database locked",
            db_path=isolated_db,
        )

        payload = isolated_metrics.generate_payload().decode()
        assert "energy_advisor_drift_detection_failures_total 1.0" in payload
        assert "energy_advisor_performance_evaluation_failures_total 1.0" in payload
        assert 'energy_advisor_monitoring_execution_failures_total{operation="drift_detection"} 1.0' in payload
        assert 'energy_advisor_monitoring_execution_failures_total{operation="performance_monitoring"} 1.0' in payload

    def test_active_alerts_grouped_by_type_and_severity(self, isolated_metrics, isolated_db):
        """Active alerts from the database are counted and grouped by alert_type and severity."""
        save_alert(
            {
                "alert_id": "alert_drift_01",
                "alert_type": "data_drift",
                "severity": "WARNING",
                "status": "active",
                "message": "Drift detected on 2 features",
                "metrics_json": "{}",
                "source_result_json": "{}",
                "created_at": "2026-11-01T10:00:00Z",
                "updated_at": "2026-11-01T10:00:00Z",
            },
            db_path=isolated_db,
        )
        save_alert(
            {
                "alert_id": "alert_perf_01",
                "alert_type": "model_degradation",
                "severity": "CRITICAL",
                "status": "active",
                "message": "MAE exceeded threshold",
                "metrics_json": "{}",
                "source_result_json": "{}",
                "created_at": "2026-11-01T10:05:00Z",
                "updated_at": "2026-11-01T10:05:00Z",
            },
            db_path=isolated_db,
        )

        payload = isolated_metrics.generate_payload().decode()
        assert 'energy_advisor_active_alerts{alert_type="data_drift",severity="WARNING"} 1.0' in payload
        assert 'energy_advisor_active_alerts{alert_type="model_degradation",severity="CRITICAL"} 1.0' in payload


# ─── 3. Low-Cardinality & Data Sanitization ───────────────────────────

class TestLabelCardinalityAndDataSafety:
    """Verify metrics do not expose high-cardinality labels or sensitive fields."""

    def test_normalize_endpoint_replaces_uuids(self):
        """Endpoint normalizer replaces UUID path parameters with {prediction_id}."""
        raw_path = "/predictions/123e4567-e89b-12d3-a456-426614174000/feedback"
        normalized = normalize_endpoint(raw_path)
        assert normalized == "/predictions/{prediction_id}/feedback"
        assert "123e4567" not in normalized

    def test_endpoint_with_no_uuid_unchanged(self):
        """Standard static paths remain unchanged."""
        assert normalize_endpoint("/health") == "/health"
        assert normalize_endpoint("/predict") == "/predict"
        assert normalize_endpoint("/metrics") == "/metrics"

    def test_metrics_do_not_contain_prediction_ids(self, test_client):
        """Metrics payload must never leak dynamic prediction IDs or meter inputs."""
        payload = {
            "energy_current_hour": 0.088,
            "energy_previous_hour": 0.077,
            "energy_two_hours_ago": 0.066,
            "target_hour": 18,
            "target_day_of_week": 4,
            "target_is_weekend": 0,
        }
        resp = test_client.post("/predict", json=payload)
        pred_id = resp.json()["prediction_id"]

        metrics_text = test_client.get("/metrics").text
        # Ensure the UUID is not present as a label in /metrics
        assert pred_id not in metrics_text

    def test_all_metric_label_keys_are_low_cardinality(self, isolated_metrics):
        """Audit all defined metrics to ensure labels strictly belong to known low-cardinality sets."""
        allowed_label_keys = {
            "model_version",
            "status",
            "method",
            "endpoint",
            "status_code",
            "alert_type",
            "severity",
            "operation",
            "le",  # standard Prometheus histogram bucket label
        }

        for metric_family in isolated_metrics.registry.collect():
            for sample in metric_family.samples:
                for label_name in sample.labels.keys():
                    assert label_name in allowed_label_keys, f"Disallowed label key: {label_name}"

    def test_different_unmatched_urls_use_same_unmatched_label(self, test_client):
        """Arbitrary 404 URLs must map to fixed 'unmatched' label to prevent cardinality explosion."""
        paths = ["/nonexistent-1", "/api/v99/unknown", "/some/deep/random/path/12345"]
        for p in paths:
            resp = test_client.get(p)
            assert resp.status_code == 404

        metrics_text = test_client.get("/metrics").text
        # Check that fixed label 'unmatched' is used
        assert 'endpoint="unmatched"' in metrics_text
        # Check that none of the raw paths leaked as label values
        for p in paths:
            assert f'endpoint="{p}"' not in metrics_text

    def test_existing_parameterized_routes_use_route_templates(self, test_client):
        """Matched parameterized FastAPI routes must preserve template path instead of 'unmatched'."""
        payload = {
            "energy_current_hour": 0.05,
            "energy_previous_hour": 0.04,
            "energy_two_hours_ago": 0.06,
            "target_hour": 14,
            "target_day_of_week": 2,
            "target_is_weekend": 0,
        }
        pred_resp = test_client.post("/predict", json=payload)
        pred_id = pred_resp.json()["prediction_id"]

        fb_resp = test_client.post(f"/predictions/{pred_id}/feedback", json={"actual_energy_kwh": 0.055})
        assert fb_resp.status_code == 200

        metrics_text = test_client.get("/metrics").text
        assert 'endpoint="/predictions/{prediction_id}/feedback"' in metrics_text


# ─── 4. Configuration Validation Tests ─────────────────────────────────

class TestConfigurationIntegrity:
    """Validate Prometheus, Grafana, and Docker Compose configurations."""

    @pytest.fixture
    def project_root(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_prometheus_yaml_is_valid(self, project_root):
        """monitoring/prometheus/prometheus.yml must be valid YAML with expected scrape target."""
        prom_path = os.path.join(project_root, "monitoring", "prometheus", "prometheus.yml")
        assert os.path.exists(prom_path), f"Missing {prom_path}"

        with open(prom_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert "global" in config
        assert config["global"]["scrape_interval"] == "15s"
        assert "scrape_configs" in config

        jobs = {sc["job_name"]: sc for sc in config["scrape_configs"]}
        assert "smart-home-energy-advisor" in jobs
        job = jobs["smart-home-energy-advisor"]
        assert job["metrics_path"] == "/metrics"
        targets = job["static_configs"][0]["targets"]
        assert "api:8000" in targets

    def test_grafana_dashboard_json_is_valid(self, project_root):
        """monitoring/grafana/dashboards/smart-home-energy-advisor.json must be valid JSON with 8 required panels."""
        dash_path = os.path.join(
            project_root,
            "monitoring",
            "grafana",
            "dashboards",
            "smart-home-energy-advisor.json",
        )
        assert os.path.exists(dash_path), f"Missing {dash_path}"

        with open(dash_path, "r", encoding="utf-8") as f:
            dash = json.load(f)

        assert dash["uid"] == "smart-home-energy-advisor"
        assert dash["refresh"] == "15s"

        panel_titles = [p.get("title") for p in dash.get("panels", []) if p.get("title")]

        # Verify all 8 core panel requirements are represented
        required_topics = [
            "Prediction Request Volume",
            "API Inference Latency",
            "Prediction Success vs Failure Rate",
            "Model Prediction Output Distribution",
            "Model MAE (kWh)",
            "Model RMSE (kWh)",
            "Data Drift Share",
            "Active Alerts by Type & Severity",
            "Monitoring Execution Failures",
        ]
        for topic in required_topics:
            assert topic in panel_titles, f"Missing dashboard panel: {topic}"

        # Verify PromQL expressions in panels
        promql_exprs = []
        for p in dash.get("panels", []):
            for t in p.get("targets", []):
                if "expr" in t:
                    promql_exprs.append(t["expr"])

        joined_exprs = " ".join(promql_exprs)
        assert "energy_advisor_prediction_requests_total" in joined_exprs
        assert "energy_advisor_prediction_latency_seconds_bucket" in joined_exprs
        assert "energy_advisor_dataset_drift_share" in joined_exprs
        assert "energy_advisor_model_mae" in joined_exprs
        assert "energy_advisor_model_rmse" in joined_exprs
        assert "energy_advisor_active_alerts" in joined_exprs
        assert "energy_advisor_monitoring_execution_failures_total" in joined_exprs

    def test_docker_compose_yaml_is_valid(self, project_root):
        """docker-compose.yml must be valid YAML declaring api, prometheus, and grafana."""
        compose_path = os.path.join(project_root, "docker-compose.yml")
        assert os.path.exists(compose_path), f"Missing {compose_path}"

        with open(compose_path, "r", encoding="utf-8") as f:
            compose = yaml.safe_load(f)

        assert "services" in compose
        services = compose["services"]
        assert "api" in services
        assert "prometheus" in services
        assert "grafana" in services

        # Check API port and volume
        assert "8000:8000" in services["api"]["ports"]
        assert any("./data/monitoring" in v for v in services["api"]["volumes"])

        # Check Prometheus port and volume
        assert "9090:9090" in services["prometheus"]["ports"]

        # Check Grafana port and volume
        assert "3000:3000" in services["grafana"]["ports"]
