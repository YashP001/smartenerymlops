"""
Unit and integration tests for Automated Monitoring and Alerting (Phase 3.5).

Covers:
- Drift alert creation, severity rules, and resolution.
- Performance degradation alerts and strict suppression on insufficient feedback.
- Monitoring execution failures (distinct from model/drift alerts).
- Deduplication and configurable cooldown behavior.
- Multi-model version alert isolation.
- Alert lifecycle (ACTIVE -> RESOLVED) and history queries.
- Notification formatting (plain text, markdown, dict).
- Edge cases: invalid payloads, database connection failures.
- Monitoring orchestrator end-to-end execution.
"""

import json
import os
import sqlite3
import numpy as np
import pandas as pd
import pytest

from src.monitoring.database import (
    init_db,
    log_prediction,
    log_feedback,
    get_alert,
    get_active_alerts,
    get_alert_history,
)
from src.monitoring.alert_manager import (
    AlertManager,
    Alert,
    AlertType,
    AlertSeverity,
    AlertStatus,
    NotificationMessage,
)
from src.monitoring.monitor_orchestrator import run_monitoring_cycle


# ─── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_db(tmp_path):
    """Create a temporary initialized SQLite database."""
    db_file = str(tmp_path / "test_monitoring.db")
    init_db(db_file)
    return db_file


@pytest.fixture
def alert_mgr(mock_db):
    """Instantiate AlertManager with the temporary test database."""
    return AlertManager(db_path=mock_db, default_cooldown_seconds=3600)


@pytest.fixture
def sample_drift_result():
    """Deterministic sample drift detection result."""
    return {
        "timestamp": "2026-11-01T12:00:00+00:00",
        "reference_samples": 1000,
        "production_samples": 200,
        "dataset_drift_detected": True,
        "drift_share": 0.6667,
        "drift_share_threshold": 0.50,
        "total_features_count": 6,
        "drifted_features_count": 4,
        "drifted_features": [
            "energy_current_hour",
            "energy_previous_hour",
            "energy_two_hours_ago",
            "target_hour",
        ],
        "feature_metrics": {},
    }


@pytest.fixture
def sample_healthy_drift_result():
    """Deterministic sample drift result with no drift."""
    return {
        "timestamp": "2026-11-01T12:00:00+00:00",
        "reference_samples": 1000,
        "production_samples": 200,
        "dataset_drift_detected": False,
        "drift_share": 0.1667,
        "drift_share_threshold": 0.50,
        "total_features_count": 6,
        "drifted_features_count": 1,
        "drifted_features": ["target_hour"],
        "feature_metrics": {},
    }


@pytest.fixture
def sample_degraded_perf_result():
    """Deterministic sample performance result with degraded accuracy."""
    return {
        "evaluation_timestamp": "2026-11-01T12:00:00+00:00",
        "model_version": "GradientBoostingRegressor",
        "status": "degraded",
        "is_degraded": True,
        "total_eligible_feedback_records": 50,
        "evaluated_predictions": 50,
        "metrics": {
            "mae": 0.1150,
            "rmse": 0.2200,
            "r2": 0.05,
            "bias": 0.01,
            "sample_count": 50,
        },
        "degradation_assessment": {
            "status": "degraded",
            "is_degraded": True,
            "reasons": [
                "MAE (0.1150 kWh) increased by 55.6% over baseline (0.0739 kWh), exceeding 25.0% threshold.",
                "R² score (0.0500) dropped below minimum acceptable threshold of 0.10.",
            ],
            "baseline_used": {
                "MAE": 0.0739,
                "RMSE": 0.1512,
                "R2": 0.3723,
                "BIAS": 0.0,
            },
        },
        "thresholds": {"min_samples": 30},
    }


@pytest.fixture
def sample_healthy_perf_result():
    """Deterministic sample performance result within baseline bounds."""
    return {
        "evaluation_timestamp": "2026-11-01T12:00:00+00:00",
        "model_version": "GradientBoostingRegressor",
        "status": "healthy",
        "is_degraded": False,
        "total_eligible_feedback_records": 50,
        "evaluated_predictions": 50,
        "metrics": {
            "mae": 0.0650,
            "rmse": 0.1300,
            "r2": 0.4200,
            "bias": 0.005,
            "sample_count": 50,
        },
        "degradation_assessment": {
            "status": "healthy",
            "is_degraded": False,
            "reasons": [],
            "baseline_used": {
                "MAE": 0.0739,
                "RMSE": 0.1512,
                "R2": 0.3723,
                "BIAS": 0.0,
            },
        },
        "thresholds": {"min_samples": 30},
    }


# ─── 1. Drift Alert Tests ─────────────────────────────────────────────

class TestDriftAlerts:
    """Tests for data drift alert triggering, severity, and resolution."""

    def test_drift_alert_created_when_dataset_drift_detected(self, alert_mgr, sample_drift_result):
        """Dataset drift creates an active alert with WARNING severity and details."""
        alert, notif = alert_mgr.evaluate_drift_result(sample_drift_result)
        assert alert is not None
        assert alert.alert_type == AlertType.DATA_DRIFT.value
        assert alert.severity == AlertSeverity.WARNING.value
        assert alert.status == AlertStatus.ACTIVE.value
        assert "Data drift detected: 4 feature(s) drifted (66.7% share >= 50.0% threshold)" in alert.message
        assert alert.metrics["drifted_features_count"] == 4
        assert alert.metrics["drift_share"] == 0.6667
        assert alert.trigger_count == 1

        assert notif is not None
        assert notif.event_type == "triggered"
        assert notif.severity == AlertSeverity.WARNING.value

        # Verify active alert persisted in database
        active = alert_mgr.get_active_alerts(alert_type=AlertType.DATA_DRIFT.value)
        assert len(active) == 1
        assert active[0]["alert_id"] == alert.alert_id

    def test_critical_drift_severity_when_drift_share_high(self, alert_mgr, sample_drift_result):
        """Drift share >= 75% escalates severity to CRITICAL."""
        sample_drift_result["drift_share"] = 0.8333
        sample_drift_result["drifted_features_count"] = 5
        alert, notif = alert_mgr.evaluate_drift_result(sample_drift_result)
        assert alert.severity == AlertSeverity.CRITICAL.value
        assert notif.severity == AlertSeverity.CRITICAL.value

    def test_no_drift_alert_when_dataset_drift_false(self, alert_mgr, sample_healthy_drift_result):
        """When dataset drift is false and no alert was active, no alert is created."""
        alert, notif = alert_mgr.evaluate_drift_result(sample_healthy_drift_result)
        assert alert is None
        assert notif is None
        assert len(alert_mgr.get_active_alerts(alert_type=AlertType.DATA_DRIFT.value)) == 0

    def test_drift_alert_resolution(self, alert_mgr, sample_drift_result, sample_healthy_drift_result):
        """Subsequent healthy drift run transitions existing active drift alert to RESOLVED."""
        # Step 1: Trigger active drift alert
        alert1, notif1 = alert_mgr.evaluate_drift_result(sample_drift_result)
        assert alert1.status == AlertStatus.ACTIVE.value
        assert len(alert_mgr.get_active_alerts(alert_type=AlertType.DATA_DRIFT.value)) == 1

        # Step 2: Healthy drift run resolves it
        alert2, notif2 = alert_mgr.evaluate_drift_result(sample_healthy_drift_result)
        assert alert2 is not None
        assert alert2.alert_id == alert1.alert_id
        assert alert2.status == AlertStatus.RESOLVED.value
        assert alert2.resolved_at is not None
        assert notif2.event_type == "resolved"

        # Step 3: Database confirms zero active drift alerts and one resolved alert
        active = alert_mgr.get_active_alerts(alert_type=AlertType.DATA_DRIFT.value)
        assert len(active) == 0
        history = alert_mgr.get_alert_history()
        assert len(history) == 1
        assert history[0]["status"] == AlertStatus.RESOLVED.value


# ─── 2. Model Performance Degradation Alert Tests ──────────────────────

class TestPerformanceDegradationAlerts:
    """Tests for model degradation alerts, feedback guardrails, and resolution."""

    def test_degradation_alert_created_when_is_degraded(self, alert_mgr, sample_degraded_perf_result):
        """Degraded performance generates an active alert with metric reasons."""
        alert, notif = alert_mgr.evaluate_performance_result(sample_degraded_perf_result)
        assert alert is not None
        assert alert.alert_type == AlertType.MODEL_DEGRADATION.value
        assert alert.status == AlertStatus.ACTIVE.value
        assert alert.model_version == "GradientBoostingRegressor"
        assert "MAE (0.1150 kWh) increased by 55.6%" in alert.message
        assert notif.event_type == "triggered"

    def test_insufficient_feedback_suppresses_degradation_alert(self, alert_mgr):
        """insufficient_data status strictly suppresses degradation alert creation."""
        perf_result = {
            "evaluation_timestamp": "2026-11-01T12:00:00+00:00",
            "model_version": "GradientBoostingRegressor",
            "status": "insufficient_data",
            "is_degraded": False,
            "total_eligible_feedback_records": 10,
            "evaluated_predictions": 10,
            "metrics": {"mae": 0.12, "sample_count": 10},
        }
        alert, notif = alert_mgr.evaluate_performance_result(perf_result)
        assert alert is None
        assert notif is None
        assert len(alert_mgr.get_active_alerts()) == 0

    def test_no_feedback_data_does_not_create_alert(self, alert_mgr):
        """no_feedback_data status creates no alert."""
        perf_result = {
            "status": "no_feedback_data",
            "is_degraded": False,
            "model_version": "GradientBoostingRegressor",
        }
        alert, notif = alert_mgr.evaluate_performance_result(perf_result)
        assert alert is None
        assert notif is None

    def test_critical_severity_on_severe_degradation(self, alert_mgr, sample_degraded_perf_result):
        """Severe degradation (MAE increase >= 50% or R² < 0) escalates to CRITICAL."""
        # sample_degraded_perf_result has MAE 0.1150 vs baseline 0.0739 (+55.6% increase)
        alert, notif = alert_mgr.evaluate_performance_result(sample_degraded_perf_result)
        assert alert.severity == AlertSeverity.CRITICAL.value

    def test_warning_severity_on_moderate_degradation(self, alert_mgr):
        """Moderate degradation (< 50% increase) assigns WARNING severity."""
        moderate_perf = {
            "status": "degraded",
            "is_degraded": True,
            "model_version": "GradientBoostingRegressor",
            "metrics": {
                "mae": 0.0950,  # ~28.5% increase over 0.0739 (exceeds 25% threshold, but < 50%)
                "rmse": 0.1600,
                "r2": 0.25,
                "bias": 0.01,
                "sample_count": 50,
            },
            "degradation_assessment": {
                "reasons": ["MAE increased by 28.5%."],
                "baseline_used": {"MAE": 0.0739, "RMSE": 0.1512},
            },
        }
        alert, notif = alert_mgr.evaluate_performance_result(moderate_perf)
        assert alert.severity == AlertSeverity.WARNING.value

    def test_performance_alert_resolution_on_healthy(
        self,
        alert_mgr,
        sample_degraded_perf_result,
        sample_healthy_perf_result,
    ):
        """Healthy performance evaluation resolves existing active degradation alert."""
        # Step 1: Trigger degradation alert
        alert1, notif1 = alert_mgr.evaluate_performance_result(sample_degraded_perf_result)
        assert alert1.status == AlertStatus.ACTIVE.value

        # Step 2: Healthy evaluation resolves it
        alert2, notif2 = alert_mgr.evaluate_performance_result(sample_healthy_perf_result)
        assert alert2 is not None
        assert alert2.status == AlertStatus.RESOLVED.value
        assert alert2.alert_id == alert1.alert_id
        assert notif2.event_type == "resolved"
        assert len(alert_mgr.get_active_alerts(model_version="GradientBoostingRegressor")) == 0

    def test_insufficient_feedback_does_not_resolve_active_alert(
        self,
        alert_mgr,
        sample_degraded_perf_result,
    ):
        """insufficient_data evaluation does NOT resolve an existing active alert."""
        # Step 1: Trigger degradation alert
        alert1, notif1 = alert_mgr.evaluate_performance_result(sample_degraded_perf_result)
        assert alert1.status == AlertStatus.ACTIVE.value

        # Step 2: Insufficient data evaluation occurs
        insufficient_perf = {
            "status": "insufficient_data",
            "is_degraded": False,
            "model_version": "GradientBoostingRegressor",
            "metrics": {"sample_count": 5},
        }
        alert2, notif2 = alert_mgr.evaluate_performance_result(insufficient_perf)
        assert alert2 is None
        assert notif2 is None

        # Verify alert remains active
        active = alert_mgr.get_active_alerts(model_version="GradientBoostingRegressor")
        assert len(active) == 1
        assert active[0]["status"] == AlertStatus.ACTIVE.value


# ─── 3. Monitoring Execution Failure Tests ─────────────────────────────

class TestMonitoringExecutionFailures:
    """Tests for recording pipeline execution failures."""

    def test_monitoring_failure_alert_creation(self, alert_mgr):
        """Execution failure records an alert with ERROR severity."""
        exc = sqlite3.OperationalError("database is locked")
        alert, notif = alert_mgr.record_execution_failure(
            operation_name="performance_monitor_query",
            error=exc,
            model_version="GradientBoostingRegressor",
        )
        assert alert.alert_type == AlertType.MONITORING_EXECUTION_FAILURE.value
        assert alert.severity == AlertSeverity.ERROR.value
        assert "OperationalError" in alert.message
        assert alert.metrics["operation"] == "performance_monitor_query"
        assert notif.event_type == "failure"

    def test_failure_distinct_from_drift_and_degradation(self, alert_mgr):
        """Execution failure is categorized separately from drift and model degradation."""
        alert, _ = alert_mgr.record_execution_failure(
            operation_name="data_loader",
            error=FileNotFoundError("missing file"),
        )
        assert alert.alert_type != AlertType.DATA_DRIFT.value
        assert alert.alert_type != AlertType.MODEL_DEGRADATION.value
        assert alert.alert_type == AlertType.MONITORING_EXECUTION_FAILURE.value

    def test_drift_failure_followed_by_successful_drift_recovery(self, alert_mgr, sample_healthy_drift_result):
        """Drift failure is automatically resolved when subsequent drift evaluation succeeds."""
        # 1. Record drift execution failure
        alert_fail, notif_fail = alert_mgr.record_execution_failure("drift_detection", "CSV read error")
        assert alert_fail.status == AlertStatus.ACTIVE.value
        active_fails = alert_mgr.get_active_alerts(alert_type=AlertType.MONITORING_EXECUTION_FAILURE.value)
        assert len(active_fails) == 1

        # 2. Subsequent successful drift evaluation occurs
        alert_mgr.evaluate_drift_result(sample_healthy_drift_result)

        # 3. Verify the drift failure alert is now resolved
        active_fails_after = alert_mgr.get_active_alerts(alert_type=AlertType.MONITORING_EXECUTION_FAILURE.value)
        assert len(active_fails_after) == 0

        fetched = alert_mgr.get_alert_by_id(alert_fail.alert_id)
        assert fetched["status"] == AlertStatus.RESOLVED.value
        assert fetched["resolved_at"] is not None

    def test_performance_failure_followed_by_successful_performance_recovery(self, alert_mgr, sample_healthy_perf_result):
        """Performance monitoring failure is automatically resolved when subsequent performance run succeeds."""
        # 1. Record performance execution failure
        alert_fail, notif_fail = alert_mgr.record_execution_failure("performance_monitoring", "DB query locked")
        assert alert_fail.status == AlertStatus.ACTIVE.value

        # 2. Subsequent successful performance evaluation occurs
        alert_mgr.evaluate_performance_result(sample_healthy_perf_result)

        # 3. Verify performance failure alert is now resolved
        active_fails = alert_mgr.get_active_alerts(alert_type=AlertType.MONITORING_EXECUTION_FAILURE.value)
        assert len(active_fails) == 0

        fetched = alert_mgr.get_alert_by_id(alert_fail.alert_id)
        assert fetched["status"] == AlertStatus.RESOLVED.value
        assert fetched["resolved_at"] is not None

    def test_drift_recovery_does_not_resolve_performance_failure(self, alert_mgr, sample_healthy_drift_result):
        """A successful drift run does not resolve an active performance monitoring failure."""
        alert_fail, _ = alert_mgr.record_execution_failure("performance_monitoring", "DB query locked")
        assert alert_fail.status == AlertStatus.ACTIVE.value

        # Drift succeeds
        alert_mgr.evaluate_drift_result(sample_healthy_drift_result)

        # Performance failure MUST remain ACTIVE
        active_fails = alert_mgr.get_active_alerts(alert_type=AlertType.MONITORING_EXECUTION_FAILURE.value)
        assert len(active_fails) == 1
        assert active_fails[0]["alert_id"] == alert_fail.alert_id
        assert active_fails[0]["status"] == AlertStatus.ACTIVE.value

    def test_performance_recovery_does_not_resolve_drift_failure(self, alert_mgr, sample_healthy_perf_result):
        """A successful performance run does not resolve an active drift monitoring failure."""
        alert_fail, _ = alert_mgr.record_execution_failure("drift_detection", "Reference file missing")
        assert alert_fail.status == AlertStatus.ACTIVE.value

        # Performance succeeds
        alert_mgr.evaluate_performance_result(sample_healthy_perf_result)

        # Drift failure MUST remain ACTIVE
        active_fails = alert_mgr.get_active_alerts(alert_type=AlertType.MONITORING_EXECUTION_FAILURE.value)
        assert len(active_fails) == 1
        assert active_fails[0]["alert_id"] == alert_fail.alert_id
        assert active_fails[0]["status"] == AlertStatus.ACTIVE.value

    def test_repeated_failure_followed_by_recovery_preserves_alert_history(self, alert_mgr):
        """Repeated failures increment trigger count and recovery preserves full history."""
        # Failure 1
        alert1, notif1 = alert_mgr.record_execution_failure("drift_detection", "Error 1")
        assert alert1.trigger_count == 1

        # Failure 2 (deduplicated within cooldown)
        alert2, notif2 = alert_mgr.record_execution_failure("drift_detection", "Error 2", cooldown_seconds=3600)
        assert alert2.alert_id == alert1.alert_id
        assert alert2.trigger_count == 2
        assert notif2.event_type == "suppressed"

        # Auto-recovery occurs
        res = alert_mgr.resolve_execution_failure("drift_detection")
        assert res is not None
        res_alert, res_notif = res
        assert res_alert.alert_id == alert1.alert_id
        assert res_alert.status == AlertStatus.RESOLVED.value
        assert res_alert.trigger_count == 2

        # Verify alert history
        history = alert_mgr.get_alert_history()
        assert len(history) == 1
        assert history[0]["alert_id"] == alert1.alert_id
        assert history[0]["trigger_count"] == 2
        assert history[0]["status"] == AlertStatus.RESOLVED.value
        assert history[0]["created_at"] == alert1.created_at
        assert history[0]["resolved_at"] is not None


# ─── 4. Deduplication and Cooldown Tests ───────────────────────────────

class TestDeduplicationAndCooldown:
    """Tests for duplicate alert suppression within cooldown windows."""

    def test_duplicate_alert_suppression_within_cooldown(self, alert_mgr, sample_drift_result):
        """Immediate duplicate drift trigger within cooldown is suppressed."""
        # Run 1: Creates new alert
        alert1, notif1 = alert_mgr.evaluate_drift_result(sample_drift_result, cooldown_seconds=3600)
        assert alert1.trigger_count == 1
        assert notif1.event_type == "triggered"

        # Run 2: Immediate duplicate trigger
        alert2, notif2 = alert_mgr.evaluate_drift_result(sample_drift_result, cooldown_seconds=3600)
        assert alert2.alert_id == alert1.alert_id
        assert alert2.trigger_count == 2
        assert notif2.event_type == "suppressed"

        # Only one active alert in database
        active = alert_mgr.get_active_alerts(alert_type=AlertType.DATA_DRIFT.value)
        assert len(active) == 1
        assert active[0]["trigger_count"] == 2

    def test_alert_reminder_after_cooldown_expired(self, alert_mgr, sample_drift_result):
        """Trigger occurring after cooldown expiration produces un-suppressed reminder."""
        # Cooldown of 0 seconds simulates expired cooldown
        alert1, notif1 = alert_mgr.evaluate_drift_result(sample_drift_result, cooldown_seconds=0)
        assert notif1.event_type == "triggered"

        alert2, notif2 = alert_mgr.evaluate_drift_result(sample_drift_result, cooldown_seconds=0)
        assert alert2.alert_id == alert1.alert_id
        assert alert2.trigger_count == 2
        assert notif2.event_type == "triggered"  # Cooldown elapsed, reminder emitted


# ─── 5. Multi-Model Version Separation Tests ───────────────────────────

class TestModelVersionSeparation:
    """Tests that alerts for different model versions do not interfere with each other."""

    def test_multiple_model_versions_isolated_alerts(self, alert_mgr, sample_degraded_perf_result):
        """Alerts for GradientBoostingRegressor and RandomForestRegressor remain isolated."""
        gbm_result = dict(sample_degraded_perf_result)
        gbm_result["model_version"] = "GradientBoostingRegressor"

        rf_result = dict(sample_degraded_perf_result)
        rf_result["model_version"] = "RandomForestRegressor"

        alert_gbm, _ = alert_mgr.evaluate_performance_result(gbm_result)
        alert_rf, _ = alert_mgr.evaluate_performance_result(rf_result)

        assert alert_gbm.alert_id != alert_rf.alert_id
        assert alert_gbm.model_version == "GradientBoostingRegressor"
        assert alert_rf.model_version == "RandomForestRegressor"

        # Resolving GBM does not resolve RF
        healthy_gbm = {
            "status": "healthy",
            "is_degraded": False,
            "model_version": "GradientBoostingRegressor",
            "metrics": {"mae": 0.05},
        }
        alert_mgr.evaluate_performance_result(healthy_gbm)

        active_gbm = alert_mgr.get_active_alerts(model_version="GradientBoostingRegressor")
        active_rf = alert_mgr.get_active_alerts(model_version="RandomForestRegressor")
        assert len(active_gbm) == 0
        assert len(active_rf) == 1


# ─── 6. Alert History & Lifecycle Query Tests ──────────────────────────

class TestAlertHistoryAndQueries:
    """Tests for alert retrieval, history preservation, and manual resolution."""

    def test_alert_history_retrieval(self, alert_mgr, sample_drift_result, sample_healthy_drift_result):
        """Alert history preserves both active and resolved alerts."""
        alert_mgr.evaluate_drift_result(sample_drift_result)
        alert_mgr.evaluate_drift_result(sample_healthy_drift_result)  # resolves drift

        # Add another failure alert
        alert_mgr.record_execution_failure("eval_pipeline", "timeout error")

        history = alert_mgr.get_alert_history(limit=50)
        assert len(history) == 2
        statuses = {a["status"] for a in history}
        assert AlertStatus.RESOLVED.value in statuses
        assert AlertStatus.ACTIVE.value in statuses

    def test_get_alert_by_id(self, alert_mgr, sample_drift_result):
        """get_alert_by_id retrieves specific alert record with parsed JSON fields."""
        alert, _ = alert_mgr.evaluate_drift_result(sample_drift_result)
        fetched = alert_mgr.get_alert_by_id(alert.alert_id)
        assert fetched is not None
        assert fetched["alert_id"] == alert.alert_id
        assert isinstance(fetched["metrics"], dict)
        assert fetched["metrics"]["drift_share"] == 0.6667

    def test_manual_resolution(self, alert_mgr, sample_drift_result):
        """resolve_alert_by_id transitions active alert to resolved status."""
        alert, _ = alert_mgr.evaluate_drift_result(sample_drift_result)
        success = alert_mgr.resolve_alert_by_id(alert.alert_id)
        assert success is True
        fetched = alert_mgr.get_alert_by_id(alert.alert_id)
        assert fetched["status"] == AlertStatus.RESOLVED.value
        assert fetched["resolved_at"] is not None


# ─── 7. Notification Formatting Tests ─────────────────────────────────

class TestNotificationFormatting:
    """Tests for NotificationMessage formatting."""

    def test_notification_text_and_markdown_formatting(self):
        """NotificationMessage formats to text, markdown, and dictionary."""
        notif = NotificationMessage(
            title="Data Drift Alert",
            severity="WARNING",
            event_type="triggered",
            timestamp="2026-11-01T12:00:00+00:00",
            body="3 features drifted exceeding threshold.",
            metrics_summary="drift_share=0.50",
            recommended_action="Inspect incoming telemetry sensors.",
            alert_id="alert_123",
            model_version="GradientBoostingRegressor",
        )

        text = notif.to_text()
        assert "[WARNING] Data Drift Alert" in text
        assert "Event: TRIGGERED" in text
        assert "Model Version: GradientBoostingRegressor" in text
        assert "Action: Inspect incoming telemetry sensors." in text

        md = notif.to_markdown()
        assert "### **[WARNING] WARNING** (TRIGGERED)" in md
        assert "- **Model Version:** `GradientBoostingRegressor`" in md
        assert "**Recommended Action:** Inspect incoming telemetry sensors." in md

        d = notif.to_dict()
        assert d["alert_id"] == "alert_123"
        assert d["severity"] == "WARNING"


# ─── 8. Robustness & Safe Execution Tests ──────────────────────────────

class TestRobustnessAndEdgeCases:
    """Tests handling of invalid inputs and database errors."""

    def test_invalid_payloads_safe(self, alert_mgr):
        """Non-dictionary or empty inputs return None without raising unhandled exceptions."""
        assert alert_mgr.evaluate_drift_result(None) == (None, None)
        assert alert_mgr.evaluate_drift_result("not-a-dict") == (None, None)
        assert alert_mgr.evaluate_performance_result(None) == (None, None)
        assert alert_mgr.evaluate_performance_result([]) == (None, None)

    def test_database_failure_handling(self, tmp_path):
        """Operations against a non-existent or invalid DB handle gracefully."""
        invalid_path = str(tmp_path / "nonexistent_dir" / "bad.db")
        # Creating AlertManager initializes tables in directory if possible;
        # if directory does not exist, init_db creates parent directory safely.
        mgr = AlertManager(db_path=invalid_path)
        assert os.path.exists(invalid_path)


# ─── 9. Orchestrator End-to-End Tests ──────────────────────────────────

class TestMonitorOrchestrator:
    """Tests end-to-end monitoring execution cycle."""

    def test_orchestrator_end_to_end_cycle(self, mock_db):
        """Orchestrator runs drift + performance cycle and emits structured summary."""
        # Seed 40 predictions and feedback into mock_db
        base_dt = pd.to_datetime("2026-11-01T00:00:00Z")
        for i in range(40):
            pred_id = f"pred_orch_{i:03d}"
            actual = 0.20 + (i % 5) * 0.05
            pred = actual + (0.02 if i % 2 == 0 else -0.02)
            ts = (base_dt + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            log_prediction(
                prediction_id=pred_id,
                features={
                    "energy_current_hour": actual,
                    "energy_previous_hour": actual,
                    "energy_two_hours_ago": actual,
                    "target_hour": 12,
                    "target_day_of_week": 2,
                    "target_is_weekend": 0,
                },
                predicted_energy_kwh=pred,
                model_version="GradientBoostingRegressor",
                timestamp=ts,
                db_path=mock_db,
            )
            log_feedback(
                prediction_id=pred_id,
                actual_energy_kwh=actual,
                feedback_timestamp=ts,
                db_path=mock_db,
            )

        # Run cycle (without prod_data skips drift, evaluates performance)
        summary = run_monitoring_cycle(
            db_path=mock_db,
            model_version="GradientBoostingRegressor",
        )

        assert "cycle_id" in summary
        assert summary["drift_status"] == "skipped"
        assert summary["performance_status"] == "evaluated"
        assert summary["performance_result"]["status"] == "healthy"
        assert len(summary["execution_failures"]) == 0
        assert summary["active_alerts_count"] == 0

    def test_orchestrator_records_drift_failure_safely(self, mock_db):
        """Missing or corrupted drift data records execution failure without crashing."""
        summary = run_monitoring_cycle(
            ref_data="nonexistent_ref_path_xyz.csv",
            prod_data="nonexistent_prod_path_xyz.csv",
            db_path=mock_db,
        )
        assert summary["drift_status"] == "failed"
        assert len(summary["execution_failures"]) == 1
        assert summary["execution_failures"][0]["alert_type"] == AlertType.MONITORING_EXECUTION_FAILURE.value
