"""
Alert Evaluation and Lifecycle Engine for Smart Home Energy Advisor (Phase 3.5).

Consumes monitoring results from:
- Data Drift Detection Engine (src/monitoring/drift_detector.py)
- Model Performance Monitor (src/monitoring/performance_monitor.py)
- Monitoring Orchestrator execution failures

Manages:
- Structured alert generation and severity assignment.
- Alert lifecycle: ACTIVE vs RESOLVED transitions.
- Deduplication and configurable cooldown periods.
- Full chronological alert history preservation in SQLite.
- Provider-independent, notification-ready message formatting.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

# Ensure project root is in path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.monitoring.database import (
    get_db_path,
    init_db,
    save_alert,
    get_alert,
    get_active_alerts,
    resolve_alert as db_resolve_alert,
    get_alert_history,
)

logger = logging.getLogger("monitoring.alert_manager")

DEFAULT_COOLDOWN_SECONDS = 3600  # 1 hour


# ─── Enumerations & Data Structures ──────────────────────────────────

class AlertType(str, Enum):
    """Supported alert classification categories."""
    DATA_DRIFT = "data_drift"
    MODEL_DEGRADATION = "model_degradation"
    MONITORING_EXECUTION_FAILURE = "monitoring_execution_failure"


class AlertSeverity(str, Enum):
    """Standardized alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"


class AlertStatus(str, Enum):
    """Alert lifecycle states."""
    ACTIVE = "active"
    RESOLVED = "resolved"


@dataclass
class Alert:
    """Structured representation of an operational monitoring alert."""
    alert_id: str
    alert_type: str
    severity: str
    status: str
    model_version: Optional[str]
    message: str
    metrics: Dict[str, Any]
    source_monitoring_result: Dict[str, Any]
    created_at: str
    updated_at: str
    resolved_at: Optional[str] = None
    trigger_count: int = 1
    last_triggered_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to serializable dictionary."""
        d = asdict(self)
        if self.last_triggered_at is None:
            d["last_triggered_at"] = self.updated_at
        return d


@dataclass
class NotificationMessage:
    """Provider-independent structured alert notification payload."""
    title: str
    severity: str
    event_type: str  # "triggered", "suppressed", "resolved", "failure"
    timestamp: str
    body: str
    metrics_summary: str
    recommended_action: str
    alert_id: Optional[str] = None
    model_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert notification to dictionary."""
        return asdict(self)

    def to_text(self) -> str:
        """Format notification as human-readable plain text."""
        lines = [
            f"[{self.severity}] {self.title}",
            f"Event: {self.event_type.upper()} | Time: {self.timestamp}",
        ]
        if self.model_version:
            lines.append(f"Model Version: {self.model_version}")
        if self.alert_id:
            lines.append(f"Alert ID: {self.alert_id}")
        lines.append("")
        lines.append(f"Summary: {self.body}")
        if self.metrics_summary:
            lines.append(f"Metrics: {self.metrics_summary}")
        lines.append(f"Action: {self.recommended_action}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Format notification as rich GitHub-flavored markdown."""
        icon = {
            "INFO": "[INFO]",
            "WARNING": "[WARNING]",
            "CRITICAL": "[CRITICAL]",
            "ERROR": "[ERROR]",
        }.get(self.severity, "[ALERT]")

        badge = f"**{icon} {self.severity}** ({self.event_type.upper()})"
        md = [
            f"### {badge} {self.title}",
            f"- **Timestamp:** `{self.timestamp}`",
        ]
        if self.model_version:
            md.append(f"- **Model Version:** `{self.model_version}`")
        if self.alert_id:
            md.append(f"- **Alert ID:** `{self.alert_id}`")
        md.append("")
        md.append(f"{self.body}\n")
        if self.metrics_summary:
            md.append(f"> **Metrics:**\n> {self.metrics_summary}\n")
        md.append(f"**Recommended Action:** {self.recommended_action}")
        return "\n".join(md)


# ─── Alert Evaluation Engine ──────────────────────────────────────────

class AlertManager:
    """Consumes monitoring outputs, evaluates alert rules, and manages alert lifecycle."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        default_cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS,
    ):
        """Initialize the AlertManager.

        Args:
            db_path: Path to monitoring SQLite database.
            default_cooldown_seconds: Default cooldown suppression window in seconds.
        """
        self.db_path = db_path or get_db_path()
        self.default_cooldown_seconds = default_cooldown_seconds
        # Ensure database tables exist
        init_db(self.db_path)

    # ─── Evaluation Methods ───────────────────────────────────────────

    def evaluate_drift_result(
        self,
        drift_result: Dict[str, Any],
        cooldown_seconds: Optional[int] = None,
    ) -> Tuple[Optional[Alert], Optional[NotificationMessage]]:
        """Evaluate a data drift detection summary and update alert state.

        Args:
            drift_result: Summary dictionary emitted by run_drift_analysis().
            cooldown_seconds: Optional cooldown period override.

        Returns:
            Tuple of (Alert or None, NotificationMessage or None).
        """
        if not isinstance(drift_result, dict):
            logger.warning("Invalid drift result passed to evaluate_drift_result: not a dictionary")
            return None, None

        dataset_drift = bool(drift_result.get("dataset_drift_detected", False))
        cooldown = cooldown_seconds if cooldown_seconds is not None else self.default_cooldown_seconds
        now_iso = datetime.now(timezone.utc).isoformat()

        # Automatic recovery: A successful drift result resolves prior drift_detection execution failure
        self.resolve_execution_failure("drift_detection")

        # Check existing active drift alert
        active_alerts = get_active_alerts(db_path=self.db_path, alert_type=AlertType.DATA_DRIFT.value)
        existing_alert = active_alerts[0] if active_alerts else None

        if dataset_drift:
            drifted_features = drift_result.get("drifted_features", [])
            drift_share = float(drift_result.get("drift_share", 0.0))
            threshold = float(drift_result.get("drift_share_threshold", 0.50))

            # Severity determination:
            # If >= 75% features drifted or target drift flagged -> CRITICAL; else WARNING
            target_drift_detected = False
            if "target_drift" in drift_result and isinstance(drift_result["target_drift"], dict):
                target_drift_detected = drift_result["target_drift"].get("target_drift_detected", False)

            if drift_share >= 0.75 or target_drift_detected:
                severity = AlertSeverity.CRITICAL.value
            else:
                severity = AlertSeverity.WARNING.value

            features_str = ", ".join(drifted_features) if drifted_features else "none"
            msg = (
                f"Data drift detected: {len(drifted_features)} feature(s) drifted "
                f"({drift_share * 100:.1f}% share >= {threshold * 100:.1f}% threshold). "
                f"Drifted features: [{features_str}]."
            )
            metrics_summary = (
                f"drift_share={drift_share:.2f}, threshold={threshold:.2f}, "
                f"drifted_count={len(drifted_features)}, total_features={drift_result.get('total_features_count', 0)}"
            )

            metrics_payload = {
                "drift_share": drift_share,
                "drift_share_threshold": threshold,
                "drifted_features": drifted_features,
                "drifted_features_count": len(drifted_features),
                "total_features_count": drift_result.get("total_features_count", 0),
                "target_drift_detected": target_drift_detected,
            }

            if existing_alert:
                # Deduplication and cooldown check
                last_trig = existing_alert.get("last_triggered_at") or existing_alert["updated_at"]
                elapsed = self._calculate_elapsed_seconds(last_trig, now_iso)

                trigger_count = int(existing_alert.get("trigger_count", 1)) + 1
                updated_alert = Alert(
                    alert_id=existing_alert["alert_id"],
                    alert_type=AlertType.DATA_DRIFT.value,
                    severity=severity,
                    status=AlertStatus.ACTIVE.value,
                    model_version=None,
                    message=msg,
                    metrics=metrics_payload,
                    source_monitoring_result=drift_result,
                    created_at=existing_alert["created_at"],
                    updated_at=now_iso,
                    resolved_at=None,
                    trigger_count=trigger_count,
                    last_triggered_at=now_iso,
                )
                save_alert(updated_alert.to_dict(), db_path=self.db_path)

                if elapsed < cooldown:
                    # Suppressed notification within cooldown window
                    notif = NotificationMessage(
                        title="Data Drift Detected (Recurring - Notification Suppressed)",
                        severity=severity,
                        event_type="suppressed",
                        timestamp=now_iso,
                        body=f"{msg} (Occurrences: {trigger_count}, within cooldown of {cooldown}s).",
                        metrics_summary=metrics_summary,
                        recommended_action="Monitor feature distributions; model retraining may be required if drift persists.",
                        alert_id=updated_alert.alert_id,
                    )
                    return updated_alert, notif
                else:
                    # Cooldown expired: produce full triggered notification
                    notif = NotificationMessage(
                        title="Data Drift Alert (Recurring)",
                        severity=severity,
                        event_type="triggered",
                        timestamp=now_iso,
                        body=f"{msg} (Occurrences: {trigger_count}).",
                        metrics_summary=metrics_summary,
                        recommended_action="Inspect incoming telemetry sensors for data distribution shift or HVAC usage shifts.",
                        alert_id=updated_alert.alert_id,
                    )
                    return updated_alert, notif
            else:
                # Create brand new active drift alert
                new_alert_id = f"alert_drift_{uuid.uuid4().hex[:10]}"
                new_alert = Alert(
                    alert_id=new_alert_id,
                    alert_type=AlertType.DATA_DRIFT.value,
                    severity=severity,
                    status=AlertStatus.ACTIVE.value,
                    model_version=None,
                    message=msg,
                    metrics=metrics_payload,
                    source_monitoring_result=drift_result,
                    created_at=now_iso,
                    updated_at=now_iso,
                    resolved_at=None,
                    trigger_count=1,
                    last_triggered_at=now_iso,
                )
                save_alert(new_alert.to_dict(), db_path=self.db_path)

                notif = NotificationMessage(
                    title="Data Drift Alert Triggered",
                    severity=severity,
                    event_type="triggered",
                    timestamp=now_iso,
                    body=msg,
                    metrics_summary=metrics_summary,
                    recommended_action="Evaluate distribution shifts across production features and plan reference re-baselining.",
                    alert_id=new_alert_id,
                )
                return new_alert, notif

        else:
            # No dataset drift detected: check if an active drift alert should be resolved
            if existing_alert:
                db_resolve_alert(existing_alert["alert_id"], resolved_at=now_iso, db_path=self.db_path)
                resolved_alert = Alert(
                    alert_id=existing_alert["alert_id"],
                    alert_type=AlertType.DATA_DRIFT.value,
                    severity=AlertSeverity.INFO.value,
                    status=AlertStatus.RESOLVED.value,
                    model_version=None,
                    message="Data drift condition resolved: dataset distribution returned to reference bounds.",
                    metrics=drift_result.get("feature_metrics", {}),
                    source_monitoring_result=drift_result,
                    created_at=existing_alert["created_at"],
                    updated_at=now_iso,
                    resolved_at=now_iso,
                    trigger_count=existing_alert.get("trigger_count", 1),
                    last_triggered_at=existing_alert.get("last_triggered_at", existing_alert["updated_at"]),
                )
                notif = NotificationMessage(
                    title="Data Drift Alert Resolved",
                    severity=AlertSeverity.INFO.value,
                    event_type="resolved",
                    timestamp=now_iso,
                    body="Drift analysis verified dataset features have returned within expected statistical thresholds.",
                    metrics_summary=f"drift_share={drift_result.get('drift_share', 0.0):.2f} < threshold",
                    recommended_action="Normal monitoring operational state restored.",
                    alert_id=existing_alert["alert_id"],
                )
                return resolved_alert, notif

            return None, None

    def evaluate_performance_result(
        self,
        perf_result: Dict[str, Any],
        model_version: Optional[str] = None,
        cooldown_seconds: Optional[int] = None,
    ) -> Tuple[Optional[Alert], Optional[NotificationMessage]]:
        """Evaluate a model performance monitoring summary and update alert state.

        Args:
            perf_result: Summary dictionary emitted by evaluate_model_performance().
            model_version: Optional explicit model version override.
            cooldown_seconds: Optional cooldown period override.

        Returns:
            Tuple of (Alert or None, NotificationMessage or None).
        """
        if not isinstance(perf_result, dict):
            logger.warning("Invalid performance result passed to evaluate_performance_result")
            return None, None

        status = perf_result.get("status", "unknown")
        is_degraded = bool(perf_result.get("is_degraded", False))
        m_version = model_version or perf_result.get("model_version") or "all_models"
        cooldown = cooldown_seconds if cooldown_seconds is not None else self.default_cooldown_seconds
        now_iso = datetime.now(timezone.utc).isoformat()

        # Automatic recovery: A successful performance result resolves prior performance_monitoring execution failure
        self.resolve_execution_failure("performance_monitoring")

        # Rule: Insufficient or missing feedback NEVER creates a degradation alert
        if status in ("insufficient_data", "no_feedback_data"):
            logger.info("Performance monitoring reported '%s' for %s: skipping alert creation.", status, m_version)
            return None, None

        # Check existing active performance alerts for this model version
        active_alerts = get_active_alerts(
            db_path=self.db_path,
            alert_type=AlertType.MODEL_DEGRADATION.value,
            model_version=m_version,
        )
        existing_alert = active_alerts[0] if active_alerts else None

        if is_degraded:
            assessment = perf_result.get("degradation_assessment", {})
            reasons = assessment.get("reasons", ["Performance metric breached baseline threshold."])
            metrics = perf_result.get("metrics") or {}

            # Severity determination:
            # If RMSE or MAE increased > 50% or R² < 0.0 -> CRITICAL, else WARNING
            baseline = assessment.get("baseline_used", {})
            mae = metrics.get("mae", 0.0)
            rmse = metrics.get("rmse", 0.0)
            r2 = metrics.get("r2")
            base_mae = baseline.get("MAE", 0.0739)
            base_rmse = baseline.get("RMSE", 0.1512)

            mae_growth = (mae - base_mae) / base_mae if base_mae > 0 else 0.0
            rmse_growth = (rmse - base_rmse) / base_rmse if base_rmse > 0 else 0.0

            if mae_growth >= 0.50 or rmse_growth >= 0.50 or (r2 is not None and r2 < 0.0):
                severity = AlertSeverity.CRITICAL.value
            else:
                severity = AlertSeverity.WARNING.value

            reasons_str = "; ".join(reasons)
            msg = f"Model performance degraded for '{m_version}': {reasons_str}"
            metrics_summary = (
                f"MAE={mae} (base={base_mae}), RMSE={rmse} (base={base_rmse}), "
                f"R²={r2}, bias={metrics.get('bias')}, samples={metrics.get('sample_count')}"
            )

            metrics_payload = {
                "metrics": metrics,
                "baseline": baseline,
                "reasons": reasons,
                "thresholds": perf_result.get("thresholds", {}),
                "evaluated_predictions": perf_result.get("evaluated_predictions", 0),
            }

            if existing_alert:
                # Deduplication and cooldown
                last_trig = existing_alert.get("last_triggered_at") or existing_alert["updated_at"]
                elapsed = self._calculate_elapsed_seconds(last_trig, now_iso)

                trigger_count = int(existing_alert.get("trigger_count", 1)) + 1
                updated_alert = Alert(
                    alert_id=existing_alert["alert_id"],
                    alert_type=AlertType.MODEL_DEGRADATION.value,
                    severity=severity,
                    status=AlertStatus.ACTIVE.value,
                    model_version=m_version,
                    message=msg,
                    metrics=metrics_payload,
                    source_monitoring_result=perf_result,
                    created_at=existing_alert["created_at"],
                    updated_at=now_iso,
                    resolved_at=None,
                    trigger_count=trigger_count,
                    last_triggered_at=now_iso,
                )
                save_alert(updated_alert.to_dict(), db_path=self.db_path)

                if elapsed < cooldown:
                    notif = NotificationMessage(
                        title=f"Model Performance Degraded: {m_version} (Suppressed)",
                        severity=severity,
                        event_type="suppressed",
                        timestamp=now_iso,
                        body=f"{msg} (Occurrences: {trigger_count}, within cooldown of {cooldown}s).",
                        metrics_summary=metrics_summary,
                        recommended_action="Suppressed notification. Monitor for automated retraining trigger.",
                        alert_id=updated_alert.alert_id,
                        model_version=m_version,
                    )
                    return updated_alert, notif
                else:
                    notif = NotificationMessage(
                        title=f"Model Performance Degraded: {m_version} (Recurring)",
                        severity=severity,
                        event_type="triggered",
                        timestamp=now_iso,
                        body=f"{msg} (Occurrences: {trigger_count}).",
                        metrics_summary=metrics_summary,
                        recommended_action="Model accuracy degraded past policy thresholds. Prioritize model re-training.",
                        alert_id=updated_alert.alert_id,
                        model_version=m_version,
                    )
                    return updated_alert, notif
            else:
                # Create brand new degradation alert
                new_alert_id = f"alert_perf_{uuid.uuid4().hex[:10]}"
                new_alert = Alert(
                    alert_id=new_alert_id,
                    alert_type=AlertType.MODEL_DEGRADATION.value,
                    severity=severity,
                    status=AlertStatus.ACTIVE.value,
                    model_version=m_version,
                    message=msg,
                    metrics=metrics_payload,
                    source_monitoring_result=perf_result,
                    created_at=now_iso,
                    updated_at=now_iso,
                    resolved_at=None,
                    trigger_count=1,
                    last_triggered_at=now_iso,
                )
                save_alert(new_alert.to_dict(), db_path=self.db_path)

                notif = NotificationMessage(
                    title=f"Model Performance Degraded: {m_version}",
                    severity=severity,
                    event_type="triggered",
                    timestamp=now_iso,
                    body=msg,
                    metrics_summary=metrics_summary,
                    recommended_action="Trigger retraining pipeline or evaluate feature drift correlation.",
                    alert_id=new_alert_id,
                    model_version=m_version,
                )
                return new_alert, notif

        else:
            # Model performance is healthy: resolve any existing active alert for this model version
            if status == "healthy" and existing_alert:
                db_resolve_alert(existing_alert["alert_id"], resolved_at=now_iso, db_path=self.db_path)
                resolved_alert = Alert(
                    alert_id=existing_alert["alert_id"],
                    alert_type=AlertType.MODEL_DEGRADATION.value,
                    severity=AlertSeverity.INFO.value,
                    status=AlertStatus.RESOLVED.value,
                    model_version=m_version,
                    message=f"Model performance for '{m_version}' recovered to healthy baseline bounds.",
                    metrics=perf_result.get("metrics") or {},
                    source_monitoring_result=perf_result,
                    created_at=existing_alert["created_at"],
                    updated_at=now_iso,
                    resolved_at=now_iso,
                    trigger_count=existing_alert.get("trigger_count", 1),
                    last_triggered_at=existing_alert.get("last_triggered_at", existing_alert["updated_at"]),
                )
                notif = NotificationMessage(
                    title=f"Model Performance Alert Resolved: {m_version}",
                    severity=AlertSeverity.INFO.value,
                    event_type="resolved",
                    timestamp=now_iso,
                    body=f"Metrics for model '{m_version}' have returned to healthy baseline parameters.",
                    metrics_summary=f"MAE={perf_result.get('metrics', {}).get('mae')}, RMSE={perf_result.get('metrics', {}).get('rmse')}",
                    recommended_action="Normal performance verified.",
                    alert_id=existing_alert["alert_id"],
                    model_version=m_version,
                )
                return resolved_alert, notif

            return None, None

    def record_execution_failure(
        self,
        operation_name: str,
        error: Union[Exception, str],
        model_version: Optional[str] = None,
        cooldown_seconds: Optional[int] = None,
    ) -> Tuple[Alert, NotificationMessage]:
        """Record an alert for an unhandled failure during monitoring operations.

        Supports deduplication and cooldown for repeated failures of the same operation.

        Args:
            operation_name: Name of the monitoring component that failed.
            error: The Exception instance or error message string.
            model_version: Optional model version associated with the run.
            cooldown_seconds: Optional cooldown period override.

        Returns:
            Tuple of (Alert, NotificationMessage).
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        err_msg = str(error)
        err_type = type(error).__name__ if isinstance(error, Exception) else "ExecutionError"
        cooldown = cooldown_seconds if cooldown_seconds is not None else self.default_cooldown_seconds

        msg = f"Monitoring execution failed in '{operation_name}': [{err_type}] {err_msg}"
        metrics = {
            "operation": operation_name,
            "error_type": err_type,
            "error_message": err_msg,
        }

        # Check existing active failure alert for this specific operation
        active_failures = get_active_alerts(
            db_path=self.db_path,
            alert_type=AlertType.MONITORING_EXECUTION_FAILURE.value,
        )
        existing = next(
            (
                a for a in active_failures
                if isinstance(a.get("metrics"), dict) and a["metrics"].get("operation") == operation_name
            ),
            None,
        )

        if existing:
            last_trig = existing.get("last_triggered_at") or existing["updated_at"]
            elapsed = self._calculate_elapsed_seconds(last_trig, now_iso)
            trigger_count = int(existing.get("trigger_count", 1)) + 1

            updated_alert = Alert(
                alert_id=existing["alert_id"],
                alert_type=AlertType.MONITORING_EXECUTION_FAILURE.value,
                severity=AlertSeverity.ERROR.value,
                status=AlertStatus.ACTIVE.value,
                model_version=model_version or existing.get("model_version"),
                message=msg,
                metrics=metrics,
                source_monitoring_result={"error": err_msg, "error_type": err_type, "operation": operation_name},
                created_at=existing["created_at"],
                updated_at=now_iso,
                resolved_at=None,
                trigger_count=trigger_count,
                last_triggered_at=now_iso,
            )
            save_alert(updated_alert.to_dict(), db_path=self.db_path)

            if elapsed < cooldown:
                notif = NotificationMessage(
                    title=f"Monitoring Failure: {operation_name} (Suppressed)",
                    severity=AlertSeverity.ERROR.value,
                    event_type="suppressed",
                    timestamp=now_iso,
                    body=f"{msg} (Occurrences: {trigger_count}, within cooldown of {cooldown}s).",
                    metrics_summary=f"operation={operation_name}, error={err_type}",
                    recommended_action="Check database integrity, data file paths, and environment dependencies.",
                    alert_id=updated_alert.alert_id,
                    model_version=model_version,
                )
                return updated_alert, notif
            else:
                notif = NotificationMessage(
                    title=f"Monitoring Failure: {operation_name} (Recurring)",
                    severity=AlertSeverity.ERROR.value,
                    event_type="failure",
                    timestamp=now_iso,
                    body=f"{msg} (Occurrences: {trigger_count}).",
                    metrics_summary=f"operation={operation_name}, error={err_type}",
                    recommended_action="Check database integrity, data file paths, and environment dependencies.",
                    alert_id=updated_alert.alert_id,
                    model_version=model_version,
                )
                return updated_alert, notif
        else:
            alert_id = f"alert_fail_{uuid.uuid4().hex[:10]}"
            new_alert = Alert(
                alert_id=alert_id,
                alert_type=AlertType.MONITORING_EXECUTION_FAILURE.value,
                severity=AlertSeverity.ERROR.value,
                status=AlertStatus.ACTIVE.value,
                model_version=model_version,
                message=msg,
                metrics=metrics,
                source_monitoring_result={"error": err_msg, "error_type": err_type, "operation": operation_name},
                created_at=now_iso,
                updated_at=now_iso,
                resolved_at=None,
                trigger_count=1,
                last_triggered_at=now_iso,
            )
            save_alert(new_alert.to_dict(), db_path=self.db_path)

            notif = NotificationMessage(
                title=f"Monitoring Failure: {operation_name}",
                severity=AlertSeverity.ERROR.value,
                event_type="failure",
                timestamp=now_iso,
                body=msg,
                metrics_summary=f"operation={operation_name}, error={err_type}",
                recommended_action="Check database integrity, data file paths, and environment dependencies.",
                alert_id=alert_id,
                model_version=model_version,
            )
            return new_alert, notif

    def resolve_execution_failure(
        self,
        operation_name: str,
    ) -> Optional[Tuple[Alert, NotificationMessage]]:
        """Resolve any active failure alert for the specified operation.

        Args:
            operation_name: Component operation that succeeded (e.g. 'drift_detection').

        Returns:
            Tuple of (resolved Alert, NotificationMessage) if an active alert was resolved, else None.
        """
        active_failures = get_active_alerts(
            db_path=self.db_path,
            alert_type=AlertType.MONITORING_EXECUTION_FAILURE.value,
        )
        matching = [
            a for a in active_failures
            if isinstance(a.get("metrics"), dict) and a["metrics"].get("operation") == operation_name
        ]
        if not matching:
            return None

        now_iso = datetime.now(timezone.utc).isoformat()
        target = matching[0]
        db_resolve_alert(target["alert_id"], resolved_at=now_iso, db_path=self.db_path)

        resolved_alert = Alert(
            alert_id=target["alert_id"],
            alert_type=AlertType.MONITORING_EXECUTION_FAILURE.value,
            severity=AlertSeverity.INFO.value,
            status=AlertStatus.RESOLVED.value,
            model_version=target.get("model_version"),
            message=f"Monitoring execution failure in '{operation_name}' has recovered.",
            metrics=target.get("metrics") or {},
            source_monitoring_result=target.get("source_monitoring_result") or {},
            created_at=target["created_at"],
            updated_at=now_iso,
            resolved_at=now_iso,
            trigger_count=int(target.get("trigger_count", 1)),
            last_triggered_at=target.get("last_triggered_at", target["updated_at"]),
        )

        notif = NotificationMessage(
            title=f"Monitoring Recovery: {operation_name}",
            severity=AlertSeverity.INFO.value,
            event_type="resolved",
            timestamp=now_iso,
            body=f"Execution for component '{operation_name}' succeeded. Active failure alert resolved.",
            metrics_summary=f"operation={operation_name}, status=recovered",
            recommended_action="Normal monitoring operations restored for this component.",
            alert_id=target["alert_id"],
            model_version=target.get("model_version"),
        )
        return resolved_alert, notif

    # ─── Query & Lifecycle Helpers ────────────────────────────────────

    def get_active_alerts(
        self,
        alert_type: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve currently active alerts."""
        return get_active_alerts(db_path=self.db_path, alert_type=alert_type, model_version=model_version)

    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve full chronological alert history."""
        return get_alert_history(limit=limit, db_path=self.db_path)

    def get_alert_by_id(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an alert record by ID."""
        return get_alert(alert_id, db_path=self.db_path)

    def resolve_alert_by_id(self, alert_id: str) -> bool:
        """Manually mark an active alert as resolved."""
        return db_resolve_alert(alert_id, db_path=self.db_path)

    # ─── Internal Utilities ───────────────────────────────────────────

    @staticmethod
    def _calculate_elapsed_seconds(start_iso: str, end_iso: str) -> float:
        """Calculate elapsed seconds between two ISO-8601 timestamps."""
        try:
            t0 = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
            return max(0.0, (t1 - t0).total_seconds())
        except Exception:
            return float("inf")
