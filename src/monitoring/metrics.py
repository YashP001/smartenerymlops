"""
Prometheus Observability and Metrics Instrumentation (Phase 3.6).

Exposes metrics for:
1. API performance and reliability (request counts, status codes, latency histograms).
2. Machine Learning inference activity (predictions count, latency, output distribution).
3. Out-of-band monitoring and alert orchestration (drift share, MAE/RMSE, active alerts,
   execution failures) queried dynamically from the SQLite persistence store.
"""

import os
import re
import sys
from typing import Any, Dict, Optional, Tuple, Union

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

# Add project root for imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.monitoring.database import get_db_path, get_monitoring_stats

# Canonical latency buckets in seconds for sub-second inference calls
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5)

# Energy consumption distribution buckets in kWh
ENERGY_KWH_BUCKETS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0)

# UUID regex pattern for low-cardinality endpoint sanitization
_UUID_REGEX = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def normalize_endpoint(path: str) -> str:
    """Normalize raw request paths to low-cardinality route templates.

    Replaces UUIDs and dynamic path parameters with placeholder tokens
    to prevent high-cardinality label explosion.
    """
    if not path or path == "unmatched":
        return "unmatched"
    # Replace UUIDs
    clean = _UUID_REGEX.sub("{prediction_id}", path)
    return clean


class MonitoringMetricsCollector:
    """Custom Prometheus collector querying out-of-band monitoring stats from SQLite.

    Ensures the monitoring orchestrator's state (drift, degradation, alerts, failures)
    is visible through the API metrics exposition without sharing Python in-memory state.
    Emits no data when values are not available, preventing fabricated zero readings.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def collect(self):
        try:
            stats = get_monitoring_stats(self.db_path)
        except Exception:
            return

        # 1. Total drift detection runs
        drift_runs = CounterMetricFamily(
            "energy_advisor_drift_detection_runs_total",
            "Total number of completed or failed data drift detection runs.",
        )
        drift_runs.add_metric([], stats.get("drift_runs_total", 0))
        yield drift_runs

        # 2. Drift detection failures
        drift_failures = CounterMetricFamily(
            "energy_advisor_drift_detection_failures_total",
            "Total number of failed data drift detection runs.",
        )
        drift_failures.add_metric([], stats.get("drift_failures_total", 0))
        yield drift_failures

        # 3. Current drift share (only emitted if drift analysis has occurred)
        drift_share = stats.get("latest_drift_share")
        if drift_share is not None:
            drift_gauge = GaugeMetricFamily(
                "energy_advisor_dataset_drift_share",
                "Current share of drifted input features (fraction between 0.0 and 1.0).",
            )
            drift_gauge.add_metric([], float(drift_share))
            yield drift_gauge

        # 4. Total performance evaluation runs
        perf_runs = CounterMetricFamily(
            "energy_advisor_performance_evaluation_runs_total",
            "Total number of completed or failed model performance evaluation runs.",
        )
        perf_runs.add_metric([], stats.get("performance_runs_total", 0))
        yield perf_runs

        # 5. Performance evaluation failures
        perf_failures = CounterMetricFamily(
            "energy_advisor_performance_evaluation_failures_total",
            "Total number of failed model performance evaluation runs.",
        )
        perf_failures.add_metric([], stats.get("performance_failures_total", 0))
        yield perf_failures

        # 6. Current MAE and RMSE (only emitted when sufficient feedback exists)
        latest_mae = stats.get("latest_mae", {})
        if latest_mae:
            mae_gauge = GaugeMetricFamily(
                "energy_advisor_model_mae",
                "Current Mean Absolute Error (MAE) in kWh evaluated on recent feedback.",
                labels=["model_version"],
            )
            for mv, val in latest_mae.items():
                mae_gauge.add_metric([mv], float(val))
            yield mae_gauge

        latest_rmse = stats.get("latest_rmse", {})
        if latest_rmse:
            rmse_gauge = GaugeMetricFamily(
                "energy_advisor_model_rmse",
                "Current Root Mean Squared Error (RMSE) in kWh evaluated on recent feedback.",
                labels=["model_version"],
            )
            for mv, val in latest_rmse.items():
                rmse_gauge.add_metric([mv], float(val))
            yield rmse_gauge

        # 7. Active alerts grouped by alert_type and severity
        active_alerts = stats.get("active_alerts", {})
        alert_gauge = GaugeMetricFamily(
            "energy_advisor_active_alerts",
            "Current count of active monitoring alerts grouped by alert type and severity.",
            labels=["alert_type", "severity"],
        )
        if active_alerts:
            for (alert_type, severity), count in active_alerts.items():
                alert_gauge.add_metric([alert_type, severity], float(count))
        else:
            # When no alerts exist, report 0 for standard active categories
            pass
        yield alert_gauge

        # 8. Monitoring execution failures grouped by operation
        exec_failures = stats.get("execution_failures", {})
        fail_counter = CounterMetricFamily(
            "energy_advisor_monitoring_execution_failures_total",
            "Total count of monitoring orchestration execution failures grouped by operation.",
            labels=["operation"],
        )
        if exec_failures:
            for op, count in exec_failures.items():
                fail_counter.add_metric([op], float(count))
        else:
            # Emit zero for default operations so series are discoverable
            fail_counter.add_metric(["drift_detection"], 0.0)
            fail_counter.add_metric(["performance_monitoring"], 0.0)
        yield fail_counter


class MetricsContainer:
    """Encapsulates Prometheus metric definitions bound to a specific CollectorRegistry."""

    def __init__(self, registry: Optional[CollectorRegistry] = None, db_path: Optional[str] = None):
        self.registry = registry or CollectorRegistry()
        self.db_path = db_path

        # ─── API & Inference Metrics ──────────────────────────────────
        self.prediction_requests = Counter(
            "energy_advisor_prediction_requests_total",
            "Total number of prediction requests processed by the API service.",
            ["model_version", "status"],
            registry=self.registry,
        )

        self.prediction_latency = Histogram(
            "energy_advisor_prediction_latency_seconds",
            "Prediction inference latency in seconds.",
            ["model_version"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )

        self.http_requests = Counter(
            "energy_advisor_http_requests_total",
            "Total HTTP requests handled by the API service.",
            ["method", "endpoint", "status_code"],
            registry=self.registry,
        )

        self.http_request_duration = Histogram(
            "energy_advisor_http_request_duration_seconds",
            "HTTP request duration in seconds.",
            ["method", "endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )

        self.model_predictions = Counter(
            "energy_advisor_model_predictions_total",
            "Total count of successful energy consumption predictions generated.",
            ["model_version"],
            registry=self.registry,
        )

        self.predicted_energy_distribution = Histogram(
            "energy_advisor_predicted_energy_kwh",
            "Distribution of predicted hourly electricity consumption in kWh.",
            ["model_version"],
            buckets=ENERGY_KWH_BUCKETS,
            registry=self.registry,
        )

        # ─── Custom Monitoring Collector ──────────────────────────────
        self.collector = MonitoringMetricsCollector(db_path=self.db_path)
        self.registry.register(self.collector)

    def record_prediction(
        self,
        model_version: str,
        status: str,
        latency_seconds: Optional[float] = None,
        predicted_energy: Optional[float] = None,
    ) -> None:
        """Record prediction request counters, latency, and distribution."""
        self.prediction_requests.labels(model_version=model_version, status=status).inc()
        if status == "success":
            self.model_predictions.labels(model_version=model_version).inc()
            if latency_seconds is not None:
                self.prediction_latency.labels(model_version=model_version).observe(latency_seconds)
            if predicted_energy is not None:
                self.predicted_energy_distribution.labels(model_version=model_version).observe(predicted_energy)

    def record_http(
        self,
        method: str,
        endpoint: str,
        status_code: Union[int, str],
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Record HTTP request counts and durations with low-cardinality endpoint paths."""
        norm_endpoint = normalize_endpoint(endpoint)
        self.http_requests.labels(
            method=method.upper(),
            endpoint=norm_endpoint,
            status_code=str(status_code),
        ).inc()
        if duration_seconds is not None:
            self.http_request_duration.labels(
                method=method.upper(),
                endpoint=norm_endpoint,
            ).observe(duration_seconds)

    def generate_payload(self) -> bytes:
        """Export latest metrics in Prometheus exposition format."""
        return generate_latest(self.registry)


# Global default metrics container for application lifespan
default_metrics = MetricsContainer()
