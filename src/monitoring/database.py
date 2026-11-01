"""
SQLite persistence layer for inference prediction logging and ground-truth feedback.

Phase 3.1 deliverable: Implements decoupled, non-blocking storage for
operational model predictions and delayed actual meter feedback.
"""

import json
import math
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid


class PredictionNotFoundError(Exception):
    """Raised when feedback is submitted for a non-existent prediction ID."""
    pass


class DuplicateFeedbackError(Exception):
    """Raised when feedback has already been registered for a prediction ID."""
    pass


class DuplicatePredictionError(Exception):
    """Raised when attempting to insert a duplicate prediction ID."""
    pass


class InvalidFeedbackValueError(Exception):
    """Raised when feedback contains an invalid actual energy value."""
    pass


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "monitoring", "predictions.db")


def get_db_path() -> str:
    """Return the active monitoring database path from environment or default."""
    return os.getenv("MONITORING_DB_PATH", DEFAULT_DB_PATH)


@contextmanager
def get_connection(db_path: Optional[str] = None):
    """Create, configure, and safely close a SQLite database connection."""
    path = db_path or get_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize database tables, views, and indexes."""
    path = db_path or get_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with get_connection(path) as conn:
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
        except sqlite3.OperationalError:
            pass  # In-memory or restricted environments may skip WAL

        conn.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            energy_current_hour REAL NOT NULL,
            energy_previous_hour REAL NOT NULL,
            energy_two_hours_ago REAL NOT NULL,
            target_hour INTEGER NOT NULL,
            target_day_of_week INTEGER NOT NULL,
            target_is_weekend INTEGER NOT NULL,
            predicted_energy_kwh REAL NOT NULL,
            model_version TEXT NOT NULL,
            latency_ms REAL
        );

        CREATE TABLE IF NOT EXISTS feedback (
            prediction_id TEXT PRIMARY KEY,
            actual_energy_kwh REAL NOT NULL,
            feedback_timestamp TEXT NOT NULL,
            FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
        CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(feedback_timestamp);

        CREATE VIEW IF NOT EXISTS v_monitoring_records AS
        SELECT 
            p.prediction_id,
            p.timestamp AS prediction_timestamp,
            p.energy_current_hour,
            p.energy_previous_hour,
            p.energy_two_hours_ago,
            p.target_hour,
            p.target_day_of_week,
            p.target_is_weekend,
            p.predicted_energy_kwh,
            p.model_version,
            p.latency_ms,
            f.actual_energy_kwh,
            f.feedback_timestamp,
            CASE 
                WHEN f.actual_energy_kwh IS NOT NULL 
                THEN (f.actual_energy_kwh - p.predicted_energy_kwh) 
                ELSE NULL 
            END AS error,
            CASE 
                WHEN f.actual_energy_kwh IS NOT NULL 
                THEN ABS(f.actual_energy_kwh - p.predicted_energy_kwh) 
                ELSE NULL 
            END AS absolute_error
        FROM predictions p
        LEFT JOIN feedback f ON p.prediction_id = f.prediction_id;

        CREATE TABLE IF NOT EXISTS alerts (
            alert_id TEXT PRIMARY KEY,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            status TEXT NOT NULL,
            model_version TEXT,
            message TEXT NOT NULL,
            metrics_json TEXT,
            source_result_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            resolved_at TEXT,
            trigger_count INTEGER NOT NULL DEFAULT 1,
            last_triggered_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
        CREATE INDEX IF NOT EXISTS idx_alerts_type_version ON alerts(alert_type, model_version, status);
        CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);

        CREATE TABLE IF NOT EXISTS monitoring_runs (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            model_version TEXT,
            drift_status TEXT NOT NULL,
            drift_share REAL,
            performance_status TEXT NOT NULL,
            mae REAL,
            rmse REAL,
            r2 REAL,
            bias REAL,
            sample_count INTEGER,
            alerts_triggered INTEGER NOT NULL,
            alerts_resolved INTEGER NOT NULL,
            active_alerts_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS monitoring_execution_failures (
            failure_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            operation TEXT NOT NULL,
            error_message TEXT,
            model_version TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_monitoring_runs_timestamp ON monitoring_runs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_monitoring_failures_op ON monitoring_execution_failures(operation);
        """)
        conn.commit()


def log_prediction(
    prediction_id: str,
    features: Dict[str, Any],
    predicted_energy_kwh: float,
    model_version: str = "GradientBoostingRegressor",
    latency_ms: Optional[float] = None,
    timestamp: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    """Record an inference prediction and its feature vector."""
    req_time = timestamp or datetime.now(timezone.utc).isoformat()
    path = db_path or get_db_path()

    with get_connection(path) as conn:
        try:
            conn.execute(
                """
                INSERT INTO predictions (
                    prediction_id,
                    timestamp,
                    energy_current_hour,
                    energy_previous_hour,
                    energy_two_hours_ago,
                    target_hour,
                    target_day_of_week,
                    target_is_weekend,
                    predicted_energy_kwh,
                    model_version,
                    latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    prediction_id,
                    req_time,
                    float(features["energy_current_hour"]),
                    float(features["energy_previous_hour"]),
                    float(features["energy_two_hours_ago"]),
                    int(features["target_hour"]),
                    int(features["target_day_of_week"]),
                    int(features["target_is_weekend"]),
                    float(predicted_energy_kwh),
                    str(model_version),
                    float(latency_ms) if latency_ms is not None else None,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            err_msg = str(e).lower()
            if "unique" in err_msg or "primary" in err_msg:
                raise DuplicatePredictionError(
                    f"Prediction ID '{prediction_id}' already exists in database."
                ) from e
            raise


def get_prediction(prediction_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a prediction record by its unique ID."""
    path = db_path or get_db_path()
    with get_connection(path) as conn:
        cursor = conn.execute(
            "SELECT * FROM predictions WHERE prediction_id = ?;",
            (prediction_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def log_feedback(
    prediction_id: str,
    actual_energy_kwh: float,
    feedback_timestamp: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach ground-truth smart-meter reading to an existing prediction.

    Raises:
        InvalidFeedbackValueError: If actual_energy_kwh is non-numeric, NaN, infinite, or negative.
        PredictionNotFoundError: If prediction_id does not exist.
        DuplicateFeedbackError: If feedback was already recorded for this prediction.
    """
    if not isinstance(actual_energy_kwh, (int, float)) or isinstance(actual_energy_kwh, bool):
        raise InvalidFeedbackValueError(
            f"Actual energy consumption must be a numeric value, got: {type(actual_energy_kwh).__name__}"
        )

    if math.isnan(actual_energy_kwh) or math.isinf(actual_energy_kwh):
        raise InvalidFeedbackValueError(
            f"Actual energy consumption must be a finite number, got: {actual_energy_kwh}"
        )

    if actual_energy_kwh < 0:
        raise InvalidFeedbackValueError(
            f"Actual energy consumption cannot be negative: {actual_energy_kwh}"
        )

    fb_time = feedback_timestamp or datetime.now(timezone.utc).isoformat()
    path = db_path or get_db_path()

    with get_connection(path) as conn:
        # Check prediction existence
        pred_cursor = conn.execute(
            "SELECT prediction_id FROM predictions WHERE prediction_id = ?;",
            (prediction_id,),
        )
        if not pred_cursor.fetchone():
            raise PredictionNotFoundError(
                f"Prediction ID '{prediction_id}' not found in monitoring store."
            )

        # Check existing feedback
        fb_cursor = conn.execute(
            "SELECT prediction_id FROM feedback WHERE prediction_id = ?;",
            (prediction_id,),
        )
        if fb_cursor.fetchone():
            raise DuplicateFeedbackError(
                f"Feedback has already been registered for prediction ID '{prediction_id}'."
            )

        try:
            conn.execute(
                """
                INSERT INTO feedback (prediction_id, actual_energy_kwh, feedback_timestamp)
                VALUES (?, ?, ?);
                """,
                (prediction_id, float(actual_energy_kwh), fb_time),
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            err_msg = str(e).lower()
            if "unique" in err_msg or "primary" in err_msg:
                raise DuplicateFeedbackError(
                    f"Feedback has already been registered for prediction ID '{prediction_id}'."
                ) from e
            if "foreign" in err_msg:
                raise PredictionNotFoundError(
                    f"Prediction ID '{prediction_id}' not found in monitoring store."
                ) from e
            raise

    return {
        "prediction_id": prediction_id,
        "actual_energy_kwh": actual_energy_kwh,
        "feedback_timestamp": fb_time,
    }


def get_feedback(prediction_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve a feedback record by prediction ID."""
    path = db_path or get_db_path()
    with get_connection(path) as conn:
        cursor = conn.execute(
            "SELECT * FROM feedback WHERE prediction_id = ?;",
            (prediction_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_recent_monitoring_records(limit: int = 100, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve joined predictions and feedback records."""
    path = db_path or get_db_path()
    with get_connection(path) as conn:
        cursor = conn.execute(
            """
            SELECT * FROM v_monitoring_records
            ORDER BY prediction_timestamp DESC
            LIMIT ?;
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_monitoring_counts(db_path: Optional[str] = None) -> Dict[str, int]:
    """Return counts of total predictions and paired feedback."""
    path = db_path or get_db_path()
    with get_connection(path) as conn:
        pred_count = conn.execute("SELECT COUNT(*) FROM predictions;").fetchone()[0]
        fb_count = conn.execute("SELECT COUNT(*) FROM feedback;").fetchone()[0]
        return {
            "total_predictions": pred_count,
            "total_feedback": fb_count,
        }


# ─── Alert Persistence and Lifecycle ──────────────────────────────────

def save_alert(alert_data: Dict[str, Any], db_path: Optional[str] = None) -> None:
    """Insert or update an alert record in the monitoring database."""
    path = db_path or get_db_path()
    metrics_str = (
        json.dumps(alert_data["metrics"])
        if isinstance(alert_data.get("metrics"), (dict, list))
        else alert_data.get("metrics_json")
    )
    source_str = (
        json.dumps(alert_data["source_monitoring_result"])
        if isinstance(alert_data.get("source_monitoring_result"), (dict, list))
        else alert_data.get("source_result_json")
    )

    with get_connection(path) as conn:
        conn.execute(
            """
            INSERT INTO alerts (
                alert_id,
                alert_type,
                severity,
                status,
                model_version,
                message,
                metrics_json,
                source_result_json,
                created_at,
                updated_at,
                resolved_at,
                trigger_count,
                last_triggered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(alert_id) DO UPDATE SET
                status = excluded.status,
                severity = excluded.severity,
                message = excluded.message,
                metrics_json = excluded.metrics_json,
                source_result_json = excluded.source_result_json,
                updated_at = excluded.updated_at,
                resolved_at = excluded.resolved_at,
                trigger_count = excluded.trigger_count,
                last_triggered_at = excluded.last_triggered_at;
            """,
            (
                alert_data["alert_id"],
                alert_data["alert_type"],
                alert_data["severity"],
                alert_data["status"],
                alert_data.get("model_version"),
                alert_data["message"],
                metrics_str,
                source_str,
                alert_data["created_at"],
                alert_data["updated_at"],
                alert_data.get("resolved_at"),
                int(alert_data.get("trigger_count", 1)),
                alert_data.get("last_triggered_at", alert_data["updated_at"]),
            ),
        )
        conn.commit()


def get_alert(alert_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieve an alert record by its unique ID."""
    path = db_path or get_db_path()
    with get_connection(path) as conn:
        cursor = conn.execute(
            "SELECT * FROM alerts WHERE alert_id = ?;",
            (alert_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        if res.get("metrics_json"):
            try:
                res["metrics"] = json.loads(res["metrics_json"])
            except Exception:
                res["metrics"] = res["metrics_json"]
        if res.get("source_result_json"):
            try:
                res["source_monitoring_result"] = json.loads(res["source_result_json"])
            except Exception:
                res["source_monitoring_result"] = res["source_result_json"]
        return res


def get_active_alerts(
    db_path: Optional[str] = None,
    alert_type: Optional[str] = None,
    model_version: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve currently active alerts, optionally filtered by type or model version."""
    path = db_path or get_db_path()
    query = "SELECT * FROM alerts WHERE status = 'active'"
    params: List[Any] = []

    if alert_type:
        query += " AND alert_type = ?"
        params.append(alert_type)

    if model_version:
        query += " AND (model_version = ? OR model_version IS NULL)"
        params.append(model_version)

    query += " ORDER BY created_at DESC;"

    with get_connection(path) as conn:
        cursor = conn.execute(query, params)
        records = []
        for row in cursor.fetchall():
            res = dict(row)
            if res.get("metrics_json"):
                try:
                    res["metrics"] = json.loads(res["metrics_json"])
                except Exception:
                    res["metrics"] = res["metrics_json"]
            if res.get("source_result_json"):
                try:
                    res["source_monitoring_result"] = json.loads(res["source_result_json"])
                except Exception:
                    res["source_monitoring_result"] = res["source_result_json"]
            records.append(res)
        return records


def resolve_alert(
    alert_id: str,
    resolved_at: Optional[str] = None,
    db_path: Optional[str] = None,
) -> bool:
    """Transition an active alert to resolved status."""
    path = db_path or get_db_path()
    res_time = resolved_at or datetime.now(timezone.utc).isoformat()
    with get_connection(path) as conn:
        cursor = conn.execute(
            """
            UPDATE alerts
            SET status = 'resolved',
                resolved_at = ?,
                updated_at = ?
            WHERE alert_id = ? AND status = 'active';
            """,
            (res_time, res_time, alert_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def get_alert_history(
    limit: int = 100,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve full chronological alert history (both active and resolved)."""
    path = db_path or get_db_path()
    with get_connection(path) as conn:
        cursor = conn.execute(
            """
            SELECT * FROM alerts
            ORDER BY created_at DESC
            LIMIT ?;
            """,
            (limit,),
        )
        records = []
        for row in cursor.fetchall():
            res = dict(row)
            if res.get("metrics_json"):
                try:
                    res["metrics"] = json.loads(res["metrics_json"])
                except Exception:
                    res["metrics"] = res["metrics_json"]
            if res.get("source_result_json"):
                try:
                    res["source_monitoring_result"] = json.loads(res["source_result_json"])
                except Exception:
                    res["source_monitoring_result"] = res["source_result_json"]
            records.append(res)
        return records


# ─── Monitoring Runs and Observability Persistence ────────────────────

def log_monitoring_run(
    run_id: str,
    timestamp: str,
    drift_status: str,
    performance_status: str,
    alerts_triggered: int,
    alerts_resolved: int,
    active_alerts_count: int,
    model_version: Optional[str] = None,
    drift_share: Optional[float] = None,
    mae: Optional[float] = None,
    rmse: Optional[float] = None,
    r2: Optional[float] = None,
    bias: Optional[float] = None,
    sample_count: Optional[int] = None,
    db_path: Optional[str] = None,
) -> None:
    """Record a completed monitoring cycle and its top-level evaluation metrics."""
    path = db_path or get_db_path()
    with get_connection(path) as conn:
        conn.execute(
            """
            INSERT INTO monitoring_runs (
                run_id, timestamp, model_version, drift_status, drift_share,
                performance_status, mae, rmse, r2, bias, sample_count,
                alerts_triggered, alerts_resolved, active_alerts_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                run_id,
                timestamp,
                model_version,
                drift_status,
                float(drift_share) if drift_share is not None else None,
                performance_status,
                float(mae) if mae is not None else None,
                float(rmse) if rmse is not None else None,
                float(r2) if r2 is not None else None,
                float(bias) if bias is not None else None,
                int(sample_count) if sample_count is not None else None,
                int(alerts_triggered),
                int(alerts_resolved),
                int(active_alerts_count),
            ),
        )
        conn.commit()


def log_monitoring_execution_failure(
    operation: str,
    error_message: str,
    model_version: Optional[str] = None,
    timestamp: Optional[str] = None,
    db_path: Optional[str] = None,
) -> str:
    """Record an operational execution failure during monitoring orchestration."""
    path = db_path or get_db_path()
    failure_id = f"fail_{uuid.uuid4().hex[:8]}"
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    with get_connection(path) as conn:
        conn.execute(
            """
            INSERT INTO monitoring_execution_failures (
                failure_id, timestamp, operation, error_message, model_version
            ) VALUES (?, ?, ?, ?, ?);
            """,
            (failure_id, ts, operation, str(error_message), model_version),
        )
        conn.commit()
    return failure_id


def get_monitoring_runs(
    limit: int = 100,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve historical monitoring run records."""
    path = db_path or get_db_path()
    with get_connection(path) as conn:
        cursor = conn.execute(
            "SELECT * FROM monitoring_runs ORDER BY timestamp DESC LIMIT ?;",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_monitoring_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve aggregate observability metrics for Prometheus scraping."""
    path = db_path or get_db_path()
    stats: Dict[str, Any] = {
        "drift_runs_total": 0,
        "drift_failures_total": 0,
        "latest_drift_share": None,
        "performance_runs_total": 0,
        "performance_failures_total": 0,
        "latest_mae": {},
        "latest_rmse": {},
        "active_alerts": {},
        "execution_failures": {},
    }

    with get_connection(path) as conn:
        table_check = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('monitoring_runs', 'alerts', 'monitoring_execution_failures');"
        ).fetchall()
        tables = {row[0] for row in table_check}

        if "monitoring_runs" in tables:
            row = conn.execute(
                "SELECT COUNT(*) FROM monitoring_runs WHERE drift_status IN ('evaluated', 'failed');"
            ).fetchone()
            stats["drift_runs_total"] = int(row[0]) if row else 0

            row = conn.execute(
                "SELECT COUNT(*) FROM monitoring_runs WHERE drift_status = 'failed';"
            ).fetchone()
            stats["drift_failures_total"] = int(row[0]) if row else 0

            row = conn.execute(
                "SELECT drift_share FROM monitoring_runs WHERE drift_share IS NOT NULL ORDER BY timestamp DESC LIMIT 1;"
            ).fetchone()
            if row and row[0] is not None:
                stats["latest_drift_share"] = float(row[0])

            row = conn.execute(
                "SELECT COUNT(*) FROM monitoring_runs WHERE performance_status IN ('evaluated', 'failed');"
            ).fetchone()
            stats["performance_runs_total"] = int(row[0]) if row else 0

            row = conn.execute(
                "SELECT COUNT(*) FROM monitoring_runs WHERE performance_status = 'failed';"
            ).fetchone()
            stats["performance_failures_total"] = int(row[0]) if row else 0

            rows = conn.execute(
                """
                SELECT model_version, mae, rmse, MAX(timestamp)
                FROM monitoring_runs
                WHERE mae IS NOT NULL AND rmse IS NOT NULL
                GROUP BY model_version;
                """
            ).fetchall()
            for r in rows:
                mv = r[0] or "GradientBoostingRegressor"
                if r[1] is not None and r[2] is not None:
                    stats["latest_mae"][mv] = float(r[1])
                    stats["latest_rmse"][mv] = float(r[2])

        if "alerts" in tables:
            rows = conn.execute(
                """
                SELECT alert_type, severity, COUNT(*)
                FROM alerts
                WHERE status = 'active'
                GROUP BY alert_type, severity;
                """
            ).fetchall()
            for r in rows:
                stats["active_alerts"][(r[0], r[1])] = int(r[2])

        if "monitoring_execution_failures" in tables:
            rows = conn.execute(
                """
                SELECT operation, COUNT(*)
                FROM monitoring_execution_failures
                GROUP BY operation;
                """
            ).fetchall()
            for r in rows:
                stats["execution_failures"][r[0]] = int(r[1])

    return stats
