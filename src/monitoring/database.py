"""
SQLite persistence layer for inference prediction logging and ground-truth feedback.

Phase 3.1 deliverable: Implements decoupled, non-blocking storage for
operational model predictions and delayed actual meter feedback.
"""

import math
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


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
