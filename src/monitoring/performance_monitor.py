"""
Model Performance Monitoring Engine for Smart Home Energy Advisor (Phase 3.4).

Evaluates operational prediction accuracy against actual smart-meter feedback
recorded in the monitoring persistence layer (v_monitoring_records view).
Calculates MAE, RMSE, R², and mean prediction error (bias).
Detects performance degradation against historical test baselines or configurable limits.
"""

import argparse
import json
import math
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Add project root for imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.monitoring.database import get_connection, get_db_path

DEFAULT_REPORT_DIR = os.path.join(_PROJECT_ROOT, "reports", "performance")

# Empirical baseline established on chronological test set (reports/phase1_report.md)
BASELINE_METRICS = {
    "MAE": 0.0739,
    "RMSE": 0.1512,
    "R2": 0.3723,
    "BIAS": 0.0,
}

# Configurable degradation threshold defaults
DEFAULT_MAX_MAE_INCREASE_PCT = 0.25   # Alert if MAE exceeds baseline by > 25%
DEFAULT_MAX_RMSE_INCREASE_PCT = 0.25  # Alert if RMSE exceeds baseline by > 25%
DEFAULT_MIN_R2_THRESHOLD = 0.10       # Alert if R2 falls below 0.10
DEFAULT_MAX_ABS_BIAS = 0.05           # Alert if absolute mean bias exceeds 0.05 kWh
DEFAULT_MIN_SAMPLES = 30              # Minimum feedback records required before evaluating degradation


class PerformanceMonitoringError(Exception):
    """Base exception for performance monitoring operations."""
    pass


class InvalidWindowError(PerformanceMonitoringError):
    """Raised when evaluation window parameters are invalid or start > end."""
    pass


class DatabaseQueryError(PerformanceMonitoringError):
    """Raised when querying monitoring records encounters database errors."""
    pass


def compute_metrics(
    y_true: Union[List[float], np.ndarray, pd.Series],
    y_pred: Union[List[float], np.ndarray, pd.Series],
) -> Dict[str, Optional[float]]:
    """Compute MAE, RMSE, R², and mean prediction bias.

    Handles edge cases such as constant targets, zero actual values, and small sample sizes.

    Args:
        y_true: Array of actual ground-truth energy values.
        y_pred: Array of model predicted energy values.

    Returns:
        Dictionary containing MAE, RMSE, R², bias, and sample count.

    Raises:
        ValueError: If inputs are mismatched in length, empty, or contain non-finite numbers.
    """
    arr_true = np.asarray(y_true, dtype=np.float64)
    arr_pred = np.asarray(y_pred, dtype=np.float64)

    if len(arr_true) == 0 or len(arr_pred) == 0:
        raise ValueError("Cannot compute performance metrics on empty arrays.")

    if len(arr_true) != len(arr_pred):
        raise ValueError(
            f"Mismatched array lengths: y_true has {len(arr_true)} items, y_pred has {len(arr_pred)} items."
        )

    if not np.all(np.isfinite(arr_true)) or not np.all(np.isfinite(arr_pred)):
        raise ValueError("Inputs contain NaN or infinite values.")

    n = len(arr_true)
    errors = arr_pred - arr_true  # predicted minus actual

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    # Safe R² calculation
    if n < 2:
        r2 = None
    else:
        target_is_constant = np.all(np.isclose(arr_true, arr_true[0]))
        ss_tot = float(np.sum((arr_true - np.mean(arr_true)) ** 2))
        ss_res = float(np.sum(errors ** 2))

        if target_is_constant or np.isclose(ss_tot, 0.0):
            # Constant target: if predictions match the constant target, R²=1.0, else 0.0
            r2 = 1.0 if np.isclose(ss_res, 0.0) else 0.0
        else:
            r2 = float(1.0 - (ss_res / ss_tot))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4) if r2 is not None else None,
        "bias": round(bias, 4),
        "sample_count": n,
    }


def parse_timestamp_iso(ts_str: Optional[str]) -> Optional[str]:
    """Validate and normalize timestamp strings to standard UTC datetime format.

    Timezone-aware strings (e.g., ISO-8601 with offset like '+05:30' or 'Z')
    are converted to UTC before formatting. Naive timestamps are assumed
    to represent UTC already and are formatted directly as '%Y-%m-%d %H:%M:%S'.

    Args:
        ts_str: ISO-8601 or parseable datetime string.

    Returns:
        Formatted UTC string '%Y-%m-%d %H:%M:%S' or None if input is empty.

    Raises:
        InvalidWindowError: If string cannot be parsed as a datetime.
    """
    if not ts_str:
        return None
    try:
        dt = pd.to_datetime(ts_str)
        if dt.tz is not None:
            dt = dt.tz_convert("UTC")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        raise InvalidWindowError(f"Invalid timestamp format '{ts_str}': {e}") from e


def fetch_paired_records(
    db_path: Optional[str] = None,
    model_version: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve joined prediction-feedback pairs from v_monitoring_records.

    Only records where feedback has been recorded (actual_energy_kwh IS NOT NULL) are returned.

    Args:
        db_path: Path to monitoring database.
        model_version: Optional filter for specific model version.
        start_time: Optional start timestamp for filtering prediction time.
        end_time: Optional end timestamp for filtering prediction time.

    Returns:
        List of paired monitoring dictionaries.

    Raises:
        InvalidWindowError: If start_time is greater than end_time.
        DatabaseQueryError: If the database is inaccessible or query fails.
    """
    path = db_path or get_db_path()
    if not os.path.exists(path):
        raise DatabaseQueryError(f"Monitoring database not found at: {path}")

    norm_start = parse_timestamp_iso(start_time)
    norm_end = parse_timestamp_iso(end_time)

    if norm_start and norm_end and norm_start > norm_end:
        raise InvalidWindowError(
            f"start_time ('{start_time}') cannot be after end_time ('{end_time}')."
        )

    query = """
        SELECT 
            prediction_id,
            prediction_timestamp,
            energy_current_hour,
            energy_previous_hour,
            energy_two_hours_ago,
            target_hour,
            target_day_of_week,
            target_is_weekend,
            predicted_energy_kwh,
            model_version,
            latency_ms,
            actual_energy_kwh,
            feedback_timestamp,
            (predicted_energy_kwh - actual_energy_kwh) AS bias_error,
            ABS(predicted_energy_kwh - actual_energy_kwh) AS abs_error
        FROM v_monitoring_records
        WHERE actual_energy_kwh IS NOT NULL
    """
    params: List[Any] = []

    if model_version:
        query += " AND model_version = ?"
        params.append(model_version)

    if norm_start:
        query += " AND datetime(prediction_timestamp) >= datetime(?)"
        params.append(norm_start)

    if norm_end:
        query += " AND datetime(prediction_timestamp) <= datetime(?)"
        params.append(norm_end)

    query += " ORDER BY prediction_timestamp ASC;"

    try:
        with get_connection(path) as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError as e:
        raise DatabaseQueryError(f"Database query failed on {path}: {e}") from e


def evaluate_degradation(
    metrics: Dict[str, Any],
    baseline: Optional[Dict[str, float]] = None,
    max_mae_increase_pct: float = DEFAULT_MAX_MAE_INCREASE_PCT,
    max_rmse_increase_pct: float = DEFAULT_MAX_RMSE_INCREASE_PCT,
    min_r2_threshold: float = DEFAULT_MIN_R2_THRESHOLD,
    max_abs_bias: float = DEFAULT_MAX_ABS_BIAS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> Dict[str, Any]:
    """Compare observed metrics against baselines and alert thresholds.

    Args:
        metrics: Dictionary returned by compute_metrics.
        baseline: Baseline dictionary with 'MAE', 'RMSE', 'R2', 'BIAS'.
        max_mae_increase_pct: Maximum permissible MAE relative increase.
        max_rmse_increase_pct: Maximum permissible RMSE relative increase.
        min_r2_threshold: Minimum permissible R² score.
        max_abs_bias: Maximum permissible absolute bias in kWh.
        min_samples: Minimum samples required before declaring degradation.

    Returns:
        Dictionary with degradation flag, status, and diagnostic reasons.
    """
    base = baseline or BASELINE_METRICS
    sample_count = metrics.get("sample_count", 0)

    # 1. Guard against small sample sizes
    if sample_count < min_samples:
        return {
            "status": "insufficient_data",
            "is_degraded": False,
            "sample_count": sample_count,
            "min_samples_required": min_samples,
            "reasons": [
                f"Sample count ({sample_count}) is below minimum evaluation threshold ({min_samples}). "
                "Insufficient feedback records to reliably evaluate performance."
            ],
            "baseline_used": base,
        }

    reasons = []

    # 2. MAE Degradation Check
    base_mae = base["MAE"]
    allowed_mae = base_mae * (1.0 + max_mae_increase_pct)
    if metrics["mae"] > allowed_mae:
        pct_diff = ((metrics["mae"] - base_mae) / base_mae) * 100.0
        reasons.append(
            f"MAE ({metrics['mae']:.4f} kWh) increased by {pct_diff:.1f}% over baseline "
            f"({base_mae:.4f} kWh), exceeding {max_mae_increase_pct * 100:.1f}% threshold."
        )

    # 3. RMSE Degradation Check
    base_rmse = base["RMSE"]
    allowed_rmse = base_rmse * (1.0 + max_rmse_increase_pct)
    if metrics["rmse"] > allowed_rmse:
        pct_diff = ((metrics["rmse"] - base_rmse) / base_rmse) * 100.0
        reasons.append(
            f"RMSE ({metrics['rmse']:.4f} kWh) increased by {pct_diff:.1f}% over baseline "
            f"({base_rmse:.4f} kWh), exceeding {max_rmse_increase_pct * 100:.1f}% threshold."
        )

    # 4. R² Score Degradation Check
    if metrics["r2"] is not None and metrics["r2"] < min_r2_threshold:
        reasons.append(
            f"R² score ({metrics['r2']:.4f}) dropped below minimum acceptable threshold ({min_r2_threshold:.4f}). "
            f"Baseline was {base['R2']:.4f}."
        )

    # 5. Prediction Bias Check
    if abs(metrics["bias"]) > max_abs_bias:
        direction = "over-prediction" if metrics["bias"] > 0 else "under-prediction"
        reasons.append(
            f"Absolute bias ({abs(metrics['bias']):.4f} kWh) exceeds limit ({max_abs_bias:.4f} kWh). "
            f"Indicates systematic model {direction}."
        )

    is_degraded = len(reasons) > 0
    status = "degraded" if is_degraded else "healthy"

    return {
        "status": status,
        "is_degraded": is_degraded,
        "sample_count": sample_count,
        "min_samples_required": min_samples,
        "reasons": reasons if is_degraded else ["All metrics within acceptable baseline thresholds."],
        "baseline_used": base,
    }


def compute_time_series_breakdown(
    records: List[Dict[str, Any]],
    interval: str = "daily",
) -> List[Dict[str, Any]]:
    """Compute performance metrics grouped by time intervals (daily or hourly).

    Args:
        records: List of paired records.
        interval: 'daily' (YYYY-MM-DD) or 'hourly' (YYYY-MM-DD HH:00).

    Returns:
        List of bucketed performance summaries chronologically sorted.
    """
    if not records:
        return []

    buckets: Dict[str, Tuple[List[float], List[float]]] = {}
    for r in records:
        ts = pd.to_datetime(r["prediction_timestamp"])
        key = ts.strftime("%Y-%m-%d") if interval == "daily" else ts.strftime("%Y-%m-%d %H:00")
        if key not in buckets:
            buckets[key] = ([], [])
        buckets[key][0].append(float(r["actual_energy_kwh"]))
        buckets[key][1].append(float(r["predicted_energy_kwh"]))

    breakdown = []
    for bucket_key in sorted(buckets.keys()):
        y_t, y_p = buckets[bucket_key]
        bucket_metrics = compute_metrics(y_t, y_p)
        breakdown.append({
            "bucket": bucket_key,
            "sample_count": len(y_t),
            "mae": bucket_metrics["mae"],
            "rmse": bucket_metrics["rmse"],
            "r2": bucket_metrics["r2"],
            "bias": bucket_metrics["bias"],
        })

    return breakdown


def evaluate_model_performance(
    db_path: Optional[str] = None,
    model_version: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    baseline: Optional[Dict[str, float]] = None,
    max_mae_increase_pct: float = DEFAULT_MAX_MAE_INCREASE_PCT,
    max_rmse_increase_pct: float = DEFAULT_MAX_RMSE_INCREASE_PCT,
    min_r2_threshold: float = DEFAULT_MIN_R2_THRESHOLD,
    max_abs_bias: float = DEFAULT_MAX_ABS_BIAS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    include_time_breakdown: bool = True,
) -> Dict[str, Any]:
    """Execute complete performance evaluation on feedback-paired prediction records.

    Args:
        db_path: Path to database.
        model_version: Target model version identifier.
        start_time: Window start timestamp.
        end_time: Window end timestamp.
        baseline: Optional baseline metrics dictionary.
        max_mae_increase_pct: Max allowed percentage increase in MAE.
        max_rmse_increase_pct: Max allowed percentage increase in RMSE.
        min_r2_threshold: Min allowed R² score.
        max_abs_bias: Max allowed absolute bias.
        min_samples: Minimum feedback records required for degradation alert.
        include_time_breakdown: Whether to compute daily performance time series.

    Returns:
        Machine-readable performance evaluation dictionary.
    """
    records = fetch_paired_records(
        db_path=db_path,
        model_version=model_version,
        start_time=start_time,
        end_time=end_time,
    )

    evaluation_time = datetime.now(timezone.utc).isoformat()
    version_label = model_version or "all_models"

    if not records:
        return {
            "evaluation_timestamp": evaluation_time,
            "model_version": version_label,
            "evaluation_window": {
                "start": start_time,
                "end": end_time,
            },
            "total_eligible_feedback_records": 0,
            "evaluated_predictions": 0,
            "status": "no_feedback_data",
            "is_degraded": False,
            "metrics": None,
            "degradation_assessment": {
                "status": "no_feedback_data",
                "is_degraded": False,
                "reasons": ["No paired prediction-feedback records found for the specified criteria."],
            },
            "thresholds": {
                "max_mae_increase_pct": max_mae_increase_pct,
                "max_rmse_increase_pct": max_rmse_increase_pct,
                "min_r2_threshold": min_r2_threshold,
                "max_abs_bias": max_abs_bias,
                "min_samples": min_samples,
            },
            "time_breakdown": [],
        }

    y_true = [r["actual_energy_kwh"] for r in records]
    y_pred = [r["predicted_energy_kwh"] for r in records]

    metrics = compute_metrics(y_true, y_pred)
    deg_assessment = evaluate_degradation(
        metrics=metrics,
        baseline=baseline,
        max_mae_increase_pct=max_mae_increase_pct,
        max_rmse_increase_pct=max_rmse_increase_pct,
        min_r2_threshold=min_r2_threshold,
        max_abs_bias=max_abs_bias,
        min_samples=min_samples,
    )

    first_ts = records[0]["prediction_timestamp"]
    last_ts = records[-1]["prediction_timestamp"]

    time_series = compute_time_series_breakdown(records, interval="daily") if include_time_breakdown else []

    return {
        "evaluation_timestamp": evaluation_time,
        "model_version": version_label,
        "evaluation_window": {
            "requested_start": start_time,
            "requested_end": end_time,
            "first_record_timestamp": first_ts,
            "last_record_timestamp": last_ts,
        },
        "total_eligible_feedback_records": len(records),
        "evaluated_predictions": len(records),
        "status": deg_assessment["status"],
        "is_degraded": deg_assessment["is_degraded"],
        "metrics": metrics,
        "degradation_assessment": deg_assessment,
        "thresholds": {
            "max_mae_increase_pct": max_mae_increase_pct,
            "max_rmse_increase_pct": max_rmse_increase_pct,
            "min_r2_threshold": min_r2_threshold,
            "max_abs_bias": max_abs_bias,
            "min_samples": min_samples,
        },
        "time_breakdown": time_series,
    }


def get_available_model_versions(db_path: Optional[str] = None) -> List[str]:
    """Retrieve distinct model versions present in the monitoring database."""
    path = db_path or get_db_path()
    if not os.path.exists(path):
        return []

    with get_connection(path) as conn:
        cursor = conn.execute("SELECT DISTINCT model_version FROM predictions ORDER BY model_version;")
        return [row[0] for row in cursor.fetchall()]


def generate_performance_report(
    db_path: Optional[str] = None,
    output_dir: str = DEFAULT_REPORT_DIR,
    model_version: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    filename: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Evaluate performance and save machine-readable JSON report to reports/performance/.

    Args:
        db_path: Database path.
        output_dir: Target output directory.
        model_version: Specific model version.
        start_time: Window start.
        end_time: Window end.
        filename: Optional custom filename.

    Returns:
        Tuple of (report_file_path, report_dict).
    """
    os.makedirs(output_dir, exist_ok=True)
    report_data = evaluate_model_performance(
        db_path=db_path,
        model_version=model_version,
        start_time=start_time,
        end_time=end_time,
    )

    ver_clean = (model_version or "all_models").replace(" ", "_")
    fname = filename or f"performance_report_{ver_clean}.json"
    report_path = os.path.join(output_dir, fname)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return report_path, report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Performance Monitoring CLI (Phase 3.4).")
    parser.add_argument("--db-path", type=str, default=None, help="Path to predictions database")
    parser.add_argument("--model-version", type=str, default=None, help="Target model version")
    parser.add_argument("--start-time", type=str, default=None, help="Start timestamp filter")
    parser.add_argument("--end-time", type=str, default=None, help="End timestamp filter")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_REPORT_DIR, help="Output directory for reports")

    args = parser.parse_args()

    print("\n=======================================================")
    print("  Smart Home Energy Advisor — Performance Monitor")
    print(f"  Target DB: {args.db_path or get_db_path()}")
    print("=======================================================\n")

    report_path, res = generate_performance_report(
        db_path=args.db_path,
        output_dir=args.output_dir,
        model_version=args.model_version,
        start_time=args.start_time,
        end_time=args.end_time,
    )

    status_str = res["status"].upper()
    print(f"Status: {status_str} (Evaluated: {res['evaluated_predictions']} predictions)")
    if res["metrics"]:
        m = res["metrics"]
        print(f"  MAE:  {m['mae']:.4f} kWh (Baseline: {BASELINE_METRICS['MAE']:.4f})")
        print(f"  RMSE: {m['rmse']:.4f} kWh (Baseline: {BASELINE_METRICS['RMSE']:.4f})")
        print(f"  R²:   {m['r2'] if m['r2'] is not None else 'N/A'} (Baseline: {BASELINE_METRICS['R2']:.4f})")
        print(f"  Bias: {m['bias']:+.4f} kWh")

    deg = res["degradation_assessment"]
    print(f"\nDegradation Assessment ({deg['status']}):")
    for r in deg["reasons"]:
        print(f"  - {r}")

    print(f"\nReport saved to: {report_path}\n")
