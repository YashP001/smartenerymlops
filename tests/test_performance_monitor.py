"""
Unit and integration tests for Model Performance Monitoring (Phase 3.4).

Covers:
- Metric calculations (MAE, RMSE, R², Bias) against known mathematical values.
- Edge cases: constant targets, zero actual energy, empty databases, small samples (< min_samples).
- Window filtering, version separation, time-based breakdown, degradation detection, and report generation.
"""

import json
import os
import sqlite3
import numpy as np
import pandas as pd
import pytest

from src.monitoring.database import init_db, log_prediction, log_feedback
from src.monitoring.performance_monitor import (
    compute_metrics,
    fetch_paired_records,
    evaluate_degradation,
    compute_time_series_breakdown,
    evaluate_model_performance,
    generate_performance_report,
    get_available_model_versions,
    parse_timestamp_iso,
    InvalidWindowError,
    DatabaseQueryError,
    BASELINE_METRICS,
)


@pytest.fixture
def mock_db(tmp_path):
    """Create a temporary initialized SQLite monitoring database."""
    db_file = str(tmp_path / "test_monitoring.db")
    init_db(db_file)
    return db_file


@pytest.fixture
def seeded_db(mock_db):
    """Seed database with deterministic prediction and feedback records."""
    base_dt = pd.to_datetime("2026-11-01T00:00:00Z")
    # 40 baseline records with small errors (MAE ~0.05, RMSE ~0.06, healthy)
    for i in range(40):
        pred_id = f"pred_base_{i:03d}"
        actual = 0.20 + (i % 5) * 0.05  # 0.20 to 0.40
        pred = actual + (0.02 if i % 2 == 0 else -0.02)
        cur_dt = base_dt + pd.Timedelta(hours=i)
        ts = cur_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        hour = cur_dt.hour
        day = cur_dt.dayofweek

        features = {
            "energy_current_hour": actual,
            "energy_previous_hour": actual - 0.01,
            "energy_two_hours_ago": actual - 0.02,
            "target_hour": hour,
            "target_day_of_week": day,
            "target_is_weekend": 1 if day >= 5 else 0,
        }
        log_prediction(
            prediction_id=pred_id,
            features=features,
            predicted_energy_kwh=pred,
            model_version="GradientBoostingRegressor",
            latency_ms=12.5,
            timestamp=ts,
            db_path=mock_db,
        )
        log_feedback(
            prediction_id=pred_id,
            actual_energy_kwh=actual,
            feedback_timestamp=ts,
            db_path=mock_db,
        )

    # 10 records for a different model version: "RandomForestRegressor"
    base_rf_dt = pd.to_datetime("2026-11-03T00:00:00Z")
    for i in range(10):
        pred_id = f"pred_rf_{i:03d}"
        actual = 0.30
        pred = 0.35
        cur_dt = base_rf_dt + pd.Timedelta(hours=i)
        ts = cur_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        features = {
            "energy_current_hour": actual,
            "energy_previous_hour": actual,
            "energy_two_hours_ago": actual,
            "target_hour": cur_dt.hour,
            "target_day_of_week": cur_dt.dayofweek,
            "target_is_weekend": 0,
        }
        log_prediction(
            prediction_id=pred_id,
            features=features,
            predicted_energy_kwh=pred,
            model_version="RandomForestRegressor",
            latency_ms=10.0,
            timestamp=ts,
            db_path=mock_db,
        )
        log_feedback(
            prediction_id=pred_id,
            actual_energy_kwh=actual,
            feedback_timestamp=ts,
            db_path=mock_db,
        )

    return mock_db


# ─── 1. Mathematical Accuracy Tests ───────────────────────────────────

class TestMetricCalculations:
    """Tests compute_metrics against exact known mathematical values."""

    def test_known_values_metrics(self):
        """Verify MAE, RMSE, R², and Bias against exact hand-calculated results."""
        # y_true = [1.0, 2.0, 3.0, 4.0]
        # y_pred = [1.1, 1.9, 3.2, 3.8]
        # errors = [+0.1, -0.1, +0.2, -0.2]
        # abs_errors = [0.1, 0.1, 0.2, 0.2] -> MAE = 0.6 / 4 = 0.15
        # sq_errors = [0.01, 0.01, 0.04, 0.04] = 0.10 -> MSE = 0.025 -> RMSE = sqrt(0.025) ~= 0.1581
        # bias = (0.1 - 0.1 + 0.2 - 0.2) / 4 = 0.0
        # y_true mean = 2.5; ss_tot = (1.5^2 + 0.5^2 + 0.5^2 + 1.5^2) = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
        # ss_res = 0.10 -> R^2 = 1 - (0.10 / 5.0) = 1 - 0.02 = 0.98
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 1.9, 3.2, 3.8]

        metrics = compute_metrics(y_true, y_pred)
        assert metrics["mae"] == 0.15
        assert metrics["rmse"] == 0.1581
        assert metrics["bias"] == 0.0
        assert metrics["r2"] == 0.98
        assert metrics["sample_count"] == 4

    def test_zero_valued_actual_energy_safe(self):
        """Zero actual energy readings are handled without ZeroDivisionError or NaN."""
        y_true = [0.0, 0.0, 0.0, 0.0]
        y_pred = [0.05, 0.02, 0.04, 0.01]

        metrics = compute_metrics(y_true, y_pred)
        assert metrics["mae"] == 0.03
        assert metrics["rmse"] > 0
        assert metrics["bias"] == 0.03
        assert np.isfinite(metrics["mae"])
        assert np.isfinite(metrics["rmse"])

    def test_constant_target_r2_handling(self):
        """Constant target handling: ss_tot == 0 returns 1.0 if perfect, else 0.0."""
        # Perfect match on constant target
        m_perf = compute_metrics([0.2, 0.2, 0.2], [0.2, 0.2, 0.2])
        assert m_perf["r2"] == 1.0
        assert m_perf["mae"] == 0.0

        # Mismatch on constant target
        m_mismatch = compute_metrics([0.2, 0.2, 0.2], [0.3, 0.3, 0.3])
        assert m_mismatch["r2"] == 0.0
        assert m_mismatch["mae"] == 0.1

    def test_single_sample_r2_returns_none(self):
        """N=1 point has undefined variance, returns r2=None."""
        m = compute_metrics([1.0], [1.1])
        assert m["mae"] == 0.1
        assert m["r2"] is None
        assert m["sample_count"] == 1

    def test_empty_or_mismatched_inputs_raise(self):
        """Empty arrays or mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="empty arrays"):
            compute_metrics([], [])

        with pytest.raises(ValueError, match="Mismatched array lengths"):
            compute_metrics([1.0, 2.0], [1.0])

        with pytest.raises(ValueError, match="NaN or infinite"):
            compute_metrics([1.0, np.nan], [1.0, 2.0])


# ─── 2. Database & Window Filtering Tests ──────────────────────────────

class TestDatabaseQueriesAndFiltering:
    """Tests data retrieval, window filtering, and model version separation."""

    def test_fetch_paired_records_basic(self, seeded_db):
        """Fetch all paired records from seeded database."""
        records = fetch_paired_records(db_path=seeded_db)
        assert len(records) == 50  # 40 GBM + 10 RF

    def test_model_version_separation(self, seeded_db):
        """Filter by model_version returns only the targeted version's records."""
        rf_records = fetch_paired_records(db_path=seeded_db, model_version="RandomForestRegressor")
        assert len(rf_records) == 10
        assert all(r["model_version"] == "RandomForestRegressor" for r in rf_records)

        gbm_records = fetch_paired_records(db_path=seeded_db, model_version="GradientBoostingRegressor")
        assert len(gbm_records) == 40
        assert all(r["model_version"] == "GradientBoostingRegressor" for r in gbm_records)

    def test_date_window_filtering(self, seeded_db):
        """Window filtering correctly slices records between start_time and end_time."""
        start = "2026-11-01T00:00:00Z"
        end = "2026-11-01T12:00:00Z"
        records = fetch_paired_records(
            db_path=seeded_db,
            model_version="GradientBoostingRegressor",
            start_time=start,
            end_time=end,
        )
        assert len(records) == 13  # hours 0 through 12
        for r in records:
            assert start <= r["prediction_timestamp"] <= end

    def test_invalid_window_raises(self, seeded_db):
        """start_time after end_time raises InvalidWindowError."""
        with pytest.raises(InvalidWindowError, match="cannot be after end_time"):
            fetch_paired_records(
                db_path=seeded_db,
                start_time="2026-11-02T00:00:00Z",
                end_time="2026-11-01T00:00:00Z",
            )

    def test_nonexistent_database_raises_query_error(self):
        """Accessing a nonexistent database file raises DatabaseQueryError."""
        with pytest.raises(DatabaseQueryError):
            fetch_paired_records(db_path="nonexistent_db_xyz.db")

    def test_get_available_model_versions(self, seeded_db):
        """Retrieve distinct model versions registered in database."""
        versions = get_available_model_versions(db_path=seeded_db)
        assert "GradientBoostingRegressor" in versions
        assert "RandomForestRegressor" in versions


# ─── 2b. Timezone Normalization Tests ─────────────────────────────────

class TestTimestampParsingAndTimezones:
    """Tests for timezone normalization and UTC conversion in window filtering."""

    def test_timezone_conversion_ist_to_utc(self):
        """Non-UTC offsets (e.g. IST +05:30) are converted to UTC correctly."""
        ts_ist = "2026-11-01T12:00:00+05:30"
        ts_utc = "2026-11-01T06:30:00Z"
        norm_ist = parse_timestamp_iso(ts_ist)
        norm_utc = parse_timestamp_iso(ts_utc)
        assert norm_ist == "2026-11-01 06:30:00"
        assert norm_utc == "2026-11-01 06:30:00"
        assert norm_ist == norm_utc

    def test_naive_timestamp_retains_utc_behavior(self):
        """Naive timestamp strings retain existing behavior and represent UTC."""
        ts_naive = "2026-11-01 06:30:00"
        norm_naive = parse_timestamp_iso(ts_naive)
        assert norm_naive == "2026-11-01 06:30:00"
        assert norm_naive == parse_timestamp_iso("2026-11-01T06:30:00Z")

    def test_invalid_timestamp_raises_invalid_window_error(self):
        """Invalid timestamp strings raise InvalidWindowError."""
        with pytest.raises(InvalidWindowError, match="Invalid timestamp format"):
            parse_timestamp_iso("invalid-timestamp-string")

    def test_database_window_filtering_timezone_equivalence(self, seeded_db):
        """Equivalent UTC and IST filter boundaries return identical prediction records."""
        # seeded_db contains hourly predictions from 2026-11-01T00:00:00Z to 2026-11-02T15:00:00Z
        # Window 00:00:00Z to 12:00:00Z in UTC is 05:30:00+05:30 to 17:30:00+05:30 in IST
        records_utc = fetch_paired_records(
            db_path=seeded_db,
            model_version="GradientBoostingRegressor",
            start_time="2026-11-01T00:00:00Z",
            end_time="2026-11-01T12:00:00Z",
        )
        records_ist = fetch_paired_records(
            db_path=seeded_db,
            model_version="GradientBoostingRegressor",
            start_time="2026-11-01T05:30:00+05:30",
            end_time="2026-11-01T17:30:00+05:30",
        )
        assert len(records_utc) == 13
        assert len(records_ist) == 13
        assert [r["prediction_id"] for r in records_utc] == [r["prediction_id"] for r in records_ist]


# ─── 3. Degradation Detection Tests ───────────────────────────────────

class TestDegradationDetection:
    """Tests performance degradation detection logic and threshold enforcement."""

    def test_healthy_performance_within_thresholds(self):
        """Performance within thresholds yields healthy status and no degradation."""
        metrics = {
            "mae": 0.0700,   # Baseline is 0.0739 (lower is better)
            "rmse": 0.1400,  # Baseline is 0.1512
            "r2": 0.4000,    # Baseline is 0.3723
            "bias": 0.0100,  # Below max_abs_bias 0.05
            "sample_count": 50,
        }
        res = evaluate_degradation(metrics)
        assert res["status"] == "healthy"
        assert res["is_degraded"] is False

    def test_insufficient_samples_distinguished_from_healthy(self):
        """Small sample size (< min_samples) reports insufficient_data, not healthy."""
        metrics = {
            "mae": 0.0500,
            "rmse": 0.0800,
            "r2": 0.5000,
            "bias": 0.0100,
            "sample_count": 15,  # Below min_samples=30
        }
        res = evaluate_degradation(metrics, min_samples=30)
        assert res["status"] == "insufficient_data"
        assert res["is_degraded"] is False
        assert "below minimum evaluation threshold" in res["reasons"][0]

    def test_mae_degradation_detected(self):
        """MAE elevated by > 25% over baseline triggers degradation alert with reason."""
        metrics = {
            "mae": 0.1050,   # 42% higher than baseline 0.0739
            "rmse": 0.1550,
            "r2": 0.3500,
            "bias": 0.0100,
            "sample_count": 50,
        }
        res = evaluate_degradation(metrics, max_mae_increase_pct=0.25)
        assert res["status"] == "degraded"
        assert res["is_degraded"] is True
        assert any("MAE" in r and "exceeding 25.0% threshold" in r for r in res["reasons"])

    def test_r2_drop_degradation_detected(self):
        """R² dropping below 0.10 triggers degradation alert."""
        metrics = {
            "mae": 0.0750,
            "rmse": 0.1550,
            "r2": 0.0500,    # Below min_r2_threshold 0.10
            "bias": 0.0100,
            "sample_count": 50,
        }
        res = evaluate_degradation(metrics, min_r2_threshold=0.10)
        assert res["status"] == "degraded"
        assert res["is_degraded"] is True
        assert any("R² score" in r for r in res["reasons"])

    def test_bias_degradation_detected(self):
        """Systematic bias exceeding 0.05 kWh triggers degradation alert."""
        metrics = {
            "mae": 0.0800,
            "rmse": 0.1600,
            "r2": 0.3500,
            "bias": 0.0750,  # Exceeds 0.05 kWh limit
            "sample_count": 50,
        }
        res = evaluate_degradation(metrics, max_abs_bias=0.05)
        assert res["status"] == "degraded"
        assert res["is_degraded"] is True
        assert any("bias" in r.lower() for r in res["reasons"])

    def test_rmse_independent_degradation_detected(self, mock_db):
        """RMSE independently exceeding threshold triggers degradation while MAE, R², Bias remain healthy."""
        # 50 records to satisfy min_samples=30
        # 46 records with small symmetric errors (+/- 0.02)
        # 4 outlier records with larger symmetric errors (+/- 0.70)
        # Resulting in low MAE (~0.0744 <= 0.0924), zero Bias (0.0 <= 0.05), high R² (~0.8592 >= 0.10),
        # but elevated RMSE (~0.1989 > 0.1890, +31.5% > +25%).
        y_true = np.linspace(0.2, 2.0, 50)
        errors = np.array([0.02 if i % 2 == 0 else -0.02 for i in range(46)] + [0.70, -0.70, 0.70, -0.70])
        base_dt = pd.to_datetime("2026-11-01T00:00:00Z")

        for i in range(50):
            pred_id = f"pred_rmse_{i:03d}"
            actual = float(y_true[i])
            pred = float(actual + errors[i])
            ts = (base_dt + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
            features = {
                "energy_current_hour": actual,
                "energy_previous_hour": actual,
                "energy_two_hours_ago": actual,
                "target_hour": 12,
                "target_day_of_week": 2,
                "target_is_weekend": 0,
            }
            log_prediction(
                prediction_id=pred_id,
                features=features,
                predicted_energy_kwh=pred,
                model_version="GradientBoostingRegressor",
                latency_ms=10.0,
                timestamp=ts,
                db_path=mock_db,
            )
            log_feedback(
                prediction_id=pred_id,
                actual_energy_kwh=actual,
                feedback_timestamp=ts,
                db_path=mock_db,
            )

        res = evaluate_model_performance(db_path=mock_db, model_version="GradientBoostingRegressor")

        # Verify sample count satisfies min_samples
        assert res["evaluated_predictions"] == 50
        assert res["status"] == "degraded"
        assert res["is_degraded"] is True

        # Verify MAE, R², Bias remain within healthy thresholds
        metrics = res["metrics"]
        assert metrics["mae"] <= BASELINE_METRICS["MAE"] * 1.25  # 0.0744 <= 0.0924
        assert metrics["r2"] >= 0.10                             # 0.8592 >= 0.10
        assert abs(metrics["bias"]) <= 0.05                      # 0.0 <= 0.05

        # Verify RMSE independently crossed its threshold
        assert metrics["rmse"] > BASELINE_METRICS["RMSE"] * 1.25  # 0.1989 > 0.1890

        # Verify reason specifically identifies RMSE and nothing else
        reasons = res["degradation_assessment"]["reasons"]
        assert len(reasons) == 1
        assert "RMSE" in reasons[0]
        assert "MAE" not in reasons[0]
        assert "R²" not in reasons[0]
        assert "bias" not in reasons[0].lower()


# ─── 4. End-to-End Evaluation & Reporting Tests ────────────────────────

class TestEndToEndEvaluationAndReporting:
    """Tests full pipeline evaluation and JSON reporting."""

    def test_empty_database_returns_clean_summary(self, mock_db):
        """Empty monitoring store returns no_feedback_data status without crashing."""
        res = evaluate_model_performance(db_path=mock_db)
        assert res["status"] == "no_feedback_data"
        assert res["evaluated_predictions"] == 0
        assert res["metrics"] is None
        assert res["is_degraded"] is False

    def test_evaluate_seeded_database(self, seeded_db):
        """Evaluate seeded GBM records end-to-end."""
        res = evaluate_model_performance(
            db_path=seeded_db,
            model_version="GradientBoostingRegressor",
            min_samples=30,
        )
        assert res["status"] == "healthy"
        assert res["evaluated_predictions"] == 40
        assert res["metrics"]["mae"] == 0.02
        assert res["metrics"]["rmse"] == 0.02
        assert len(res["time_breakdown"]) > 0

    def test_generate_performance_report_json(self, seeded_db, tmp_path):
        """generate_performance_report creates machine-readable JSON with full schema."""
        out_dir = str(tmp_path / "reports_perf")
        report_path, report_dict = generate_performance_report(
            db_path=seeded_db,
            output_dir=out_dir,
            model_version="GradientBoostingRegressor",
        )

        assert os.path.exists(report_path)
        assert report_dict["evaluated_predictions"] == 40

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["model_version"] == "GradientBoostingRegressor"
        assert "evaluation_timestamp" in data
        assert "metrics" in data
        assert "degradation_assessment" in data
        assert "thresholds" in data
        assert "time_breakdown" in data
