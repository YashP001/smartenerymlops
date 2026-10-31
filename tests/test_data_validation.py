"""
Tests for Data Quality and Validation Module (src/data_validation.py).

Member 1 Scope: Dataset Management and Data Quality
Coordinates with Member 3's test suite (test_features.py, test_model.py, test_api.py)
without duplicating test ownership.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_validation import (
    REQUIRED_COLUMNS,
    NUMERIC_COLUMNS,
    PHYSICAL_VALID_RANGES,
    SUSPICIOUS_DOMAIN_BOUNDS,
    check_schema_and_types,
    check_missing_values,
    check_duplicate_records,
    check_duplicate_meter_timestamps,
    check_timestamp_validity,
    check_timestamp_intervals,
    check_invalid_numerical_values,
    check_suspicious_values_and_anomalies,
    calculate_statistical_summaries,
    validate_dataset,
    generate_markdown_report,
)


@pytest.fixture
def clean_synthetic_data() -> pd.DataFrame:
    """Fixture providing a clean synthetic smart-meter dataset."""
    timestamps = pd.date_range("2021-01-01 00:00:00", periods=20, freq="3min")
    return pd.DataFrame({
        "x_Timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "t_kWh": [0.005] * 20,
        "z_Avg Voltage (Volt)": [235.0] * 20,
        "z_Avg Current (Amp)": [0.5] * 20,
        "y_Freq (Hz)": [50.0] * 20,
        "meter": ["MH01"] * 20,
    })


class TestSchemaValidation:
    """Tests for schema, column existence, and column type validation."""

    def test_valid_schema(self, clean_synthetic_data):
        res = check_schema_and_types(clean_synthetic_data)
        assert res["passed"] is True
        assert len(res["missing_columns"]) == 0
        assert len(res["type_issues"]) == 0

    def test_single_missing_column(self, clean_synthetic_data):
        df_missing = clean_synthetic_data.drop(columns=["t_kWh"])
        res = check_schema_and_types(df_missing)
        assert res["passed"] is False
        assert "t_kWh" in res["missing_columns"]

    def test_multiple_missing_columns(self, clean_synthetic_data):
        df_missing = clean_synthetic_data.drop(columns=["t_kWh", "meter", "z_Avg Current (Amp)"])
        res = check_schema_and_types(df_missing)
        assert res["passed"] is False
        assert len(res["missing_columns"]) == 3
        assert set(res["missing_columns"]) == {"t_kWh", "meter", "z_Avg Current (Amp)"}

    def test_unexpected_extra_columns(self, clean_synthetic_data):
        df_extra = clean_synthetic_data.copy()
        df_extra["extra_feature"] = 123
        res = check_schema_and_types(df_extra)
        # Extra columns do not fail schema check but are reported
        assert "extra_feature" in res["unexpected_columns"]

    def test_non_numeric_type_issue_voltage(self, clean_synthetic_data):
        df_corrupt = clean_synthetic_data.copy()
        df_corrupt.loc[0, "z_Avg Voltage (Volt)"] = "corrupted_text"
        res = check_schema_and_types(df_corrupt)
        assert res["passed"] is False
        assert "z_Avg Voltage (Volt)" in res["type_issues"]

    def test_non_numeric_type_issue_energy(self, clean_synthetic_data):
        df_corrupt = clean_synthetic_data.copy()
        df_corrupt.loc[1, "t_kWh"] = "not_a_number"
        res = check_schema_and_types(df_corrupt)
        assert res["passed"] is False
        assert "t_kWh" in res["type_issues"]


class TestMissingValuesValidation:
    """Tests for missing, null, empty string, and infinite values."""

    def test_no_missing_values(self, clean_synthetic_data):
        res = check_missing_values(clean_synthetic_data)
        assert res["passed"] is True
        assert res["total_missing"] == 0

    def test_null_value_detected(self, clean_synthetic_data):
        df_null = clean_synthetic_data.copy()
        df_null.loc[2, "t_kWh"] = np.nan
        res = check_missing_values(df_null)
        assert res["passed"] is False
        assert res["total_missing"] == 1
        assert "t_kWh" in res["per_column"]

    def test_infinite_value_detected(self, clean_synthetic_data):
        df_inf = clean_synthetic_data.copy()
        df_inf.loc[3, "z_Avg Voltage (Volt)"] = np.inf
        res = check_missing_values(df_inf)
        assert res["passed"] is False
        assert res["total_missing"] == 1
        assert res["per_column"]["z_Avg Voltage (Volt)"]["infinite_count"] == 1

    def test_negative_infinite_detected(self, clean_synthetic_data):
        df_ninf = clean_synthetic_data.copy()
        df_ninf.loc[4, "y_Freq (Hz)"] = -np.inf
        res = check_missing_values(df_ninf)
        assert res["passed"] is False
        assert res["total_missing"] == 1

    def test_empty_string_detected(self, clean_synthetic_data):
        df_empty = clean_synthetic_data.copy()
        df_empty.loc[1, "meter"] = "   "
        res = check_missing_values(df_empty)
        assert res["passed"] is False
        assert res["total_missing"] == 1


class TestDuplicateValidation:
    """Tests for duplicate records and duplicate meter-timestamp pairs."""

    def test_no_duplicates(self, clean_synthetic_data):
        res_full = check_duplicate_records(clean_synthetic_data)
        res_subset = check_duplicate_meter_timestamps(clean_synthetic_data)
        assert res_full["passed"] is True
        assert res_subset["passed"] is True

    def test_full_row_duplicates(self, clean_synthetic_data):
        df_dup = pd.concat([clean_synthetic_data, clean_synthetic_data.iloc[[0]]], ignore_index=True)
        res = check_duplicate_records(df_dup)
        assert res["passed"] is False
        assert res["total_duplicates"] == 1

    def test_meter_timestamp_collision(self, clean_synthetic_data):
        df_dup = clean_synthetic_data.copy()
        # Duplicate timestamp for same meter with different voltage
        new_row = clean_synthetic_data.iloc[0].copy()
        new_row["z_Avg Voltage (Volt)"] = 240.0
        df_dup = pd.concat([df_dup, pd.DataFrame([new_row])], ignore_index=True)

        res = check_duplicate_meter_timestamps(df_dup)
        assert res["passed"] is False
        assert res["total_duplicates"] == 1
        assert len(res["sample_duplicates"]) >= 1

    def test_missing_subset_columns_in_duplicates(self, clean_synthetic_data):
        df_no_meter = clean_synthetic_data.drop(columns=["meter"])
        res = check_duplicate_meter_timestamps(df_no_meter)
        assert res["passed"] is False
        assert "missing" in res["error"]


class TestTimestampValidation:
    """Tests for timestamp parsing, range, and interval analysis."""

    def test_valid_timestamps(self, clean_synthetic_data):
        res = check_timestamp_validity(clean_synthetic_data)
        assert res["passed"] is True
        assert res["unparseable_count"] == 0
        assert res["out_of_bounds_years"] == 0
        assert res["min_timestamp"] is not None
        assert res["max_timestamp"] is not None

    def test_unparseable_timestamp(self, clean_synthetic_data):
        df_bad = clean_synthetic_data.copy()
        df_bad.loc[0, "x_Timestamp"] = "invalid_date_format_2021"
        res = check_timestamp_validity(df_bad)
        assert res["passed"] is False
        assert res["unparseable_count"] == 1

    def test_out_of_bounds_year_detected(self, clean_synthetic_data):
        df_year = clean_synthetic_data.copy()
        df_year.loc[0, "x_Timestamp"] = "1990-01-01 00:00:00"
        res = check_timestamp_validity(df_year, min_year=2015, max_year=2030)
        assert res["passed"] is False
        assert res["out_of_bounds_years"] == 1

    def test_interval_regular(self, clean_synthetic_data):
        res = check_timestamp_intervals(clean_synthetic_data, expected_interval_minutes=3)
        assert res["passed"] is True
        assert res["unexpected_gaps_count"] == 0
        assert res["exact_expected_intervals"] == 19

    def test_interval_gap_detected(self, clean_synthetic_data):
        # Drop row 10 to create a 6-minute gap between row 9 and row 11
        df_gap = clean_synthetic_data.drop(index=[10]).reset_index(drop=True)
        res = check_timestamp_intervals(df_gap, expected_interval_minutes=3)
        assert res["passed"] is False
        assert res["unexpected_gaps_count"] == 1
        assert res["total_estimated_missing_readings"] == 1

    def test_short_interval_detected(self, clean_synthetic_data):
        # Insert a reading at 1.5 minutes
        df_short = clean_synthetic_data.copy()
        new_row = clean_synthetic_data.iloc[0].copy()
        new_row["x_Timestamp"] = "2021-01-01 00:01:30"
        df_short = pd.concat([df_short, pd.DataFrame([new_row])], ignore_index=True)
        res = check_timestamp_intervals(df_short, expected_interval_minutes=3)
        assert res["passed"] is False
        assert res["short_intervals_count"] >= 1


class TestNumericalAndAnomalyValidation:
    """Tests for physical validity (critical errors) vs domain anomalies (warnings)."""

    def test_physically_valid_data(self, clean_synthetic_data):
        res = check_invalid_numerical_values(clean_synthetic_data)
        assert res["passed"] is True
        assert res["total_invalid_values"] == 0

    def test_negative_energy_flagged(self, clean_synthetic_data):
        df_neg = clean_synthetic_data.copy()
        df_neg.loc[0, "t_kWh"] = -0.05
        res = check_invalid_numerical_values(df_neg)
        assert res["passed"] is False
        assert res["total_invalid_values"] == 1
        assert res["invalid_details"]["t_kWh"]["below_physical_min"] == 1

    def test_negative_voltage_flagged(self, clean_synthetic_data):
        df_neg = clean_synthetic_data.copy()
        df_neg.loc[0, "z_Avg Voltage (Volt)"] = -15.0
        res = check_invalid_numerical_values(df_neg)
        assert res["passed"] is False
        assert res["total_invalid_values"] == 1

    def test_negative_current_flagged(self, clean_synthetic_data):
        df_neg = clean_synthetic_data.copy()
        df_neg.loc[0, "z_Avg Current (Amp)"] = -2.0
        res = check_invalid_numerical_values(df_neg)
        assert res["passed"] is False
        assert res["total_invalid_values"] == 1

    def test_negative_frequency_flagged(self, clean_synthetic_data):
        df_neg = clean_synthetic_data.copy()
        df_neg.loc[0, "y_Freq (Hz)"] = -5.0
        res = check_invalid_numerical_values(df_neg)
        assert res["passed"] is False
        assert res["total_invalid_values"] == 1

    def test_suspicious_overvoltage_anomaly(self, clean_synthetic_data):
        df_surge = clean_synthetic_data.copy()
        df_surge.loc[0, "z_Avg Voltage (Volt)"] = 320.0  # > 300V
        res = check_suspicious_values_and_anomalies(df_surge)
        assert res["passed"] is False
        assert res["anomalies"]["overvoltage_spikes"]["count"] == 1

    def test_zero_voltage_with_consumption_anomaly(self, clean_synthetic_data):
        df_sensor_lag = clean_synthetic_data.copy()
        df_sensor_lag.loc[0, "z_Avg Voltage (Volt)"] = 0.0
        df_sensor_lag.loc[0, "t_kWh"] = 0.05
        res = check_suspicious_values_and_anomalies(df_sensor_lag)
        assert res["passed"] is False
        assert res["anomalies"]["zero_voltage_with_consumption"]["count"] == 1

    def test_zero_frequency_with_consumption_anomaly(self, clean_synthetic_data):
        df_lag = clean_synthetic_data.copy()
        df_lag.loc[0, "y_Freq (Hz)"] = 0.0
        df_lag.loc[0, "t_kWh"] = 0.02
        res = check_suspicious_values_and_anomalies(df_lag)
        assert res["passed"] is False
        assert res["anomalies"]["zero_frequency_with_consumption"]["count"] == 1

    def test_brownout_undervoltage_anomaly(self, clean_synthetic_data):
        df_brownout = clean_synthetic_data.copy()
        df_brownout.loc[0, "z_Avg Voltage (Volt)"] = 120.0  # < 150V non-zero
        res = check_suspicious_values_and_anomalies(df_brownout)
        assert res["passed"] is False
        assert res["anomalies"]["brownout_undervoltage"]["count"] == 1


class TestStatisticalSummaries:
    """Tests for distribution summaries and quantiles."""

    def test_statistics_calculated(self, clean_synthetic_data):
        res = calculate_statistical_summaries(clean_synthetic_data)
        assert "feature_statistics" in res
        assert "t_kWh" in res["feature_statistics"]
        stats_kwh = res["feature_statistics"]["t_kWh"]
        assert stats_kwh["count"] == 20
        assert stats_kwh["min"] == 0.005
        assert stats_kwh["max"] == 0.005
        assert stats_kwh["mean"] == 0.005
        assert stats_kwh["zero_count"] == 0

        assert res["meter_summary"]["unique_meters"] == 1
        assert res["meter_summary"]["total_records"] == 20


class TestEndToEndReportGeneration:
    """Tests for full validation pipeline and markdown output."""

    def test_validate_dataset_clean_e2e(self, clean_synthetic_data):
        report = validate_dataset(clean_synthetic_data)
        assert report["status"]["all_passed"] is True
        assert report["status"]["has_critical_errors"] is False
        assert report["status"]["has_anomalies"] is False

    def test_validate_dataset_with_anomalies(self, clean_synthetic_data):
        # Add a domain anomaly (overvoltage) but no critical errors
        df_anomaly = clean_synthetic_data.copy()
        df_anomaly.loc[0, "z_Avg Voltage (Volt)"] = 310.0
        report = validate_dataset(df_anomaly)
        assert report["status"]["has_critical_errors"] is False
        assert report["status"]["has_anomalies"] is True
        assert report["status"]["all_passed"] is False

    def test_validate_dataset_with_critical_error(self, clean_synthetic_data):
        # Add a critical error (negative consumption)
        df_error = clean_synthetic_data.copy()
        df_error.loc[0, "t_kWh"] = -1.0
        report = validate_dataset(df_error)
        assert report["status"]["has_critical_errors"] is True
        assert report["status"]["all_passed"] is False

    def test_generate_markdown_report_file(self, clean_synthetic_data, tmp_path):
        report = validate_dataset(clean_synthetic_data)
        out_file = str(tmp_path / "test_report.md")
        content = generate_markdown_report(report, output_path=out_file)
        assert os.path.exists(out_file)
        assert "# Dataset Quality and Validation Report" in content
        assert "## 1. Executive Summary" in content
        assert "## 5. Statistical Summary" in content
