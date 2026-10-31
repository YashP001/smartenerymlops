"""
Tests for the feature engineering module.

Member 3 deliverable: Tests for src/features.py as specified in
Phase 2 requirements (PDF §4, Member 3).
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add project root for imports.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_data,
    add_time_features,
    add_history_features,
    chronological_split,
    validate_inference_input,
    prepare_feature_vector,
)


# ─── Constants tests ─────────────────────────────────────────────────

class TestFeatureConstants:
    """Verify that canonical constants are correctly defined."""

    def test_feature_columns_count(self):
        assert len(FEATURE_COLUMNS) == 6

    def test_feature_columns_names(self):
        expected = [
            "energy_current_hour",
            "energy_previous_hour",
            "energy_two_hours_ago",
            "target_hour",
            "target_day_of_week",
            "target_is_weekend",
        ]
        assert FEATURE_COLUMNS == expected

    def test_target_column(self):
        assert TARGET_COLUMN == "target_next_hour"


# ─── validate_data tests ─────────────────────────────────────────────

class TestValidateData:
    """Tests for the validate_data function."""

    def test_valid_input(self):
        df = pd.DataFrame({
            "timestamp": ["2021-01-01 00:00", "2021-01-01 01:00"],
            "energy": [0.1, 0.2],
        })
        result = validate_data(df, "timestamp", "energy")
        assert len(result) == 2
        assert result["timestamp"].dtype == "datetime64[ns]"

    def test_missing_columns(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_data(df, "timestamp", "energy")

    def test_duplicate_timestamps(self):
        df = pd.DataFrame({
            "timestamp": ["2021-01-01 00:00", "2021-01-01 00:00"],
            "energy": [0.1, 0.2],
        })
        with pytest.raises(ValueError, match="Duplicate observations"):
            validate_data(df, "timestamp", "energy")

    def test_drops_na_values(self):
        df = pd.DataFrame({
            "timestamp": ["2021-01-01 00:00", None, "2021-01-01 02:00"],
            "energy": [0.1, 0.2, 0.3],
        })
        result = validate_data(df, "timestamp", "energy")
        assert len(result) == 2

    def test_sorts_chronologically(self):
        df = pd.DataFrame({
            "timestamp": ["2021-01-01 02:00", "2021-01-01 01:00"],
            "energy": [0.2, 0.1],
        })
        result = validate_data(df, "timestamp", "energy")
        assert result["energy"].iloc[0] == 0.1


# ─── add_time_features tests ────────────────────────────────────────

class TestAddTimeFeatures:
    """Tests for the add_time_features function."""

    def test_creates_expected_columns(self):
        df = pd.DataFrame({
            "hour": pd.to_datetime(["2021-01-04 14:00"])
        })
        result = add_time_features(df, "hour")
        assert "target_hour" in result.columns
        assert "target_day_of_week" in result.columns
        assert "target_is_weekend" in result.columns

    def test_offset_applies(self):
        # 2021-01-04 is a Monday (dayofweek=0), hour 14.
        # With offset=1, target is hour 15, still Monday.
        df = pd.DataFrame({
            "hour": pd.to_datetime(["2021-01-04 14:00"])
        })
        result = add_time_features(df, "hour", offset_hours=1)
        assert result["target_hour"].iloc[0] == 15
        assert result["target_day_of_week"].iloc[0] == 0
        assert result["target_is_weekend"].iloc[0] == 0

    def test_weekend_detection(self):
        # 2021-01-02 is Saturday (dayofweek=5).
        df = pd.DataFrame({
            "hour": pd.to_datetime(["2021-01-02 10:00"])
        })
        result = add_time_features(df, "hour", offset_hours=0)
        assert result["target_is_weekend"].iloc[0] == 1


# ─── add_history_features tests ─────────────────────────────────────

class TestAddHistoryFeatures:
    """Tests for the add_history_features function."""

    def test_creates_expected_columns(self):
        df = pd.DataFrame({"energy_kWh": [0.1, 0.2, 0.3, 0.4]})
        result = add_history_features(df, "energy_kWh")
        assert "energy_current_hour" in result.columns
        assert "energy_previous_hour" in result.columns
        assert "energy_two_hours_ago" in result.columns

    def test_lag_values_correct(self):
        df = pd.DataFrame({"energy_kWh": [0.1, 0.2, 0.3, 0.4]})
        result = add_history_features(df, "energy_kWh")
        # Row index 2: current=0.3, previous=0.2, two_ago=0.1
        assert result["energy_current_hour"].iloc[2] == 0.3
        assert result["energy_previous_hour"].iloc[2] == 0.2
        assert result["energy_two_hours_ago"].iloc[2] == 0.1

    def test_first_rows_have_nans(self):
        df = pd.DataFrame({"energy_kWh": [0.1, 0.2, 0.3]})
        result = add_history_features(df, "energy_kWh")
        assert pd.isna(result["energy_previous_hour"].iloc[0])
        assert pd.isna(result["energy_two_hours_ago"].iloc[0])
        assert pd.isna(result["energy_two_hours_ago"].iloc[1])


# ─── chronological_split tests ───────────────────────────────────────

class TestChronologicalSplit:
    """Tests for the chronological_split function."""

    def test_split_sizes(self):
        dates = pd.date_range("2021-01-01", periods=100, freq="h")
        df = pd.DataFrame({"ts": dates, "val": range(100)})
        train, val, test = chronological_split(df, "ts")
        assert len(train) + len(val) + len(test) == 100

    def test_no_overlap(self):
        dates = pd.date_range("2021-01-01", periods=100, freq="h")
        df = pd.DataFrame({"ts": dates, "val": range(100)})
        train, val, test = chronological_split(df, "ts")
        assert train["ts"].max() < val["ts"].min()
        assert val["ts"].max() < test["ts"].min()

    def test_invalid_ratios(self):
        dates = pd.date_range("2021-01-01", periods=10, freq="h")
        df = pd.DataFrame({"ts": dates, "val": range(10)})
        with pytest.raises(ValueError):
            chronological_split(df, "ts", train_ratio=0.9, validation_ratio=0.2)


# ─── validate_inference_input tests ──────────────────────────────────

class TestValidateInferenceInput:
    """Tests for the validate_inference_input function."""

    @pytest.fixture
    def valid_input(self):
        return {
            "energy_current_hour": 0.05,
            "energy_previous_hour": 0.04,
            "energy_two_hours_ago": 0.06,
            "target_hour": 14,
            "target_day_of_week": 2,
            "target_is_weekend": 0,
        }

    def test_valid_input_passes(self, valid_input):
        result = validate_inference_input(valid_input)
        assert len(result) == 6
        assert all(isinstance(v, float) for v in result.values())

    def test_missing_feature_raises(self, valid_input):
        del valid_input["energy_current_hour"]
        with pytest.raises(ValueError, match="Missing required features"):
            validate_inference_input(valid_input)

    def test_negative_energy_raises(self, valid_input):
        valid_input["energy_current_hour"] = -0.1
        with pytest.raises(ValueError, match="non-negative"):
            validate_inference_input(valid_input)

    def test_invalid_hour_raises(self, valid_input):
        valid_input["target_hour"] = 25
        with pytest.raises(ValueError, match="target_hour"):
            validate_inference_input(valid_input)

    def test_invalid_day_raises(self, valid_input):
        valid_input["target_day_of_week"] = 7
        with pytest.raises(ValueError, match="target_day_of_week"):
            validate_inference_input(valid_input)

    def test_invalid_weekend_flag_raises(self, valid_input):
        valid_input["target_is_weekend"] = 2
        with pytest.raises(ValueError, match="target_is_weekend"):
            validate_inference_input(valid_input)

    def test_non_numeric_raises(self, valid_input):
        valid_input["energy_current_hour"] = "abc"
        with pytest.raises(ValueError, match="must be numeric"):
            validate_inference_input(valid_input)


# ─── prepare_feature_vector tests ────────────────────────────────────

class TestPrepareFeatureVector:
    """Tests for the prepare_feature_vector function."""

    def test_output_shape(self):
        data = {
            "energy_current_hour": 0.05,
            "energy_previous_hour": 0.04,
            "energy_two_hours_ago": 0.06,
            "target_hour": 14.0,
            "target_day_of_week": 2.0,
            "target_is_weekend": 0.0,
        }
        result = prepare_feature_vector(data)
        assert result.shape == (1, 6)

    def test_column_order(self):
        data = {
            "target_is_weekend": 0.0,
            "energy_current_hour": 0.05,
            "target_hour": 14.0,
            "energy_previous_hour": 0.04,
            "target_day_of_week": 2.0,
            "energy_two_hours_ago": 0.06,
        }
        result = prepare_feature_vector(data)
        assert list(result.columns) == FEATURE_COLUMNS
