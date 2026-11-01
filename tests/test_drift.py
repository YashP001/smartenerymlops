"""
Unit and integration tests for Data Drift Detection Engine (Phase 3.3).

Covers input validation, feature-level statistical tests (KS and Cohen's d),
dataset-level aggregation, target drift separation, constant column handling,
small sample sizes, scenario-specific drift detection, and report generation.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.monitoring.drift_detector import (
    load_dataset,
    validate_monitoring_data,
    calculate_feature_drift,
    calculate_dataset_drift,
    generate_evidently_reports,
    run_drift_analysis,
)
from src.simulation import generate_simulated_data, load_reference_data


@pytest.fixture(scope="module")
def reference_data():
    """Load baseline historical training dataset."""
    return load_reference_data()


@pytest.fixture
def sample_feature_df():
    """Generate a clean synthetic DataFrame with canonical features and target."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "energy_current_hour": rng.uniform(0.1, 0.5, size=n),
        "energy_previous_hour": rng.uniform(0.1, 0.5, size=n),
        "energy_two_hours_ago": rng.uniform(0.1, 0.5, size=n),
        "target_hour": rng.integers(0, 24, size=n),
        "target_day_of_week": rng.integers(0, 7, size=n),
        "target_is_weekend": rng.choice([0, 1], size=n),
        TARGET_COLUMN: rng.uniform(0.1, 0.5, size=n),
    })


# ─── 1. Input Validation Tests ────────────────────────────────────────

class TestInputValidation:
    """Tests input data validation and schema verification."""

    def test_valid_inputs_pass(self, sample_feature_df):
        """Valid DataFrame passes validation without error."""
        validate_monitoring_data(sample_feature_df, features=FEATURE_COLUMNS, target_column=TARGET_COLUMN)

    def test_load_dataset_from_dataframe_and_path(self, sample_feature_df, tmp_path):
        """load_dataset correctly handles both DataFrame instances and CSV paths."""
        df_copy = load_dataset(sample_feature_df)
        assert len(df_copy) == len(sample_feature_df)

        csv_path = tmp_path / "test_data.csv"
        sample_feature_df.to_csv(csv_path, index=False)
        df_loaded = load_dataset(str(csv_path))
        assert len(df_loaded) == len(sample_feature_df)

    def test_load_nonexistent_path_raises_file_not_found(self):
        """Attempting to load a nonexistent CSV file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent_path_12345.csv")

    def test_missing_required_column_raises(self, sample_feature_df):
        """Missing a required canonical feature raises ValueError."""
        incomplete_df = sample_feature_df.drop(columns=["energy_current_hour"])
        with pytest.raises(ValueError, match="missing required feature columns"):
            validate_monitoring_data(incomplete_df, features=FEATURE_COLUMNS)

    def test_empty_dataframe_raises(self, sample_feature_df):
        """A 0-row DataFrame raises ValueError."""
        empty_df = sample_feature_df.iloc[0:0]
        with pytest.raises(ValueError, match="dataset has 0 rows"):
            validate_monitoring_data(empty_df, features=FEATURE_COLUMNS)

    def test_non_numeric_column_raises(self, sample_feature_df):
        """Non-numeric feature values raise ValueError."""
        corrupted_df = sample_feature_df.copy()
        corrupted_df["energy_current_hour"] = "not_numeric"
        with pytest.raises(ValueError, match="not numeric dtype"):
            validate_monitoring_data(corrupted_df, features=FEATURE_COLUMNS)

    def test_nan_values_raise(self, sample_feature_df):
        """Presence of NaN values raises ValueError."""
        nan_df = sample_feature_df.copy()
        nan_df.loc[0, "energy_current_hour"] = np.nan
        with pytest.raises(ValueError, match="contains NaN values"):
            validate_monitoring_data(nan_df, features=FEATURE_COLUMNS)

    def test_infinite_values_raise(self, sample_feature_df):
        """Presence of infinite values raises ValueError."""
        inf_df = sample_feature_df.copy()
        inf_df.loc[0, "energy_current_hour"] = np.inf
        with pytest.raises(ValueError, match="contains infinite values"):
            validate_monitoring_data(inf_df, features=FEATURE_COLUMNS)


# ─── 2. Feature-Level Drift Detection Tests ───────────────────────────

class TestFeatureLevelDrift:
    """Tests feature-level statistical tests (KS test, Cohen's d, edge cases)."""

    def test_identical_distributions_no_drift(self):
        """Identical distributions produce low KS distance and no drift."""
        rng = np.random.default_rng(42)
        s1 = pd.Series(rng.normal(0, 1, 500))
        s2 = pd.Series(rng.normal(0, 1, 500))

        res = calculate_feature_drift(s1, s2, "feature_x")
        assert res["drift_detected"] is False
        assert abs(res["cohens_d"]) < 0.20
        assert res["ks_statistic"] < 0.15

    def test_shifted_distribution_detected(self):
        """Distribution shifted by substantial mean difference is flagged as drifted."""
        rng = np.random.default_rng(42)
        s1 = pd.Series(rng.normal(0.0, 1.0, 500))
        s2 = pd.Series(rng.normal(1.0, 1.0, 500))  # 1 standard deviation shift

        res = calculate_feature_drift(s1, s2, "feature_x")
        assert res["drift_detected"] is True
        assert res["cohens_d"] > 0.80
        assert res["p_value"] < 0.001
        assert res["ks_statistic"] > 0.20

    def test_constant_columns_both_equal(self):
        """Constant columns with identical value are handled safely and do not drift."""
        s1 = pd.Series([5.0] * 50)
        s2 = pd.Series([5.0] * 50)

        res = calculate_feature_drift(s1, s2, "const_feat")
        assert res["drift_detected"] is False
        assert res["drift_reason"] == "both_constant_equal"
        assert res["ks_statistic"] == 0.0
        assert res["p_value"] == 1.0

    def test_constant_columns_different(self):
        """Constant columns with different values flag drift safely."""
        s1 = pd.Series([5.0] * 50)
        s2 = pd.Series([10.0] * 50)

        res = calculate_feature_drift(s1, s2, "const_feat")
        assert res["drift_detected"] is True
        assert res["drift_reason"] == "constant_value_changed"
        assert res["ks_statistic"] == 1.0

    def test_one_constant_one_varying(self):
        """Variance loss (one series constant, other varying) flags drift safely."""
        s1 = pd.Series([5.0] * 50)
        s2 = pd.Series(np.linspace(1, 10, 50))

        res = calculate_feature_drift(s1, s2, "variance_loss_feat")
        assert res["drift_detected"] is True
        assert res["drift_reason"] == "variance_loss_or_gain"

    def test_small_sample_size_safe_execution(self):
        """Small sample sizes (e.g. n=5) execute safely without raising errors."""
        s1 = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        s2 = pd.Series([0.12, 0.18, 0.31, 0.42, 0.49])

        res = calculate_feature_drift(s1, s2, "small_feat")
        assert isinstance(res, dict)
        assert "drift_detected" in res
        assert "cohens_d" in res
        assert "ks_statistic" in res


# ─── 3. Dataset-Level Drift & Target Separation Tests ──────────────────

class TestDatasetLevelDrift:
    """Tests aggregation of feature drift into dataset drift and target tracking."""

    def test_dataset_drift_summary_structure(self, sample_feature_df):
        """Summary contains all required dataset-level metrics and feature details."""
        prod_df = sample_feature_df.copy()
        prod_df["energy_current_hour"] += 1.0

        res = calculate_dataset_drift(sample_feature_df, prod_df)
        assert "dataset_drift_detected" in res
        assert "drift_share" in res
        assert "total_features_count" in res
        assert res["total_features_count"] == len(FEATURE_COLUMNS)
        assert "drifted_features_count" in res
        assert "drifted_features" in res
        assert "feature_metrics" in res
        assert "target_drift" in res
        assert "thresholds" in res

    def test_configurable_thresholds(self, sample_feature_df):
        """Configurable dataset drift share threshold alters dataset drift flag."""
        prod_df = sample_feature_df.copy()
        prod_df["energy_current_hour"] += 1.0  # 1 of 6 features drifted (16.7%)

        # With default 50% threshold: not drifted
        res_default = calculate_dataset_drift(sample_feature_df, prod_df, dataset_drift_share_threshold=0.50)
        assert res_default["dataset_drift_detected"] is False

        # With sensitive 15% threshold: drifted
        res_strict = calculate_dataset_drift(sample_feature_df, prod_df, dataset_drift_share_threshold=0.15)
        assert res_strict["dataset_drift_detected"] is True

    def test_target_drift_distinguished_from_input_drift(self, sample_feature_df):
        """Target drift is tracked separately and does not inflate input feature drift count."""
        prod_df = sample_feature_df.copy()
        prod_df[TARGET_COLUMN] += 2.0  # Shift only the target, leave inputs unchanged

        res = calculate_dataset_drift(sample_feature_df, prod_df)
        # Input features remain unchanged
        assert res["drifted_features_count"] == 0
        assert res["dataset_drift_detected"] is False

        # Target drift is detected independently
        assert res["target_drift"] is not None
        assert res["target_drift"]["target_drift_detected"] is True
        assert res["target_drift"]["statistics"]["cohens_d"] > 1.0


# ─── 4. Scenario Validation with Simulator ────────────────────────────

class TestScenarioValidation:
    """Validates drift detector behavior across normal, hvac_shift, and sensor_drift scenarios."""

    def test_normal_scenario_shows_no_dataset_drift(self, reference_data):
        """Normal scenario operational data shows low distribution shift and no dataset drift."""
        df_normal = generate_simulated_data("normal", n_samples=500, random_state=42, ref_df=reference_data)

        res = calculate_dataset_drift(reference_data, df_normal)
        assert res["dataset_drift_detected"] is False
        assert res["drift_share"] < 0.50

    def test_hvac_scenario_detects_energy_drift(self, reference_data):
        """HVAC scenario specifically flags energy features and target while calendar features remain stable."""
        df_hvac = generate_simulated_data("hvac_shift", n_samples=500, severity=1.2, random_state=42, ref_df=reference_data)

        res = calculate_dataset_drift(reference_data, df_hvac)
        assert res["dataset_drift_detected"] is True
        assert "energy_current_hour" in res["drifted_features"]
        assert "energy_previous_hour" in res["drifted_features"]
        assert "energy_two_hours_ago" in res["drifted_features"]

        # Calendar features should remain stable
        assert "target_hour" not in res["drifted_features"]
        assert "target_day_of_week" not in res["drifted_features"]
        assert "target_is_weekend" not in res["drifted_features"]

        # Target next hour is also drifted
        assert res["target_drift"]["target_drift_detected"] is True

    def test_sensor_scenario_detects_energy_drift(self, reference_data):
        """Sensor drift scenario specifically flags energy features due to calibration bias."""
        df_sensor = generate_simulated_data("sensor_drift", n_samples=500, severity=1.5, random_state=42, ref_df=reference_data)

        res = calculate_dataset_drift(reference_data, df_sensor)
        assert res["dataset_drift_detected"] is True
        assert "energy_current_hour" in res["drifted_features"]
        assert "energy_previous_hour" in res["drifted_features"]
        assert "energy_two_hours_ago" in res["drifted_features"]


# ─── 5. Report Generation & Reproducibility Tests ─────────────────────

class TestReportGenerationAndReproducibility:
    """Tests Evidently HTML and JSON report persistence and reproducibility."""

    def test_evidently_html_and_json_report_generation(self, sample_feature_df, tmp_path):
        """generate_evidently_reports creates both HTML and JSON files with full metadata."""
        prod_df = sample_feature_df.copy()
        prod_df["energy_current_hour"] += 0.5

        out_dir = tmp_path / "reports"
        html_p, json_p = generate_evidently_reports(
            reference_df=sample_feature_df,
            production_df=prod_df,
            output_dir=str(out_dir),
            scenario="unit_test_scenario",
            reference_name="test_ref",
            production_name="test_prod",
            summary_dict={"test_key": "test_val"},
        )

        assert os.path.exists(html_p)
        assert os.path.exists(json_p)
        assert os.path.getsize(html_p) > 0

        # Validate JSON manifest structure
        with open(json_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["scenario"] == "unit_test_scenario"
        assert manifest["reference_dataset"] == "test_ref"
        assert manifest["production_dataset"] == "test_prod"
        assert "evidently_version" in manifest
        assert "generation_timestamp" in manifest
        assert "monitored_columns" in manifest
        assert manifest["summary"]["test_key"] == "test_val"

    def test_run_drift_analysis_end_to_end(self, sample_feature_df, tmp_path):
        """run_drift_analysis executes complete pipeline and returns expected schema."""
        out_dir = tmp_path / "reports_e2e"
        res = run_drift_analysis(
            reference_data=sample_feature_df,
            production_data=sample_feature_df,
            scenario="normal_e2e",
            output_dir=str(out_dir),
            generate_reports=True,
        )

        assert res["dataset_drift_detected"] is False
        assert os.path.exists(res["html_report_path"])
        assert os.path.exists(res["json_report_path"])

    def test_reproducibility_with_identical_inputs(self, sample_feature_df):
        """Running drift calculation on identical inputs produces identical metrics."""
        res1 = calculate_dataset_drift(sample_feature_df, sample_feature_df)
        res2 = calculate_dataset_drift(sample_feature_df, sample_feature_df)

        assert res1["drift_share"] == res2["drift_share"]
        assert res1["dataset_drift_detected"] == res2["dataset_drift_detected"]
        for feat in FEATURE_COLUMNS:
            assert res1["feature_metrics"][feat]["cohens_d"] == res2["feature_metrics"][feat]["cohens_d"]
            assert res1["feature_metrics"][feat]["ks_statistic"] == res2["feature_metrics"][feat]["ks_statistic"]
