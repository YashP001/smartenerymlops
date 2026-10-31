"""
Unit and integration tests for Production Data Simulator (Phase 3.2).

Tests scenario generation, schema conformity, seed reproducibility,
physical validity, and statistical distribution shifts (KS-test).
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import pytest
from scipy import stats

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.simulation import (
    load_reference_data,
    generate_simulated_data,
    validate_simulated_dataset,
    save_simulated_dataset,
    VALID_SCENARIOS,
)


@pytest.fixture(scope="module")
def reference_data():
    """Load reference baseline training data once for testing."""
    return load_reference_data()


# ─── 1. Core Generator & Scenario Tests ──────────────────────────────

class TestSimulationScenarios:
    """Tests covering scenario execution and schema verification."""

    @pytest.mark.parametrize("scenario", VALID_SCENARIOS)
    def test_all_scenarios_generate_valid_dataframe(self, reference_data, scenario):
        """Verifies each scenario produces a valid DataFrame of requested size."""
        df = generate_simulated_data(
            scenario=scenario,
            n_samples=60,
            random_state=42,
            ref_df=reference_data,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 60
        assert df["simulation_scenario"].iloc[0] == scenario

    def test_invalid_scenario_raises_value_error(self, reference_data):
        """Requesting an invalid scenario name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scenario"):
            generate_simulated_data(scenario="unknown_drift", ref_df=reference_data)

    def test_invalid_samples_or_severity_raises(self, reference_data):
        """Invalid sample size or negative severity raises ValueError."""
        with pytest.raises(ValueError, match="n_samples must be positive"):
            generate_simulated_data(n_samples=0, ref_df=reference_data)

        with pytest.raises(ValueError, match="severity must be non-negative"):
            generate_simulated_data(severity=-0.5, ref_df=reference_data)

    def test_exact_feature_schema(self, reference_data):
        """Generated dataset contains all canonical model features, target, and metadata."""
        df = generate_simulated_data(scenario="normal", n_samples=50, ref_df=reference_data)

        # Canonical model input features
        for col in FEATURE_COLUMNS:
            assert col in df.columns
            assert pd.api.types.is_numeric_dtype(df[col])

        # Target and operational identifiers
        assert TARGET_COLUMN in df.columns
        assert "timestamp" in df.columns
        assert "meter" in df.columns
        assert "simulation_scenario" in df.columns
        assert "is_simulated" in df.columns
        assert "generation_timestamp" in df.columns

    def test_metadata_flags(self, reference_data):
        """Confirms is_simulated is explicitly True and scenario tag is preserved."""
        df = generate_simulated_data(scenario="hvac_shift", n_samples=30, ref_df=reference_data)
        assert bool(df["is_simulated"].all()) is True
        assert bool((df["simulation_scenario"] == "hvac_shift").all()) is True


# ─── 2. Reproducibility & Sizing Tests ────────────────────────────────

class TestReproducibilityAndSizing:
    """Tests seed determinism and dataset sizing."""

    def test_identical_seed_produces_identical_output(self, reference_data):
        """Same random_state produces identical numeric output."""
        df1 = generate_simulated_data(scenario="normal", n_samples=50, random_state=123, ref_df=reference_data)
        df2 = generate_simulated_data(scenario="normal", n_samples=50, random_state=123, ref_df=reference_data)

        pd.testing.assert_frame_equal(
            df1[FEATURE_COLUMNS + [TARGET_COLUMN]],
            df2[FEATURE_COLUMNS + [TARGET_COLUMN]],
        )

    def test_different_seeds_produce_different_output(self, reference_data):
        """Different random_state values produce distinct values."""
        df1 = generate_simulated_data(scenario="normal", n_samples=50, random_state=42, ref_df=reference_data)
        df2 = generate_simulated_data(scenario="normal", n_samples=50, random_state=999, ref_df=reference_data)

        diff = (df1["energy_current_hour"] - df2["energy_current_hour"]).abs().sum()
        assert diff > 0.01

    @pytest.mark.parametrize("size", [24, 72, 168])
    def test_configurable_dataset_size(self, reference_data, size):
        """Generates exactly the requested number of observations."""
        df = generate_simulated_data(scenario="normal", n_samples=size, ref_df=reference_data)
        assert len(df) == size


# ─── 3. Data Integrity & Physical Bounds Tests ────────────────────────

class TestDataIntegrity:
    """Tests physical validity, finite values, and timestamp continuity."""

    @pytest.mark.parametrize("scenario", VALID_SCENARIOS)
    def test_finite_non_negative_values(self, reference_data, scenario):
        """All energy features are non-negative, non-NaN, and finite."""
        df = generate_simulated_data(scenario=scenario, n_samples=100, ref_df=reference_data)
        val = validate_simulated_dataset(df)
        assert val["passed"] is True
        assert len(val["errors"]) == 0

        for col in ["energy_current_hour", "energy_previous_hour", "energy_two_hours_ago", TARGET_COLUMN]:
            assert (df[col] >= 0.0).all()
            assert not df[col].isna().any()
            assert np.isfinite(df[col]).all()

    def test_chronological_timestamp_progression(self, reference_data):
        """Simulated timestamps progress hourly without gaps."""
        df = generate_simulated_data(scenario="normal", n_samples=48, start_date="2021-04-01 00:00:00", ref_df=reference_data)
        ts = pd.to_datetime(df["timestamp"])
        diffs = ts.diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all()


# ─── 4. Statistical Shift Detection Tests ─────────────────────────────

class TestDistributionShifts:
    """Verifies that normal operations preserve distribution while shifts are detectable."""

    def test_normal_scenario_close_to_reference(self, reference_data):
        """Normal scenario remains statistically consistent with reference distribution.

        Distribution similarity is assessed jointly via effect size (Cohen's d) and
        KS distance statistic, rather than relying exclusively on a p-value > 0.05.
        """
        df_normal = generate_simulated_data(
            scenario="normal",
            n_samples=500,
            random_state=42,
            ref_df=reference_data,
        )

        ref_energy = reference_data["energy_current_hour"].to_numpy()
        sim_energy = df_normal["energy_current_hour"].to_numpy()

        # 1. Mean difference should be small
        mean_diff = sim_energy.mean() - ref_energy.mean()
        assert abs(mean_diff) < 0.03

        # 2. Effect size (Cohen's d) must indicate negligible difference (|d| < 0.15)
        pooled_std = np.sqrt((ref_energy.var() + sim_energy.var()) / 2)
        cohens_d = abs(mean_diff / pooled_std) if pooled_std > 0 else 0.0
        assert cohens_d < 0.15

        # 3. KS distance statistic must remain low (< 0.15)
        ks_stat, _ = stats.ks_2samp(ref_energy, sim_energy)
        assert ks_stat < 0.15

    def test_hvac_shift_produces_detectable_drift(self, reference_data):
        """HVAC shift significantly elevates afternoon/evening energy consumption."""
        df_hvac = generate_simulated_data(
            scenario="hvac_shift",
            n_samples=500,
            severity=1.2,
            random_state=42,
            ref_df=reference_data,
        )

        ref_energy = reference_data["energy_current_hour"].to_numpy()
        hvac_energy = df_hvac["energy_current_hour"].to_numpy()

        # Mean energy must be substantially higher under HVAC cooling surge
        mean_diff = hvac_energy.mean() - ref_energy.mean()
        assert mean_diff > 0.04

        # Cohen's d confirms medium-to-large shift effect size (d > 0.30)
        pooled_std = np.sqrt((ref_energy.var() + hvac_energy.var()) / 2)
        cohens_d = mean_diff / pooled_std
        assert cohens_d > 0.30

        # Two-sample Kolmogorov-Smirnov test must detect significant distribution drift
        ks_stat, p_val = stats.ks_2samp(ref_energy, hvac_energy)
        assert p_val < 0.001
        assert ks_stat > 0.20

    def test_sensor_drift_produces_detectable_bias(self, reference_data):
        """Sensor drift introduces systematic calibration shift and noise detectable by KS test."""
        df_sensor = generate_simulated_data(
            scenario="sensor_drift",
            n_samples=500,
            severity=1.5,
            random_state=42,
            ref_df=reference_data,
        )

        ref_energy = reference_data["energy_current_hour"].to_numpy()
        sensor_energy = df_sensor["energy_current_hour"].to_numpy()

        # Positive calibration bias increases mean
        mean_diff = sensor_energy.mean() - ref_energy.mean()
        assert mean_diff > 0.03

        # Cohen's d confirms substantial calibration bias effect size (d > 0.25)
        pooled_std = np.sqrt((ref_energy.var() + sensor_energy.var()) / 2)
        cohens_d = mean_diff / pooled_std
        assert cohens_d > 0.25

        # KS test detects deviation from reference distribution
        ks_stat, p_val = stats.ks_2samp(ref_energy, sensor_energy)
        assert p_val < 0.01
        assert ks_stat > 0.20

    @pytest.mark.parametrize("seed", [1, 42, 101])
    def test_drift_scenarios_distinguishable_from_normal(self, reference_data, seed):
        """Drift scenarios remain demonstrably distinguishable from normal operation across seeds."""
        df_norm = generate_simulated_data("normal", 500, random_state=seed, ref_df=reference_data)
        df_hvac = generate_simulated_data("hvac_shift", 500, random_state=seed, severity=1.0, ref_df=reference_data)
        df_sens = generate_simulated_data("sensor_drift", 500, random_state=seed, severity=1.2, ref_df=reference_data)

        ref_energy = reference_data["energy_current_hour"].to_numpy()
        norm_energy = df_norm["energy_current_hour"].to_numpy()
        hvac_energy = df_hvac["energy_current_hour"].to_numpy()
        sens_energy = df_sens["energy_current_hour"].to_numpy()

        # Effect sizes vs reference
        def calc_d(x, y):
            return (x.mean() - y.mean()) / np.sqrt((x.var() + y.var()) / 2)

        d_norm = abs(calc_d(norm_energy, ref_energy))
        d_hvac = calc_d(hvac_energy, ref_energy)
        d_sens = calc_d(sens_energy, ref_energy)

        # Normal effect size is negligible; drift effect sizes are distinctly higher
        assert d_norm < 0.15
        assert d_hvac > 0.30
        assert d_sens > 0.25


# ─── 5. File Persistence & Metadata Tests ─────────────────────────────

class TestPersistenceAndIsolation:
    """Tests saving simulation outputs and accompanying metadata."""

    def test_save_simulated_dataset(self, reference_data, tmp_path):
        """Saves scenario CSV and metadata JSON with full provenance to target directory."""
        df = generate_simulated_data(scenario="hvac_shift", n_samples=40, ref_df=reference_data)
        csv_path, json_path = save_simulated_dataset(
            df,
            scenario="hvac_shift",
            output_dir=str(tmp_path),
            random_state=42,
            severity=1.0,
            reference_dataset="data/train.csv",
        )

        assert os.path.exists(csv_path)
        assert os.path.exists(json_path)

        # Verify CSV content
        saved_df = pd.read_csv(csv_path)
        assert len(saved_df) == 40
        assert saved_df["simulation_scenario"].iloc[0] == "hvac_shift"

        # Verify JSON metadata content and provenance fields
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["simulator_version"] == "1.0.0"
        assert meta["reference_dataset"] == "data/train.csv"
        assert meta["simulation_scenario"] == "hvac_shift"
        assert meta["n_samples"] == 40
        assert meta["random_state"] == 42
        assert meta["severity"] == 1.0
        assert "generation_timestamp" in meta
        assert "summary_statistics" in meta
        assert "energy_current_hour_mean" in meta["summary_statistics"]
