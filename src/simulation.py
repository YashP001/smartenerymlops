"""
Production Data Simulator for Smart Home Energy Advisor (Phase 3.2).

Generates synthetic, production-like smart-meter observations for three distinct scenarios:
1. Normal Operation (baseline operational distribution, no drift)
2. HVAC Consumption Shift (covariate shift: afternoon/evening cooling load spike)
3. Sensor Drift (measurement noise, systematic calibration bias, and disturbances)

All generated data is explicitly tagged as simulated and isolated under data/simulated/.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FEATURE_COLUMNS, TARGET_COLUMN

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REF_PATH = os.path.join(_PROJECT_ROOT, "data", "train.csv")
DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "data", "simulated")

VALID_SCENARIOS = ("normal", "hvac_shift", "sensor_drift")
SIMULATOR_VERSION = "1.0.0"


def load_reference_data(ref_path: Optional[str] = None) -> pd.DataFrame:
    """Load and validate baseline reference dataset from training data.

    Args:
        ref_path: Optional path to reference CSV. Defaults to data/train.csv.

    Returns:
        DataFrame containing reference smart-meter observations.
    """
    path = ref_path or DEFAULT_REF_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference dataset not found at: {path}")

    df = pd.read_csv(path)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Reference dataset missing required columns: {missing}")

    return df


def generate_simulated_data(
    scenario: str = "normal",
    n_samples: int = 500,
    random_state: int = 42,
    severity: float = 1.0,
    start_date: str = "2021-03-01 00:00:00",
    meter_id: str = "MH01",
    ref_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Generate reproducible synthetic smart-meter observations for a specific scenario.

    Args:
        scenario: 'normal', 'hvac_shift', or 'sensor_drift'.
        n_samples: Number of hourly observations to generate.
        random_state: Seed for reproducibility.
        severity: Magnitude multiplier for drift scenarios (default 1.0).
        start_date: Beginning timestamp for simulated timeline.
        meter_id: Identifier for simulated meter.
        ref_df: Optional preloaded reference DataFrame.

    Returns:
        DataFrame with canonical feature schema, ground truth, and simulation metadata.
    """
    if scenario not in VALID_SCENARIOS:
        raise ValueError(
            f"Invalid scenario '{scenario}'. Must be one of {VALID_SCENARIOS}"
        )
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if severity < 0:
        raise ValueError(f"severity must be non-negative, got {severity}")

    reference = ref_df if ref_df is not None else load_reference_data()
    rng = np.random.default_rng(random_state)

    # 1. Generate chronological timestamp series
    base_ts = pd.to_datetime(start_date)
    timestamps = [base_ts + timedelta(hours=i) for i in range(n_samples)]
    target_hours = np.array([ts.hour for ts in timestamps])
    target_days = np.array([ts.dayofweek for ts in timestamps])
    target_weekends = np.array([1 if d >= 5 else 0 for d in target_days])

    # 2. Extract baseline conditional distributions from reference data
    # Standard deviation of energy in reference dataset for perturbation scaling
    ref_energy_std = float(reference["energy_current_hour"].std())
    gen_time_utc = datetime.now(timezone.utc).isoformat()

    # Pre-index reference records by (hour, is_weekend) for fast conditioned sampling
    groups = {}
    for (h, w), grp in reference.groupby(["target_hour", "target_is_weekend"]):
        energy_cols = ["energy_two_hours_ago", "energy_previous_hour", "energy_current_hour", TARGET_COLUMN]
        groups[(h, w)] = grp[energy_cols].to_numpy()

    # Fallback to entire dataset array if a specific hour/weekend group is empty
    all_energy = reference[["energy_two_hours_ago", "energy_previous_hour", "energy_current_hour", TARGET_COLUMN]].to_numpy()

    sampled_energies = np.zeros((n_samples, 4), dtype=np.float64)
    for i in range(n_samples):
        h, w = target_hours[i], target_weekends[i]
        pool = groups.get((h, w), all_energy)
        idx = rng.integers(0, len(pool))
        sampled_energies[i] = pool[idx]

    e_lag2 = sampled_energies[:, 0].copy()
    e_lag1 = sampled_energies[:, 1].copy()
    e_curr = sampled_energies[:, 2].copy()
    e_next = sampled_energies[:, 3].copy()

    # 3. Apply Scenario-Specific Transformations
    if scenario == "normal":
        # Realistic stochastic jitter preserving joint distribution without drift
        noise_scale = 0.05 * ref_energy_std
        e_lag2 += rng.normal(0, noise_scale, n_samples)
        e_lag1 += rng.normal(0, noise_scale, n_samples)
        e_curr += rng.normal(0, noise_scale, n_samples)
        e_next += rng.normal(0, noise_scale, n_samples)

    elif scenario == "hvac_shift":
        # Covariate shift: Elevated afternoon/evening air conditioning cooling demand
        # Peak HVAC load occurs between 12:00 and 20:00 (hours 12 to 20)
        hvac_peak_mask = (target_hours >= 12) & (target_hours <= 20)
        
        # Base HVAC consumption boost scaled by hour profile and severity
        # Hour shape: triangular/bell peak around 15:00-16:00
        hour_factor = np.sin(np.pi * np.clip(target_hours - 12, 0, 8) / 8.0)
        base_hvac_add = severity * (0.12 + 0.15 * hour_factor)
        mult_factor = 1.0 + (0.45 * severity)

        # Apply shift to active peak hours
        e_curr[hvac_peak_mask] = (e_curr[hvac_peak_mask] * mult_factor) + base_hvac_add[hvac_peak_mask]
        e_next[hvac_peak_mask] = (e_next[hvac_peak_mask] * mult_factor) + base_hvac_add[hvac_peak_mask]

        # Carry forward into lags so sequential features maintain natural lag consistency
        e_lag1[hvac_peak_mask] = (e_lag1[hvac_peak_mask] * mult_factor) + (base_hvac_add[hvac_peak_mask] * 0.8)
        e_lag2[hvac_peak_mask] = (e_lag2[hvac_peak_mask] * mult_factor) + (base_hvac_add[hvac_peak_mask] * 0.6)

        # Off-peak hours experience subtle residual thermal load
        off_peak_mask = ~hvac_peak_mask
        e_curr[off_peak_mask] += severity * 0.02
        e_next[off_peak_mask] += severity * 0.02

    elif scenario == "sensor_drift":
        # Measurement sensor degradation: systematic calibration bias + elevated measurement noise
        calibration_bias = severity * 0.05
        sensor_noise_std = severity * (0.10 * ref_energy_std)

        # Systematic additive calibration drift
        e_curr += calibration_bias + rng.normal(0, sensor_noise_std, n_samples)
        e_next += calibration_bias + rng.normal(0, sensor_noise_std, n_samples)
        e_lag1 += calibration_bias + rng.normal(0, sensor_noise_std, n_samples)
        e_lag2 += calibration_bias + rng.normal(0, sensor_noise_std, n_samples)

        # Intermittent measurement disturbances/spikes (3% probability)
        spike_mask = rng.random(n_samples) < 0.03
        e_curr[spike_mask] += severity * rng.uniform(0.15, 0.35, size=np.sum(spike_mask))

    # 4. Enforce physical non-negativity and round to realistic decimal precision
    e_lag2 = np.round(np.clip(e_lag2, 0.0, None), 4)
    e_lag1 = np.round(np.clip(e_lag1, 0.0, None), 4)
    e_curr = np.round(np.clip(e_curr, 0.0, None), 4)
    e_next = np.round(np.clip(e_next, 0.0, None), 4)

    # 5. Assemble final structured DataFrame with explicit metadata
    df_sim = pd.DataFrame({
        "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
        "meter": [meter_id] * n_samples,
        "energy_current_hour": e_curr,
        "energy_previous_hour": e_lag1,
        "energy_two_hours_ago": e_lag2,
        "target_hour": target_hours,
        "target_day_of_week": target_days,
        "target_is_weekend": target_weekends,
        TARGET_COLUMN: e_next,
        "simulation_scenario": [scenario] * n_samples,
        "is_simulated": [True] * n_samples,
        "generation_timestamp": [gen_time_utc] * n_samples,
    })

    return df_sim


def validate_simulated_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate that simulated data adheres to schema, physical, and timestamp rules.

    Args:
        df: Simulated DataFrame to validate.

    Returns:
        Validation report dictionary with passed flag and detailed checks.
    """
    errors = []
    required_cols = (
        ["timestamp", "meter"]
        + FEATURE_COLUMNS
        + [TARGET_COLUMN, "simulation_scenario", "is_simulated", "generation_timestamp"]
    )

    # 1. Column existence
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # 2. Finite numeric check
    numeric_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' is not numeric dtype.")
            else:
                if df[col].isna().any():
                    errors.append(f"Column '{col}' contains NaN values.")
                if np.isinf(df[col]).any():
                    errors.append(f"Column '{col}' contains infinite values.")
                if (df[col] < 0).any():
                    errors.append(f"Column '{col}' contains negative energy values.")

    # 3. Categorical / metadata integrity
    if "is_simulated" in df.columns:
        if not bool(df["is_simulated"].all()):
            errors.append("Column 'is_simulated' must be True for all rows.")

    if "simulation_scenario" in df.columns:
        scenarios = df["simulation_scenario"].unique()
        invalid_sc = [s for s in scenarios if s not in VALID_SCENARIOS]
        if invalid_sc:
            errors.append(f"Invalid scenarios found: {invalid_sc}")

    # 4. Timestamp validity
    if "timestamp" in df.columns:
        parsed_ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if parsed_ts.isna().any():
            errors.append("Timestamp column contains unparseable values.")

    passed = len(errors) == 0
    return {
        "passed": passed,
        "row_count": len(df),
        "errors": errors,
    }


def save_simulated_dataset(
    df: pd.DataFrame,
    scenario: str,
    output_dir: Optional[str] = None,
    random_state: int = 42,
    severity: float = 1.0,
    reference_dataset: Optional[str] = None,
) -> Tuple[str, str]:
    """Save simulated dataset CSV and accompanying metadata JSON.

    Args:
        df: DataFrame with simulated observations.
        scenario: Scenario name.
        output_dir: Target directory. Defaults to data/simulated/.
        random_state: Seed used for generation.
        severity: Severity used for generation.
        reference_dataset: Identifier or path of baseline reference dataset.

    Returns:
        Tuple of (csv_path, json_path).
    """
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    csv_filename = f"simulated_{scenario}.csv"
    json_filename = f"simulated_{scenario}_metadata.json"

    csv_path = os.path.join(out_dir, csv_filename)
    json_path = os.path.join(out_dir, json_filename)

    df.to_csv(csv_path, index=False)

    metadata = {
        "simulator_version": SIMULATOR_VERSION,
        "reference_dataset": reference_dataset or "data/train.csv",
        "simulation_scenario": scenario,
        "n_samples": len(df),
        "random_state": random_state,
        "severity": severity,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "columns": df.columns.tolist(),
        "summary_statistics": {
            "energy_current_hour_mean": float(df["energy_current_hour"].mean()),
            "energy_current_hour_std": float(df["energy_current_hour"].std()),
            "energy_current_hour_min": float(df["energy_current_hour"].min()),
            "energy_current_hour_max": float(df["energy_current_hour"].max()),
            "target_next_hour_mean": float(df[TARGET_COLUMN].mean()),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return csv_path, json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate simulated production smart-meter data.")
    parser.add_argument(
        "--scenario",
        choices=["all"] + list(VALID_SCENARIOS),
        default="all",
        help="Simulation scenario to execute (default: all)",
    )
    parser.add_argument("--n-samples", type=int, default=500, help="Number of observations (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--severity", type=float, default=1.0, help="Drift severity multiplier (default: 1.0)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Target output directory")

    args = parser.parse_args()
    scenarios_to_run = list(VALID_SCENARIOS) if args.scenario == "all" else [args.scenario]

    print(f"Generating simulated production data (n_samples={args.n_samples}, seed={args.seed}, severity={args.severity})...")
    ref = load_reference_data()

    for sc in scenarios_to_run:
        sim_df = generate_simulated_data(
            scenario=sc,
            n_samples=args.n_samples,
            random_state=args.seed,
            severity=args.severity,
            ref_df=ref,
        )
        val = validate_simulated_dataset(sim_df)
        if not val["passed"]:
            print(f"Validation FAILED for scenario '{sc}': {val['errors']}")
            sys.exit(1)

        c_path, j_path = save_simulated_dataset(
            sim_df,
            scenario=sc,
            output_dir=args.output_dir,
            random_state=args.seed,
            severity=args.severity,
        )
        print(f"  [{sc.upper()}] Saved {len(sim_df)} rows to: {c_path}")
        print(f"         Metadata saved to: {j_path}")
