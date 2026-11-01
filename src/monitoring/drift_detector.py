"""
Data Drift Detection Engine for Smart Home Energy Advisor (Phase 3.3).

Combines SciPy statistical tests (Kolmogorov-Smirnov two-sample test and Cohen's d
standardized effect size) with Evidently AI interactive reports to monitor
distributional shifts between historical reference data and production observations.

Monitors canonical model features (src/features.py) and tracks target_next_hour
as a distinct target drift indicator.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Ensure project root is in path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.features import FEATURE_COLUMNS, TARGET_COLUMN

DEFAULT_REF_PATH = os.path.join(_PROJECT_ROOT, "data", "train.csv")
DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "reports", "drift")
DEFAULT_KS_P_THRESHOLD = 0.05
DEFAULT_KS_STAT_THRESHOLD = 0.15
DEFAULT_COHENS_D_THRESHOLD = 0.20
DEFAULT_DATASET_DRIFT_SHARE = 0.50

# Evidently imports with graceful fallback if needed
try:
    import evidently
    from evidently import Report
    from evidently.presets import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
    EVIDENTLY_VERSION = getattr(evidently, "__version__", "0.7.23")
except ImportError:
    EVIDENTLY_AVAILABLE = False
    EVIDENTLY_VERSION = "unavailable"


def load_dataset(data_input: Union[pd.DataFrame, str, os.PathLike]) -> pd.DataFrame:
    """Load a DataFrame from a file path or return a copy of a DataFrame.

    Args:
        data_input: DataFrame or path to CSV file.

    Returns:
        pd.DataFrame containing the dataset.
    """
    if isinstance(data_input, pd.DataFrame):
        return data_input.copy()
    
    path = str(data_input)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at: {path}")
    
    df = pd.read_csv(path)
    return df


def validate_monitoring_data(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    dataset_name: str = "dataset",
) -> None:
    """Validate that monitored columns exist, have numeric dtypes, and contain finite values.

    Args:
        df: DataFrame to validate.
        features: Expected model feature names. Defaults to canonical FEATURE_COLUMNS.
        target_column: Optional target column to validate if present.
        dataset_name: Label used for descriptive error messages.

    Raises:
        ValueError: If columns are missing, non-numeric, or contain NaN/infinite values.
    """
    cols_to_check = list(features or FEATURE_COLUMNS)
    if target_column and target_column in df.columns:
        cols_to_check.append(target_column)

    # 1. Check for missing columns
    missing = [c for c in (features or FEATURE_COLUMNS) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Validation failed for {dataset_name}: missing required feature columns: {missing}"
        )

    if len(df) == 0:
        raise ValueError(f"Validation failed for {dataset_name}: dataset has 0 rows.")

    # 2. Check for numeric and finite values
    for col in cols_to_check:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"Validation failed for {dataset_name}: column '{col}' is not numeric dtype ({df[col].dtype})."
            )
        if df[col].isna().any():
            raise ValueError(
                f"Validation failed for {dataset_name}: column '{col}' contains NaN values."
            )
        if np.isinf(df[col]).any():
            raise ValueError(
                f"Validation failed for {dataset_name}: column '{col}' contains infinite values."
            )


def calculate_feature_drift(
    ref_series: pd.Series,
    prod_series: pd.Series,
    feature_name: str,
    ks_p_threshold: float = DEFAULT_KS_P_THRESHOLD,
    ks_stat_threshold: float = DEFAULT_KS_STAT_THRESHOLD,
    cohens_d_threshold: float = DEFAULT_COHENS_D_THRESHOLD,
) -> Dict[str, Any]:
    """Calculate feature-level distribution drift statistics.

    Evaluates both non-parametric distribution distance (Kolmogorov-Smirnov two-sample test)
    and standardized effect size (Cohen's d). A feature is flagged as drifting when:
    (p < ks_p_threshold and statistic >= ks_stat_threshold) OR (|cohens_d| >= cohens_d_threshold and p < 0.05).
    Handles constant and nearly-constant series safely.

    Args:
        ref_series: Reference observation values.
        prod_series: Production observation values.
        feature_name: Name of the feature being monitored.
        ks_p_threshold: Maximum p-value to reject null hypothesis.
        ks_stat_threshold: Minimum KS distance statistic required for drift.
        cohens_d_threshold: Minimum standardized mean difference effect size for drift.

    Returns:
        Dictionary containing comprehensive feature-level drift statistics.
    """
    ref_clean = ref_series.to_numpy(dtype=np.float64)
    prod_clean = prod_series.to_numpy(dtype=np.float64)

    ref_mean = float(np.mean(ref_clean))
    prod_mean = float(np.mean(prod_clean))
    ref_std = float(np.std(ref_clean, ddof=1)) if len(ref_clean) > 1 else 0.0
    prod_std = float(np.std(prod_clean, ddof=1)) if len(prod_clean) > 1 else 0.0
    mean_diff = prod_mean - ref_mean

    ref_median = float(np.median(ref_clean))
    prod_median = float(np.median(prod_clean))

    # Safe handling of constant columns (variance == 0)
    ref_is_const = (len(ref_clean) > 0 and np.all(ref_clean == ref_clean[0]))
    prod_is_const = (len(prod_clean) > 0 and np.all(prod_clean == prod_clean[0]))

    if ref_is_const and prod_is_const:
        if np.isclose(ref_clean[0], prod_clean[0]):
            return {
                "feature": feature_name,
                "drift_detected": False,
                "drift_reason": "both_constant_equal",
                "test_name": "constant_check",
                "ks_statistic": 0.0,
                "p_value": 1.0,
                "cohens_d": 0.0,
                "mean_difference": 0.0,
                "ref_mean": ref_mean,
                "prod_mean": prod_mean,
                "ref_std": 0.0,
                "prod_std": 0.0,
                "ref_median": ref_median,
                "prod_median": prod_median,
            }
        else:
            return {
                "feature": feature_name,
                "drift_detected": True,
                "drift_reason": "constant_value_changed",
                "test_name": "constant_check",
                "ks_statistic": 1.0,
                "p_value": 0.0,
                "cohens_d": float("inf") if mean_diff > 0 else float("-inf"),
                "mean_difference": mean_diff,
                "ref_mean": ref_mean,
                "prod_mean": prod_mean,
                "ref_std": 0.0,
                "prod_std": 0.0,
                "ref_median": ref_median,
                "prod_median": prod_median,
            }

    if ref_is_const or prod_is_const:
        return {
            "feature": feature_name,
            "drift_detected": True,
            "drift_reason": "variance_loss_or_gain",
            "test_name": "constant_check",
            "ks_statistic": 1.0,
            "p_value": 0.0,
            "cohens_d": float("inf") if mean_diff > 0 else float("-inf"),
            "mean_difference": mean_diff,
            "ref_mean": ref_mean,
            "prod_mean": prod_mean,
            "ref_std": ref_std,
            "prod_std": prod_std,
            "ref_median": ref_median,
            "prod_median": prod_median,
        }

    # Pooled standard deviation and Cohen's d
    pooled_var = (ref_std**2 + prod_std**2) / 2.0
    pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 0.0
    cohens_d = float(mean_diff / pooled_std) if pooled_std > 0 else 0.0

    # Kolmogorov-Smirnov 2-sample test
    ks_res = stats.ks_2samp(ref_clean, prod_clean)
    ks_stat = float(ks_res.statistic)
    p_val = float(ks_res.pvalue)

    # Multi-criteria drift detection:
    # Requires statistical significance AND non-trivial effect size or distribution distance.
    drift_detected = False
    reasons = []

    if p_val < ks_p_threshold and ks_stat >= ks_stat_threshold:
        drift_detected = True
        reasons.append(f"KS distance ({ks_stat:.4f} >= {ks_stat_threshold}) with p={p_val:.4e}")

    if abs(cohens_d) >= cohens_d_threshold and p_val < 0.05:
        drift_detected = True
        reasons.append(f"Cohen's d ({cohens_d:.4f} >= {cohens_d_threshold})")

    drift_reason = " & ".join(reasons) if drift_detected else "no_significant_shift"

    return {
        "feature": feature_name,
        "drift_detected": drift_detected,
        "drift_reason": drift_reason,
        "test_name": "kolmogorov_smirnov_and_cohens_d",
        "ks_statistic": round(ks_stat, 4),
        "p_value": round(p_val, 6),
        "cohens_d": round(cohens_d, 4),
        "mean_difference": round(mean_diff, 4),
        "ref_mean": round(ref_mean, 4),
        "prod_mean": round(prod_mean, 4),
        "ref_std": round(ref_std, 4),
        "prod_std": round(prod_std, 4),
        "ref_median": round(ref_median, 4),
        "prod_median": round(prod_median, 4),
    }


def calculate_dataset_drift(
    reference_data: Union[pd.DataFrame, str],
    production_data: Union[pd.DataFrame, str],
    features: Optional[List[str]] = None,
    target_column: Optional[str] = TARGET_COLUMN,
    ks_p_threshold: float = DEFAULT_KS_P_THRESHOLD,
    ks_stat_threshold: float = DEFAULT_KS_STAT_THRESHOLD,
    cohens_d_threshold: float = DEFAULT_COHENS_D_THRESHOLD,
    dataset_drift_share_threshold: float = DEFAULT_DATASET_DRIFT_SHARE,
) -> Dict[str, Any]:
    """Calculate dataset-level and feature-level drift summary.

    Distinguishes input feature drift (covariate shift) from target column drift (concept shift).

    Args:
        reference_data: Reference DataFrame or path.
        production_data: Production DataFrame or path.
        features: Monitored input features (defaults to canonical FEATURE_COLUMNS).
        target_column: Optional target column (defaults to target_next_hour).
        ks_p_threshold: P-value threshold for KS test.
        ks_stat_threshold: KS statistic threshold.
        cohens_d_threshold: Cohen's d threshold.
        dataset_drift_share_threshold: Proportion of drifted features required for dataset drift.

    Returns:
        Structured machine-readable summary dictionary.
    """
    ref_df = load_dataset(reference_data)
    prod_df = load_dataset(production_data)

    feature_cols = list(features or FEATURE_COLUMNS)

    # Validate inputs
    validate_monitoring_data(ref_df, features=feature_cols, target_column=target_column, dataset_name="reference")
    validate_monitoring_data(prod_df, features=feature_cols, target_column=target_column, dataset_name="production")

    feature_metrics: Dict[str, Any] = {}
    drifted_features: List[str] = []

    # 1. Evaluate input feature drift (covariate shift)
    for col in feature_cols:
        f_res = calculate_feature_drift(
            ref_series=ref_df[col],
            prod_series=prod_df[col],
            feature_name=col,
            ks_p_threshold=ks_p_threshold,
            ks_stat_threshold=ks_stat_threshold,
            cohens_d_threshold=cohens_d_threshold,
        )
        feature_metrics[col] = f_res
        if f_res["drift_detected"]:
            drifted_features.append(col)

    total_features = len(feature_cols)
    drifted_count = len(drifted_features)
    drift_share = float(drifted_count / total_features) if total_features > 0 else 0.0
    dataset_drift_detected = drift_share >= dataset_drift_share_threshold

    # 2. Evaluate target column drift (concept shift) separately if present
    target_drift_info: Optional[Dict[str, Any]] = None
    if target_column and target_column in ref_df.columns and target_column in prod_df.columns:
        t_res = calculate_feature_drift(
            ref_series=ref_df[target_column],
            prod_series=prod_df[target_column],
            feature_name=target_column,
            ks_p_threshold=ks_p_threshold,
            ks_stat_threshold=ks_stat_threshold,
            cohens_d_threshold=cohens_d_threshold,
        )
        target_drift_info = {
            "target_column": target_column,
            "target_drift_detected": t_res["drift_detected"],
            "statistics": t_res,
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reference_samples": len(ref_df),
        "production_samples": len(prod_df),
        "dataset_drift_detected": dataset_drift_detected,
        "drift_share": round(drift_share, 4),
        "drift_share_threshold": dataset_drift_share_threshold,
        "total_features_count": total_features,
        "drifted_features_count": drifted_count,
        "drifted_features": drifted_features,
        "feature_metrics": feature_metrics,
        "target_drift": target_drift_info,
        "thresholds": {
            "ks_p_value": ks_p_threshold,
            "ks_statistic": ks_stat_threshold,
            "cohens_d": cohens_d_threshold,
            "dataset_drift_share": dataset_drift_share_threshold,
        },
    }


def generate_evidently_reports(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    scenario: str = "production",
    features: Optional[List[str]] = None,
    target_column: Optional[str] = TARGET_COLUMN,
    reference_name: str = "data/train.csv",
    production_name: str = "simulated_production",
    summary_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Generate interactive Evidently HTML report and JSON summary file.

    Args:
        reference_df: Reference observations.
        production_df: Production observations.
        output_dir: Directory where reports should be saved.
        scenario: Name of scenario being analyzed.
        features: Monitored input feature columns.
        target_column: Optional target column to monitor.
        reference_name: Identifier for reference dataset.
        production_name: Identifier for production dataset.
        summary_dict: Precomputed machine-readable summary.

    Returns:
        Tuple of (html_path, json_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, f"drift_report_{scenario}.html")
    json_path = os.path.join(output_dir, f"drift_report_{scenario}_summary.json")

    feature_cols = list(features or FEATURE_COLUMNS)
    cols_to_monitor = feature_cols.copy()
    if target_column and target_column in reference_df.columns and target_column in production_df.columns:
        cols_to_monitor.append(target_column)

    # 1. Run Evidently Report
    if EVIDENTLY_AVAILABLE:
        report = Report(metrics=[DataDriftPreset(columns=cols_to_monitor)])
        snapshot = report.run(
            reference_data=reference_df[cols_to_monitor],
            current_data=production_df[cols_to_monitor],
        )
        snapshot.save_html(html_path)
    else:
        # Graceful fallback: basic HTML placeholder when Evidently UI unavailable
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(f"<html><body><h1>Evidently report unavailable for scenario {scenario}</h1></body></html>")

    # 2. Compile comprehensive JSON summary combining provenance and statistical metrics
    manifest = {
        "scenario": scenario,
        "reference_dataset": reference_name,
        "production_dataset": production_name,
        "evidently_version": EVIDENTLY_VERSION,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "monitored_columns": cols_to_monitor,
        "html_report_path": html_path,
        "summary": summary_dict or {},
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return html_path, json_path


def run_drift_analysis(
    reference_data: Union[pd.DataFrame, str],
    production_data: Union[pd.DataFrame, str],
    scenario: str = "production",
    output_dir: str = DEFAULT_OUTPUT_DIR,
    features: Optional[List[str]] = None,
    target_column: Optional[str] = TARGET_COLUMN,
    generate_reports: bool = True,
    reference_name: Optional[str] = None,
    production_name: Optional[str] = None,
    ks_p_threshold: float = DEFAULT_KS_P_THRESHOLD,
    ks_stat_threshold: float = DEFAULT_KS_STAT_THRESHOLD,
    cohens_d_threshold: float = DEFAULT_COHENS_D_THRESHOLD,
    dataset_drift_share_threshold: float = DEFAULT_DATASET_DRIFT_SHARE,
) -> Dict[str, Any]:
    """Execute complete drift detection pipeline including statistical metrics and Evidently reports.

    Args:
        reference_data: Reference DataFrame or CSV path.
        production_data: Production DataFrame or CSV path.
        scenario: Scenario identifier (e.g., 'normal', 'hvac_shift', 'sensor_drift').
        output_dir: Directory for generated reports.
        features: Monitored input feature columns.
        target_column: Monitored target column.
        generate_reports: Whether to render and persist Evidently HTML and JSON reports.
        reference_name: Descriptive name/path for reference dataset.
        production_name: Descriptive name/path for production dataset.
        ks_p_threshold: P-value threshold for KS test.
        ks_stat_threshold: KS statistic threshold.
        cohens_d_threshold: Cohen's d threshold.
        dataset_drift_share_threshold: Proportion of drifted features required for dataset drift.

    Returns:
        Structured machine-readable drift analysis results dictionary.
    """
    ref_df = load_dataset(reference_data)
    prod_df = load_dataset(production_data)

    ref_label = reference_name or (str(reference_data) if isinstance(reference_data, str) else "reference_data")
    prod_label = production_name or (str(production_data) if isinstance(production_data, str) else f"simulated_{scenario}")

    summary = calculate_dataset_drift(
        reference_data=ref_df,
        production_data=prod_df,
        features=features,
        target_column=target_column,
        ks_p_threshold=ks_p_threshold,
        ks_stat_threshold=ks_stat_threshold,
        cohens_d_threshold=cohens_d_threshold,
        dataset_drift_share_threshold=dataset_drift_share_threshold,
    )

    summary["scenario"] = scenario
    summary["reference_dataset"] = ref_label
    summary["production_dataset"] = prod_label

    if generate_reports:
        html_p, json_p = generate_evidently_reports(
            reference_df=ref_df,
            production_df=prod_df,
            output_dir=output_dir,
            scenario=scenario,
            features=features,
            target_column=target_column,
            reference_name=ref_label,
            production_name=prod_label,
            summary_dict=summary,
        )
        summary["html_report_path"] = html_p
        summary["json_report_path"] = json_p

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evidently AI Data Drift Detection Engine (Phase 3.3).")
    parser.add_argument("--reference", type=str, default=DEFAULT_REF_PATH, help="Path to reference baseline CSV (default: data/train.csv)")
    parser.add_argument("--production", type=str, default=None, help="Path to production CSV to compare against")
    parser.add_argument("--scenario", choices=["normal", "hvac_shift", "sensor_drift", "all"], default="all", help="Scenario to analyze")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save drift reports")
    parser.add_argument("--samples", type=int, default=500, help="Number of simulated samples if generating on the fly")
    parser.add_argument("--seed", type=int, default=42, help="Seed for simulated production data")

    args = parser.parse_args()

    # Import simulator to generate or load scenario data
    from src.simulation import generate_simulated_data, load_reference_data

    ref_df = load_dataset(args.reference)
    scenarios = ["normal", "hvac_shift", "sensor_drift"] if args.scenario == "all" else [args.scenario]

    print(f"\n=======================================================")
    print(f"  Evidently AI Data Drift Detection (Phase 3.3)")
    print(f"  Reference: {args.reference} ({len(ref_df)} rows)")
    print(f"  Evidently Version: {EVIDENTLY_VERSION}")
    print(f"=======================================================\n")

    for sc in scenarios:
        if args.production:
            prod_df = load_dataset(args.production)
            prod_name = args.production
        else:
            prod_df = generate_simulated_data(
                scenario=sc,
                n_samples=args.samples,
                random_state=args.seed,
                ref_df=ref_df,
            )
            prod_name = f"simulated_{sc} (samples={args.samples}, seed={args.seed})"

        results = run_drift_analysis(
            reference_data=ref_df,
            production_data=prod_df,
            scenario=sc,
            output_dir=args.output_dir,
            reference_name=args.reference,
            production_name=prod_name,
        )

        status_flag = "DRIFT DETECTED" if results["dataset_drift_detected"] else "NO DATASET DRIFT"
        print(f"[{sc.upper()}] {status_flag}")
        print(f"  Drift share: {results['drift_share']:.1%} ({results['drifted_features_count']}/{results['total_features_count']} features drifted)")
        print(f"  Drifted features: {results['drifted_features']}")
        if results.get("target_drift"):
            t_flag = "DRIFTED" if results["target_drift"]["target_drift_detected"] else "STABLE"
            print(f"  Target drift ({TARGET_COLUMN}): {t_flag} (Cohen's d={results['target_drift']['statistics']['cohens_d']})")
        print(f"  HTML Report: {results.get('html_report_path')}")
        print(f"  JSON Summary: {results.get('json_report_path')}\n")
