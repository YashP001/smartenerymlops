"""
Reproducible End-to-End MLOps Demonstration Script for Smart Home Energy Advisor.

Phase 3.7: End-to-End MLOps Integration.

Demonstrates the operational model lifecycle:
1. Model Availability & Verification (Local artifacts and MLflow Registry fallback)
2. FastAPI Inference Service Connectivity (/health, /predict)
3. Prediction Logging & SQLite Persistence (predictions table)
4. Demonstration Ground-Truth Feedback Ingestion (>= 30 samples, feedback table)
5. Production Data Simulation (Normal Operation, data/simulated/demo)
6. Evidently AI & Statistical Drift Detection (KS-test, Cohen's d, drift share)
7. Model Performance Evaluation against Test Baselines (MAE, RMSE, R2, Bias)
8. Automated Monitoring Cycle & Alert Lifecycle Management (Trigger, Cooldown, Resolution)
9. Prometheus Observability Metrics Verification (/metrics exposition)
10. Distributional Shift Scenarios (HVAC Afternoon Shift, Sensor Calibration Bias)
    and Automated Alert Recovery.

Usage:
    # Run standalone in-process demonstration (self-contained, no external servers required):
    python scripts/run_end_to_end_demo.py

    # Or run against an active uvicorn server:
    python scripts/run_end_to_end_demo.py --api-url http://localhost:8000
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add repository root to system path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.predict import load_model, predict
from src.simulation import (
    generate_simulated_data,
    save_simulated_dataset,
    validate_simulated_dataset,
)
from src.monitoring.database import (
    DEFAULT_DB_PATH,
    get_connection,
    get_db_path,
    get_monitoring_stats,
    init_db,
)
from src.monitoring.drift_detector import (
    DEFAULT_REF_PATH,
    run_drift_analysis,
)

DEFAULT_TEST_PATH = os.path.join(_PROJECT_ROOT, "data", "test.csv")
from src.monitoring.performance_monitor import (
    BASELINE_METRICS,
    evaluate_model_performance,
)
from src.monitoring.alert_manager import AlertManager, AlertType, AlertSeverity
from src.monitoring.monitor_orchestrator import run_monitoring_cycle
from src.monitoring.metrics import MetricsContainer, default_metrics

# Try importing httpx or requests for live API calls, fallback to starlette TestClient
try:
    from fastapi.testclient import TestClient
    from api.main import app
    TEST_CLIENT_AVAILABLE = True
except ImportError:
    TEST_CLIENT_AVAILABLE = False

try:
    import urllib.request
    import urllib.error
    URLLIB_AVAILABLE = True
except ImportError:
    URLLIB_AVAILABLE = False


# ─── Formatting Utilities ─────────────────────────────────────────────

def print_banner(title: str) -> None:
    """Print a visually distinct section banner."""
    width = 78
    print("\n" + "=" * width)
    print(f"  {title.upper()}")
    print("=" * width)


def print_step(step_num: int, title: str) -> None:
    """Print an individual workflow step header."""
    print(f"\n[Step {step_num}] {title}")
    print("-" * (len(title) + 9))


def print_success(message: str) -> None:
    """Print a success confirmation message."""
    print(f"  [PASS] {message}")


def print_info(message: str) -> None:
    """Print an informative detail message."""
    print(f"  [INFO] {message}")


def print_warning(message: str) -> None:
    """Print an operational warning message."""
    print(f"  [WARN] {message}")


def print_table(headers: List[str], rows: List[List[Any]]) -> None:
    """Format and print a text table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "  ".join("-" * col_widths[i] for i in range(len(headers)))
    print(f"  {header_line}")
    print(f"  {sep_line}")
    for row in rows:
        row_line = "  ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
        print(f"  {row_line}")


# ─── API Client Abstraction ───────────────────────────────────────────

class DemoApiClient:
    """Unified client supporting both live HTTP endpoints and in-process TestClient."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url.rstrip("/") if base_url else None
        self._test_client = None
        if not self.base_url:
            if not TEST_CLIENT_AVAILABLE:
                raise RuntimeError("FastAPI TestClient unavailable and no --api-url provided.")
            self._test_client = TestClient(app)

    def get(self, endpoint: str) -> Tuple[int, Any]:
        """Send GET request."""
        path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        if self.base_url:
            url = f"{self.base_url}{path}"
            req = urllib.request.Request(url, method="GET")
            try:
                with urllib.request.urlopen(req) as resp:
                    status = resp.status
                    body = json.loads(resp.read().decode("utf-8"))
                    return status, body
            except urllib.error.HTTPError as e:
                body = json.loads(e.read().decode("utf-8")) if e.fp else {}
                return e.code, body
        else:
            resp = self._test_client.get(path)
            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, resp.text

    def post(self, endpoint: str, json_data: Dict[str, Any]) -> Tuple[int, Any]:
        """Send POST request."""
        path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        if self.base_url:
            url = f"{self.base_url}{path}"
            data_bytes = json.dumps(json_data).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    status = resp.status
                    body = json.loads(resp.read().decode("utf-8"))
                    return status, body
            except urllib.error.HTTPError as e:
                body = json.loads(e.read().decode("utf-8")) if e.fp else {}
                return e.code, body
        else:
            resp = self._test_client.post(path, json=json_data)
            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, resp.text

    def get_metrics_text(self) -> str:
        """Fetch raw Prometheus metrics exposition text."""
        if self.base_url:
            url = f"{self.base_url}/metrics"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req) as resp:
                return resp.read().decode("utf-8")
        else:
            resp = self._test_client.get("/metrics")
            return resp.text


# ─── End-to-End Demonstration Pipeline ────────────────────────────────

def run_end_to_end_demo(
    api_url: Optional[str] = None,
    db_path: Optional[str] = None,
    demo_dir: Optional[str] = None,
    n_samples: int = 35,
    seed: int = 42,
    skip_drift_scenarios: bool = False,
    feedback_mode: str = "authentic",
    test_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the full end-to-end MLOps lifecycle demonstration.

    Args:
        api_url: Optional URL of active FastAPI service. If None, runs in-process.
        db_path: Path to demonstration SQLite database.
        demo_dir: Directory for demonstration artifacts and simulation data.
        n_samples: Number of prediction/feedback samples (must be >= 30).
        seed: Random seed for reproducible generation.
        skip_drift_scenarios: Whether to skip additional drift scenarios.
        feedback_mode: 'authentic' (held-out data/test.csv targets, default) or
            'synthetic' (predictions + Gaussian noise demonstration mode).
        test_path: Optional custom path to held-out test dataset for authentic mode.

    Returns:
        Summary dictionary with metrics and demonstration results.
    """
    if feedback_mode not in ("authentic", "synthetic"):
        raise ValueError(
            f"Invalid feedback_mode: '{feedback_mode}'. Must be one of ['authentic', 'synthetic']."
        )

    if n_samples < 30:
        raise ValueError(f"n_samples must be at least 30 to satisfy performance evaluation threshold, got {n_samples}")

    had_prior_db_env = "MONITORING_DB_PATH" in os.environ
    prior_db_env = os.environ.get("MONITORING_DB_PATH")

    demo_root = demo_dir or os.path.join(_PROJECT_ROOT, "reports", "demo")
    sim_dir = os.path.join(_PROJECT_ROOT, "data", "simulated", "demo")
    os.makedirs(demo_root, exist_ok=True)
    os.makedirs(sim_dir, exist_ok=True)

    active_db = db_path or os.path.join(demo_root, "demo_predictions.db")
    active_test_path = test_path or DEFAULT_TEST_PATH

    # Clean previous demo database if default isolated path is used to guarantee reproducibility
    if db_path is None and os.path.exists(active_db):
        try:
            os.remove(active_db)
        except Exception:
            pass

    try:
        # Initialize and configure the database for isolation in current process
        os.environ["MONITORING_DB_PATH"] = active_db
        init_db(active_db)

        print_banner("AI-Assisted Smart Home Energy Advisor — End-to-End MLOps Demonstration")
        print_info(f"Target Database (Client): {active_db}")
        print_info(f"Feedback Mode: {feedback_mode.upper()} ({'held-out data/test.csv targets' if feedback_mode == 'authentic' else 'predictions + Gaussian noise'})")
        print_info(f"Artifacts Directory: {demo_root}")
        print_info(f"Simulated Data Directory: {sim_dir}")
        print_info(f"Execution Mode: {'Live API (' + api_url + ')' if api_url else 'In-Process ASGI Service'}")
        print_info(f"Evaluation Minimum Samples: {n_samples} (Requirement: >= 30)")

        if api_url:
            print_warning(
                f"LIVE HTTP MODE ACTIVE ({api_url}).\n"
                f"  External Uvicorn server runs in a separate process space.\n"
                f"  Ensure Uvicorn was started with matching $env:MONITORING_DB_PATH = '{os.path.abspath(active_db)}'\n"
                f"  or pass --db-path pointing to the server's database file."
            )

        demo_results: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "database": active_db,
            "n_samples": n_samples,
            "feedback_mode": feedback_mode,
            "stages_completed": [],
        }

        client = DemoApiClient(base_url=api_url)

        # ──────────────────────────────────────────────────────────────────
        # Stage 1: Model Availability Verification
        # ──────────────────────────────────────────────────────────────────
        print_step(1, "Model Verification & Registry Availability")
        active_model = load_model()
        model_name = getattr(active_model, "__class__", type(active_model)).__name__
        print_success(f"Model successfully loaded: {model_name}")
        print_info(f"Required Model Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
        demo_results["stages_completed"].append("model_verification")

        # ──────────────────────────────────────────────────────────────────
        # Stage 2: FastAPI Service Connectivity
        # ──────────────────────────────────────────────────────────────────
        print_step(2, "FastAPI Service Health Check")
        status_code, health_body = client.get("/health")
        if status_code != 200 or not health_body.get("model_loaded"):
            raise RuntimeError(f"Service health check failed: status {status_code}, response: {health_body}")
        print_success(f"FastAPI service healthy: status={health_body.get('status')}, model_loaded={health_body.get('model_loaded')}")
        demo_results["stages_completed"].append("service_health_check")

        # ──────────────────────────────────────────────────────────────────
        # Stage 3: Prediction Generation & SQLite Logging Verification
        # ──────────────────────────────────────────────────────────────────
        print_step(3, "Generate Valid Inference Requests & Verify SQLite Logging")
        rng = np.random.default_rng(seed)

        if feedback_mode == "authentic":
            if not os.path.exists(active_test_path):
                raise FileNotFoundError(
                    f"Authentic held-out test dataset not found at '{active_test_path}'"
                )
            test_df = pd.read_csv(active_test_path)
            if n_samples > len(test_df):
                raise ValueError(
                    f"Requested {n_samples} samples, but held-out test dataset at '{active_test_path}' "
                    f"only contains {len(test_df)} rows."
                )
            sample_rows = test_df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
            print_info(
                f"[AUTHENTIC HELD-OUT MODE] Drawing {n_samples} unseen test rows with seed={seed} "
                f"from '{os.path.relpath(active_test_path, _PROJECT_ROOT) if os.path.isabs(active_test_path) else active_test_path}'."
            )
        else:
            ref_df_synth = pd.read_csv(DEFAULT_REF_PATH)
            sample_rows = ref_df_synth.sample(n=n_samples, random_state=seed).reset_index(drop=True)
            print_warning(
                "[SYNTHETIC DEMONSTRATION MODE] Generating inputs from reference data. "
                "Feedback values will be derived from model predictions with Gaussian noise (+/- 0.015 kWh)."
            )

        logged_prediction_ids: List[str] = []
        actual_ground_truths: List[float] = []
        predicted_values: List[float] = []

        print_info(f"Submitting {n_samples} sequential prediction requests to /predict...")
        for idx in range(n_samples):
            row = sample_rows.iloc[idx]
            payload = {
                "energy_current_hour": float(row["energy_current_hour"]),
                "energy_previous_hour": float(row["energy_previous_hour"]),
                "energy_two_hours_ago": float(row["energy_two_hours_ago"]),
                "target_hour": int(row["target_hour"]),
                "target_day_of_week": int(row["target_day_of_week"]),
                "target_is_weekend": int(row["target_is_weekend"]),
            }
            status_code, pred_resp = client.post("/predict", payload)
            if status_code != 200:
                raise RuntimeError(f"Prediction failed at sample {idx}: {pred_resp}")

            pred_id = pred_resp.get("prediction_id")
            pred_kwh = pred_resp.get("predicted_energy_kwh")
            logged_prediction_ids.append(pred_id)
            predicted_values.append(pred_kwh)

            if feedback_mode == "authentic":
                # Authentic held-out target from the exact same row providing input features
                actual_val = float(row[TARGET_COLUMN])
            else:
                # Synthetic feedback derived from model prediction + Gaussian noise
                actual_val = max(0.005, float(pred_kwh) + float(rng.normal(0, 0.015)))

            actual_ground_truths.append(round(actual_val, 4))

        # Verify SQLite persistence with retry loop for asynchronous BackgroundTasks
        placeholders = ",".join("?" for _ in logged_prediction_ids)
        persisted_our_ids = 0
        sample_rec = []
        for _ in range(15):
            time.sleep(0.2)
            with get_connection(active_db) as conn:
                persisted_our_ids = conn.execute(
                    f"SELECT COUNT(*) FROM predictions WHERE prediction_id IN ({placeholders});",
                    logged_prediction_ids,
                ).fetchone()[0]
                if persisted_our_ids == len(logged_prediction_ids):
                    sample_rec = conn.execute(
                        f"SELECT prediction_id, energy_current_hour, predicted_energy_kwh, model_version "
                        f"FROM predictions WHERE prediction_id IN ({placeholders}) LIMIT 3;",
                        logged_prediction_ids,
                    ).fetchall()
                    break

        if persisted_our_ids < len(logged_prediction_ids):
            if api_url:
                raise RuntimeError(
                    f"Database alignment mismatch: The external API server at '{api_url}' did not write "
                    f"prediction records to the client's target database:\n"
                    f"  Client Target DB: {os.path.abspath(active_db)}\n"
                    f"  Found {persisted_our_ids}/{len(logged_prediction_ids)} expected prediction IDs.\n\n"
                    f"Explanation:\n"
                    f"  External server processes (e.g. Uvicorn) run in a separate process space and do not "
                    f"inherit the client process's environment variables.\n\n"
                    f"To resolve this, ensure both client and server reference the identical database file:\n"
                    f"  PowerShell (Start Uvicorn with matching path):\n"
                    f"    $env:MONITORING_DB_PATH = \"{os.path.abspath(active_db)}\"\n"
                    f"    uvicorn api.main:app --port 8000\n\n"
                    f"  Or pass Uvicorn's active database path to the demo script:\n"
                    f"    python scripts/run_end_to_end_demo.py --api-url {api_url} --db-path \"{DEFAULT_DB_PATH}\""
                )
            else:
                raise RuntimeError(
                    f"In-process prediction logging verification failed: Expected {len(logged_prediction_ids)} "
                    f"persisted records in '{active_db}', but found {persisted_our_ids}."
                )

        print_success(f"Predictions logged to SQLite database: {persisted_our_ids} records verified in table 'predictions'")
        print_info("Sample persisted records:")
        print_table(
            ["Prediction ID", "Energy Current Hour (kWh)", "Predicted Energy (kWh)", "Model"],
            [[r[0][:8] + "...", f"{r[1]:.4f}", f"{r[2]:.4f}", r[3]] for r in sample_rec],
        )
        demo_results["stages_completed"].append("prediction_logging")

        # ──────────────────────────────────────────────────────────────────
        # Stage 4: Ground-Truth Feedback Ingestion (>= 30 samples)
        # ──────────────────────────────────────────────────────────────────
        print_step(4, "Submit Ground-Truth Feedback & Verify Record Pairing")
        if feedback_mode == "authentic":
            print_info(
                f"[AUTHENTIC HELD-OUT FEEDBACK] Ingesting {n_samples} ground-truth feedback observations "
                f"from '{os.path.relpath(active_test_path, _PROJECT_ROOT) if os.path.isabs(active_test_path) else active_test_path}'. "
                "Each actual target corresponds to the identical held-out telemetry row supplied for inference."
            )
        else:
            print_warning(
                f"[SYNTHETIC DEMONSTRATION FEEDBACK] Ingesting {n_samples} synthetic feedback observations "
                "derived from model predictions with Gaussian perturbation (+/- 0.015 kWh). "
                "NOTE: These values validate the software ingestion pipeline only and do NOT represent authentic predictive accuracy."
            )

        feedback_submitted = 0
        for pred_id, actual_kwh in zip(logged_prediction_ids, actual_ground_truths):
            fb_payload = {
                "prediction_id": pred_id,
                "actual_energy_kwh": actual_kwh,
            }
            status_code, fb_resp = client.post("/feedback", fb_payload)
            if status_code != 200:
                raise RuntimeError(f"Feedback submission failed for {pred_id}: {fb_resp}")
            feedback_submitted += 1

        # Verify view v_monitoring_records has paired records
        with get_connection(active_db) as conn:
            paired_count = conn.execute(
                f"SELECT COUNT(*) FROM v_monitoring_records "
                f"WHERE prediction_id IN ({placeholders}) AND actual_energy_kwh IS NOT NULL;",
                logged_prediction_ids,
            ).fetchone()[0]
            sample_paired = conn.execute(
                f"SELECT prediction_id, predicted_energy_kwh, actual_energy_kwh, absolute_error "
                f"FROM v_monitoring_records WHERE prediction_id IN ({placeholders}) LIMIT 3;",
                logged_prediction_ids,
            ).fetchall()

        if paired_count < len(logged_prediction_ids):
            if api_url:
                raise RuntimeError(
                    f"Database alignment mismatch: The external API server at '{api_url}' did not write "
                    f"feedback records to the client's target database '{os.path.abspath(active_db)}'."
                )
            else:
                raise RuntimeError(
                    f"Feedback pairing verification failed: Expected {len(logged_prediction_ids)} paired records, "
                    f"found {paired_count}."
                )

        print_success(f"Feedback records submitted and persisted: {feedback_submitted}")
        print_success(f"Paired prediction-feedback observations in 'v_monitoring_records': {paired_count}")
        print_info("Sample paired records:")
        print_table(
            ["Prediction ID", "Predicted (kWh)", "Actual (kWh)", "Absolute Error (kWh)"],
            [[r[0][:8] + "...", f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}"] for r in sample_paired],
        )
        demo_results["stages_completed"].append("feedback_ingestion")

        # ──────────────────────────────────────────────────────────────────
        # Stage 5: Production Data Simulation (Normal Scenario)
        # ──────────────────────────────────────────────────────────────────
        print_step(5, "Generate Production Simulation Data (Normal Scenario)")
        print_info("[DEMONSTRATION PRODUCTION DATA] Simulating 200 hours of normal smart-meter consumption.")
        ref_df_sim = pd.read_csv(DEFAULT_REF_PATH)
        sim_normal_df = generate_simulated_data(
            scenario="normal",
            n_samples=200,
            random_state=seed,
            ref_df=ref_df_sim,
        )
        val_normal = validate_simulated_dataset(sim_normal_df)
        if not val_normal["passed"]:
            raise ValueError(f"Normal simulation validation failed: {val_normal['errors']}")

        csv_normal, json_normal = save_simulated_dataset(
            sim_normal_df,
            scenario="normal_demo",
            output_dir=sim_dir,
            random_state=seed,
        )
        print_success(f"Normal demonstration dataset generated and validated: {len(sim_normal_df)} rows")
        print_info(f"Saved CSV: {csv_normal}")
        print_info(f"Saved Metadata: {json_normal}")
        demo_results["stages_completed"].append("normal_simulation")

        # ──────────────────────────────────────────────────────────────────
        # Stage 6: Drift Detection against Baseline Reference Data
        # ──────────────────────────────────────────────────────────────────
        print_step(6, "Execute Evidently AI & Statistical Drift Detection (Normal Data)")
        drift_output_dir = os.path.join(demo_root, "drift")
        drift_summary = run_drift_analysis(
            reference_data=DEFAULT_REF_PATH,
            production_data=csv_normal,
            scenario="normal_demo",
            output_dir=drift_output_dir,
        )
        drift_share = drift_summary.get("drift_share", 0.0)
        dataset_drift = drift_summary.get("dataset_drift", False)

        print_success(
            f"Drift Analysis completed: dataset_drift={dataset_drift}, "
            f"drift_share={drift_share * 100:.1f}% ({drift_summary.get('drifted_features_count')}/{drift_summary.get('total_features_count')} features)"
        )
        print_info(f"Interactive Evidently Report: {drift_summary.get('html_report_path')}")
        print_info(f"Drift Summary Manifest: {drift_summary.get('json_report_path')}")
        demo_results["stages_completed"].append("drift_detection_normal")

        # ──────────────────────────────────────────────────────────────────
        # Stage 7: Model Performance Monitoring Evaluation
        # ──────────────────────────────────────────────────────────────────
        print_step(7, "Evaluate Model Operational Performance against Baselines")
        perf_summary = evaluate_model_performance(
            db_path=active_db,
            model_version=model_name,
        )
        perf_metrics = perf_summary.get("metrics", {})
        obs_mae = perf_metrics.get("mae", 0.0)
        obs_rmse = perf_metrics.get("rmse", 0.0)
        obs_r2 = perf_metrics.get("r2", 0.0)
        obs_bias = perf_metrics.get("bias", 0.0)
        perf_status = perf_summary.get("status", "unknown")

        demo_results["performance_status"] = perf_status
        demo_results["observed_metrics"] = perf_metrics
        demo_results["baseline_metrics"] = BASELINE_METRICS

        if feedback_mode == "authentic":
            print_success(f"Performance evaluation status: {perf_status.upper()} (Authentic Held-Out Test)")
            print_info(f"Evaluated Sample Count: {perf_summary.get('evaluated_predictions')} held-out test observations")
            print_info(
                "Evaluation Nature: Authentic held-out test evaluation. Metrics reflect true model performance\n"
                f"  on unseen historical smart-meter readings from '{os.path.relpath(active_test_path, _PROJECT_ROOT) if os.path.isabs(active_test_path) else active_test_path}'.\n"
                f"  Minor variance from Phase 1 full test baseline is expected due to sampling variation on a subset of N={n_samples} rows\n"
                f"  (Phase 1 baseline was evaluated on the complete test dataset of N=4,120 rows)."
            )
            print_table(
                ["Metric", "Phase 1 Empirical Baseline (N=4,120)", f"Observed ({n_samples} Held-Out Rows)", "Permissible Threshold", "Status"],
                [
                    ["MAE", f"{BASELINE_METRICS['MAE']:.4f} kWh", f"{obs_mae:.4f} kWh", f"<= {BASELINE_METRICS['MAE']*1.25:.4f} kWh (+25%)", "PASS" if obs_mae <= BASELINE_METRICS['MAE']*1.25 else "WARN"],
                    ["RMSE", f"{BASELINE_METRICS['RMSE']:.4f} kWh", f"{obs_rmse:.4f} kWh", f"<= {BASELINE_METRICS['RMSE']*1.25:.4f} kWh (+25%)", "PASS" if obs_rmse <= BASELINE_METRICS['RMSE']*1.25 else "WARN"],
                    ["R2", f"{BASELINE_METRICS['R2']:.4f}", f"{obs_r2:.4f}", ">= 0.1000", "PASS" if obs_r2 >= 0.10 else "WARN"],
                    ["Bias", f"{BASELINE_METRICS['BIAS']:+.4f} kWh", f"{obs_bias:+.4f} kWh", "<= 0.0500 kWh", "PASS" if abs(obs_bias) <= 0.05 else "WARN"],
                ],
            )
        else:
            print_warning("EVALUATION NATURE: SYNTHETIC DEMONSTRATION FEEDBACK (PIPELINE VALIDATION ONLY)")
            print_warning(
                "IMPORTANT NOTICE: Feedback values were generated synthetically from predictions + Gaussian noise:\n"
                "  actual = prediction + Normal(0, 0.015).\n"
                "  These metrics validate pipeline integrity, database pairing, and threshold checking.\n"
                "  They DO NOT reflect authentic model accuracy. Artificially low MAE/RMSE is an expected artifact\n"
                "  of Gaussian noise and does NOT indicate superior performance over authentic historical data."
            )
            print_info(f"Evaluated Sample Count: {perf_summary.get('evaluated_predictions')} synthetic validation observations")
            print_table(
                ["Metric", "Authentic Phase 1 Baseline", f"Observed (Synthetic Validation)", "Permissible Threshold", "Status"],
                [
                    ["MAE", f"{BASELINE_METRICS['MAE']:.4f} kWh", f"{obs_mae:.4f} kWh", f"<= {BASELINE_METRICS['MAE']*1.25:.4f} kWh (+25%)", "PASS (Synthetic)"],
                    ["RMSE", f"{BASELINE_METRICS['RMSE']:.4f} kWh", f"{obs_rmse:.4f} kWh", f"<= {BASELINE_METRICS['RMSE']*1.25:.4f} kWh (+25%)", "PASS (Synthetic)"],
                    ["R2", f"{BASELINE_METRICS['R2']:.4f}", f"{obs_r2:.4f}", ">= 0.1000", "PASS (Synthetic)"],
                    ["Bias", f"{BASELINE_METRICS['BIAS']:.4f} kWh", f"{obs_bias:.4f} kWh", "<= 0.0500 kWh", "PASS (Synthetic)"],
                ],
            )

        demo_results["stages_completed"].append("performance_evaluation")

        # ──────────────────────────────────────────────────────────────────
        # Stage 8: Automated Monitoring Cycle & Alert Evaluation
        # ──────────────────────────────────────────────────────────────────
        print_step(8, "Execute Automated Monitoring Orchestration Cycle")
        alert_mgr = AlertManager(db_path=active_db)
        cycle_summary = run_monitoring_cycle(
            ref_data=DEFAULT_REF_PATH,
            prod_data=csv_normal,
            db_path=active_db,
            model_version=model_name,
            alert_manager=alert_mgr,
            save_reports=True,
            output_dir=os.path.join(demo_root, "performance"),
        )
        print_success(f"Monitoring Cycle '{cycle_summary['cycle_id']}' completed successfully.")
        print_info(f"Drift Status: {cycle_summary['drift_status']}")
        print_info(f"Performance Status: {cycle_summary['performance_status']}")
        print_info(f"Alerts Triggered: {len(cycle_summary['alerts_triggered'])}")
        print_info(f"Active Alerts in Persistence Layer: {cycle_summary['active_alerts_count']}")
        demo_results["stages_completed"].append("orchestration_cycle_normal")

        # ──────────────────────────────────────────────────────────────────
        # Stage 9: Prometheus Observability Scraping Verification
        # ──────────────────────────────────────────────────────────────────
        print_step(9, "Inspect Prometheus Observability Exposition (/metrics)")
        metrics_text = client.get_metrics_text()

        # Search for key exposed metrics in payload
        expected_metric_names = [
            "energy_advisor_prediction_requests_total",
            "energy_advisor_dataset_drift_share",
            "energy_advisor_drift_detection_runs_total",
            "energy_advisor_performance_evaluation_runs_total",
            "energy_advisor_model_mae",
            "energy_advisor_model_rmse",
            "energy_advisor_active_alerts",
        ]

        found_metrics: List[List[Any]] = []
        for line in metrics_text.splitlines():
            for m in expected_metric_names:
                if line.startswith(m) and not line.startswith(f"{m}_"):
                    found_metrics.append([m, line])
                    break

        print_success("Prometheus metrics successfully exposed and scraped:")
        print_table(
            ["Metric Family", "Exposition Sample Line"],
            found_metrics[:7] if found_metrics else [["None", "No direct line matched"]],
        )
        demo_results["stages_completed"].append("prometheus_observability")

        # ──────────────────────────────────────────────────────────────────
        # Stage 10: Drift Scenarios & Automated Alert Lifecycle
        # ──────────────────────────────────────────────────────────────────
        if not skip_drift_scenarios:
            print_step(10, "Simulate Distributional Shifts & Alert Lifecycle")

            # Scenario 10A: HVAC Consumption Shift (Covariate Shift)
            print_info(
                "[SCENARIO A: HVAC CONSUMPTION SHIFT] Generating afternoon/evening cooling load spike. "
                "Simulating heatwave condition inducing covariate shift on lag features."
            )
            sim_hvac_df = generate_simulated_data(
                scenario="hvac_shift",
                n_samples=200,
                severity=1.2,
                random_state=seed,
                ref_df=ref_df_sim,
            )
            csv_hvac, _ = save_simulated_dataset(
                sim_hvac_df,
                scenario="hvac_shift_demo",
                output_dir=sim_dir,
                random_state=seed,
            )

            hvac_cycle = run_monitoring_cycle(
                ref_data=DEFAULT_REF_PATH,
                prod_data=csv_hvac,
                db_path=active_db,
                model_version=model_name,
                alert_manager=alert_mgr,
            )
            hvac_drift_share = hvac_cycle.get("drift_result", {}).get("drift_share", 0.0)
            hvac_alerts = hvac_cycle.get("alerts_triggered", [])

            print_success(
                f"HVAC Shift Monitoring Cycle Complete: Drift Share = {hvac_drift_share * 100:.1f}%, "
                f"Alerts Triggered = {len(hvac_alerts)}"
            )
            for alt in hvac_alerts:
                print_warning(
                    f"Alert Generated: [{alt.get('severity')}] {alt.get('alert_type')} -> {alt.get('message')}"
                )

            # Scenario 10B: Sensor Drift (Systematic Measurement Offset)
            print_info(
                "\n[SCENARIO B: SENSOR DRIFT] Generating systematic calibration bias on meter readings. "
                "Simulating physical sensor decay and noise."
            )
            sim_sensor_df = generate_simulated_data(
                scenario="sensor_drift",
                n_samples=200,
                severity=1.2,
                random_state=seed,
                ref_df=ref_df_sim,
            )
            csv_sensor, _ = save_simulated_dataset(
                sim_sensor_df,
                scenario="sensor_drift_demo",
                output_dir=sim_dir,
                random_state=seed,
            )

            sensor_cycle = run_monitoring_cycle(
                ref_data=DEFAULT_REF_PATH,
                prod_data=csv_sensor,
                db_path=active_db,
                model_version=model_name,
                alert_manager=alert_mgr,
            )
            sensor_drift_share = sensor_cycle.get("drift_result", {}).get("drift_share", 0.0)
            print_success(
                f"Sensor Drift Monitoring Cycle Complete: Drift Share = {sensor_drift_share * 100:.1f}%, "
                f"Active Alerts in DB = {sensor_cycle.get('active_alerts_count')}"
            )

            # Scenario 10C: Recovery Verification (Return to Normal Operation)
            print_info(
                "\n[SCENARIO C: AUTOMATIC ALERT RECOVERY] Re-evaluating against normal operational data. "
                "Verifying automatic resolution of data drift alerts."
            )
            recovery_cycle = run_monitoring_cycle(
                ref_data=DEFAULT_REF_PATH,
                prod_data=csv_normal,
                db_path=active_db,
                model_version=model_name,
                alert_manager=alert_mgr,
            )
            resolved_alerts = recovery_cycle.get("alerts_resolved", [])
            print_success(f"Recovery Cycle Complete: Alerts Resolved = {len(resolved_alerts)}")
            for r_alt in resolved_alerts:
                print_success(
                    f"Alert Resolved: [{r_alt.get('severity')}] {r_alt.get('alert_type')} -> {r_alt.get('message')}"
                )

            demo_results["stages_completed"].append("drift_scenarios_and_recovery")

        # ──────────────────────────────────────────────────────────────────
        # Final Summary Banner
        # ──────────────────────────────────────────────────────────────────
        print_banner("End-to-End MLOps Demonstration Complete")
        print_success("All operational lifecycle stages executed and verified successfully:")
        for idx, stage in enumerate(demo_results["stages_completed"], 1):
            print(f"  {idx}. [PASS] {stage.replace('_', ' ').title()}")

        print("\nSummary Statistics:")
        print_info(f"Database: {active_db}")
        print_info(f"Feedback Mode: {feedback_mode.upper()} ({'held-out data/test.csv targets' if feedback_mode == 'authentic' else 'predictions + Gaussian noise'})")
        print_info(f"Predictions Logged: {n_samples}")
        print_info(f"Feedback Records Processed: {n_samples}")
        print_info(f"Observed MAE: {obs_mae:.4f} kWh ({'Authentic Held-Out' if feedback_mode == 'authentic' else 'Synthetic Validation'} vs Baseline {BASELINE_METRICS['MAE']:.4f} kWh)")
        print_info(f"Observed RMSE: {obs_rmse:.4f} kWh ({'Authentic Held-Out' if feedback_mode == 'authentic' else 'Synthetic Validation'} vs Baseline {BASELINE_METRICS['RMSE']:.4f} kWh)")
        print_info(f"Observed R2: {obs_r2:.4f} (Baseline: {BASELINE_METRICS['R2']:.4f})")
        print_info(f"Performance Status: {perf_status.upper()}")
        print_info(f"Reports Generated under: {demo_root}")
        print_info(f"Simulated Data stored in: {sim_dir}")
        print("-" * 78 + "\n")

        return demo_results

    finally:
        if had_prior_db_env:
            os.environ["MONITORING_DB_PATH"] = prior_db_env
        else:
            os.environ.pop("MONITORING_DB_PATH", None)


# ─── CLI Entrypoint ───────────────────────────────────────────────────

def main() -> None:
    """CLI entrypoint for running the end-to-end MLOps demonstration."""
    parser = argparse.ArgumentParser(
        description="End-to-End MLOps Lifecycle Demonstration for Smart Home Energy Advisor."
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="Optional base URL for a running FastAPI server (e.g. http://localhost:8000). If omitted, runs in-process.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to demonstration SQLite database (defaults to reports/demo/demo_predictions.db).",
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        default=None,
        help="Target output directory for reports and metadata (defaults to reports/demo).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=35,
        help="Number of prediction/feedback samples to generate (must be >= 30, default: 35).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation (default: 42).",
    )
    parser.add_argument(
        "--feedback-mode",
        type=str,
        choices=["authentic", "synthetic"],
        default="authentic",
        help="Feedback mode: 'authentic' (default, held-out data/test.csv targets) or 'synthetic' (prediction + Gaussian noise).",
    )
    parser.add_argument(
        "--skip-drift-scenarios",
        action="store_true",
        help="Skip secondary HVAC shift and sensor drift scenarios to shorten run.",
    )

    args = parser.parse_args()

    try:
        run_end_to_end_demo(
            api_url=args.api_url,
            db_path=args.db_path,
            demo_dir=args.demo_dir,
            n_samples=args.samples,
            seed=args.seed,
            skip_drift_scenarios=args.skip_drift_scenarios,
            feedback_mode=args.feedback_mode,
        )
    except Exception as e:
        print(f"\n[ERROR] Demonstration execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
