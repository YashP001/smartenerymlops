"""
Demonstration Artifact Logger for MLflow.

Provides modular functions to log evaluation reports, feature importance plots,
held-out test set diagnostics, and dataset lineage tags to an MLflow run.
Designed for both production demonstration against the local MLflow registry
and isolated unit/integration testing in CI environments.

Does NOT retrain the model, modify registration, or overwrite existing artifacts.
"""

import hashlib
import os
import pathlib
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

# Configure matplotlib to non-interactive Agg backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Silence MLflow agent tracing hints
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import mlflow
from mlflow.tracking import MlflowClient

DEFAULT_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///E:/mlops%20phase%202/smart-home-energy-advisor/mlflow.db",
)
DEFAULT_RUN_ID = os.getenv("MLFLOW_RUN_ID", "e9922ecbb3af4c029aebf2f19b755694")
DEFAULT_MODEL_URI = "models:/smart-home-energy-advisor/1"

FEATURE_COLUMNS = [
    "energy_current_hour",
    "energy_previous_hour",
    "energy_two_hours_ago",
    "target_hour",
    "target_day_of_week",
    "target_is_weekend",
]
TARGET_COLUMN = "target_next_hour"


def compute_file_hashes(file_path: pathlib.Path) -> Tuple[str, str]:
    """Compute MD5 and SHA256 hashes of a file."""
    with open(file_path, "rb") as f:
        content = f.read()
        md5 = hashlib.md5(content).hexdigest()
        sha256 = hashlib.sha256(content).hexdigest()
    return md5, sha256


def generate_feature_importance_plot(
    model,
    output_path: str,
    feature_names: Optional[List[str]] = None,
    title_info: Optional[str] = None,
) -> None:
    """Generate and save a horizontal feature importance bar chart."""
    features = feature_names or FEATURE_COLUMNS
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise ValueError("Model does not expose 'feature_importances_'.")

    indices = np.argsort(importances)[::-1]
    sorted_features = [features[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(9, 5.5))
    bars = plt.barh(
        range(len(sorted_features)),
        sorted_importances[::-1],
        color="#1f77b4",
        edgecolor="#0b3d60",
        height=0.6,
    )
    plt.yticks(range(len(sorted_features)), sorted_features[::-1], fontsize=10)
    plt.xlabel(
        "Gini / Mean Decrease in Impurity (Feature Importance)",
        fontsize=11,
        fontweight="bold",
    )
    subtitle = (
        f"\nModel: {title_info}"
        if title_info
        else "\nModel: smart-home-energy-advisor:1 (MLflow Run e9922ecbb3af4c029aebf2f19b755694)"
    )
    plt.title(
        f"Feature Importance — Gradient Boosting Regressor{subtitle}",
        fontsize=12,
        pad=12,
        fontweight="bold",
    )
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        pct = width * 100
        plt.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}% ({width:.4f})",
            va="center",
            ha="left",
            fontsize=9,
            color="#222222",
        )

    max_w = max(sorted_importances) if len(sorted_importances) > 0 else 1.0
    plt.xlim(0, max_w * 1.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_actual_vs_predicted_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: str,
    metrics: Dict[str, float],
    title_info: Optional[str] = None,
    note_text: Optional[str] = None,
) -> None:
    """Generate and save an Actual vs Predicted scatter plot."""
    plt.figure(figsize=(8, 7))
    plt.scatter(
        y_true,
        y_pred,
        alpha=0.35,
        edgecolors="none",
        s=18,
        color="#1f77b4",
        label=f"Predictions (N={len(y_true):,})",
    )

    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    padding = (max_val - min_val) * 0.05 if max_val > min_val else 0.1
    line_min = max(0.0, min_val - padding)
    line_max = max_val + padding
    plt.plot(
        [line_min, line_max],
        [line_min, line_max],
        color="#d62728",
        linestyle="--",
        linewidth=1.75,
        label="Perfect Forecast (y = y_hat)",
    )

    plt.xlabel(
        "Actual Energy Consumption Next Hour (kWh)",
        fontsize=11,
        fontweight="bold",
    )
    plt.ylabel(
        "Predicted Energy Consumption Next Hour (kWh)",
        fontsize=11,
        fontweight="bold",
    )
    subtitle = (
        f"\nModel Checkpoint: {title_info}"
        if title_info
        else "\nModel Checkpoint: MLflow Registry smart-home-energy-advisor:1 (Run e9922ecbb3af4c029aebf2f19b755694)"
    )
    plt.title(
        f"Actual vs. Predicted Energy Consumption — Held-Out Test Set{subtitle}",
        fontsize=11.5,
        pad=12,
        fontweight="bold",
    )
    plt.xlim(line_min, line_max)
    plt.ylim(line_min, line_max)
    plt.grid(True, linestyle="--", alpha=0.5)

    bias_val = metrics.get("Bias", 0.0)
    mae_val = metrics.get("MAE", 0.0)
    rmse_val = metrics.get("RMSE", 0.0)
    r2_val = metrics.get("R2", 0.0)

    annotation_lines = [
        f"Dataset: Test Set (N={len(y_true):,})",
        f"Test MAE:   {mae_val:.4f} kWh",
        f"Test RMSE:  {rmse_val:.4f} kWh",
        f"Test R²:    {r2_val:.4f}",
        f"Mean Bias:  {bias_val:+.4f} kWh",
    ]
    if note_text:
        annotation_lines.extend(["---", note_text])

    plt.text(
        0.04,
        0.96,
        "\n".join(annotation_lines),
        transform=plt.gca().transAxes,
        fontsize=8.5,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="#ffffff",
            edgecolor="#cccccc",
            alpha=0.92,
        ),
    )

    plt.legend(loc="lower right", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_residuals_plot(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: str,
    metrics: Dict[str, float],
    title_info: Optional[str] = None,
) -> None:
    """Generate and save dual-panel residual diagnostic plots."""
    residuals = np.array(y_true) - np.array(y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Residuals vs Predicted Values
    ax1.scatter(
        y_pred, residuals, alpha=0.35, edgecolors="none", s=16, color="#2ca02c"
    )
    ax1.axhline(
        0, color="#d62728", linestyle="--", linewidth=1.5, label="Zero Error Reference"
    )
    mean_res = float(np.mean(residuals))
    ax1.axhline(
        mean_res,
        color="#1f77b4",
        linestyle=":",
        linewidth=1.5,
        label=f"Mean Residual ({mean_res:+.4f} kWh)",
    )
    ax1.set_xlabel(
        "Predicted Energy Consumption (kWh)", fontsize=10.5, fontweight="bold"
    )
    ax1.set_ylabel(
        "Residual: Actual - Predicted (kWh)", fontsize=10.5, fontweight="bold"
    )
    ax1.set_title("Residuals vs. Fitted Values", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: Distribution of Residuals
    n, bins, patches = ax2.hist(
        residuals,
        bins=50,
        density=True,
        alpha=0.65,
        color="#1f77b4",
        edgecolor="#0b3d60",
        label="Empirical Residuals",
    )
    res_std = float(np.std(residuals))
    if res_std > 0:
        x_axis = np.linspace(float(np.min(residuals)), float(np.max(residuals)), 200)
        gaussian = stats.norm.pdf(x_axis, mean_res, res_std)
        ax2.plot(
            x_axis,
            gaussian,
            color="#d62728",
            linewidth=2.0,
            label=f"Normal Fit (μ={mean_res:+.3f}, σ={res_std:.3f})",
        )
    ax2.axvline(0, color="#333333", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.set_xlabel("Residual (kWh)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=10.5, fontweight="bold")
    ax2.set_title(
        "Residual Distribution (Error Normality)", fontsize=11, fontweight="bold"
    )
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    subtitle = (
        f"\nModel Checkpoint: {title_info}"
        if title_info
        else "\nModel Checkpoint: MLflow Registry smart-home-energy-advisor:1 (Run e9922ecbb3af4c029aebf2f19b755694)"
    )
    fig.suptitle(
        f"Residual Analysis — Held-Out Test Set (N={len(y_true):,}){subtitle}",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def log_evaluation_artifacts(
    client: MlflowClient,
    run_id: str,
    val_results_path: pathlib.Path,
    model_comparison_path: pathlib.Path,
) -> None:
    """Log existing evaluation report files to the run under 'evaluation/'."""
    if not val_results_path.exists():
        raise FileNotFoundError(f"Validation results not found at: {val_results_path}")
    if not model_comparison_path.exists():
        raise FileNotFoundError(f"Model comparison not found at: {model_comparison_path}")

    client.log_artifact(run_id, str(val_results_path), artifact_path="evaluation")
    client.log_artifact(run_id, str(model_comparison_path), artifact_path="evaluation")


def log_visualization_artifacts(
    client: MlflowClient,
    run_id: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: Optional[Dict[str, float]] = None,
    feature_names: Optional[List[str]] = None,
    title_info: Optional[str] = None,
    note_text: Optional[str] = None,
) -> Dict[str, str]:
    """Generate and log feature importance and test diagnostic plots under 'visualizations/'."""
    y_pred = model.predict(X_test)

    if metrics is None:
        metrics = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "RMSE": float(mean_squared_error(y_test, y_pred) ** 0.5),
            "R2": float(r2_score(y_test, y_pred)),
            "Bias": float(np.mean(y_pred - y_test)),
        }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)

        fi_path = str(tmp_path / "feature_importance.png")
        avp_path = str(tmp_path / "actual_vs_predicted.png")
        res_path = str(tmp_path / "residuals_plot.png")

        generate_feature_importance_plot(
            model,
            fi_path,
            feature_names=feature_names or list(X_test.columns),
            title_info=title_info,
        )
        generate_actual_vs_predicted_plot(
            y_test,
            y_pred,
            avp_path,
            metrics,
            title_info=title_info,
            note_text=note_text,
        )
        generate_residuals_plot(
            y_test, y_pred, res_path, metrics, title_info=title_info
        )

        client.log_artifact(run_id, fi_path, artifact_path="visualizations")
        client.log_artifact(run_id, avp_path, artifact_path="visualizations")
        client.log_artifact(run_id, res_path, artifact_path="visualizations")

    return {
        "feature_importance": "visualizations/feature_importance.png",
        "actual_vs_predicted": "visualizations/actual_vs_predicted.png",
        "residuals_plot": "visualizations/residuals_plot.png",
    }


def log_dataset_lineage_tags(
    client: MlflowClient, run_id: str, tags: Dict[str, str]
) -> None:
    """Log verified dataset lineage and evaluation metadata tags to an MLflow run."""
    for key, value in tags.items():
        client.set_tag(run_id, key, str(value))


def log_demonstration_artifacts(
    client: MlflowClient,
    run_id: str,
    model,
    train_path: pathlib.Path,
    val_path: pathlib.Path,
    test_path: pathlib.Path,
    val_results_path: pathlib.Path,
    model_comparison_path: pathlib.Path,
    dvc_file: Optional[pathlib.Path] = None,
    model_checkpoint_origin: str = "MLflow Registry smart-home-energy-advisor:1 (trained on data/train.csv)",
) -> Dict[str, float]:
    """Orchestrate end-to-end demonstration artifact and tag logging for a run."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    y_pred = model.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, y_pred))
    test_rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
    test_r2 = float(r2_score(y_test, y_pred))
    test_bias = float(np.mean(y_pred - y_test))

    metrics = {
        "MAE": test_mae,
        "RMSE": test_rmse,
        "R2": test_r2,
        "Bias": test_bias,
    }

    # 1. Log evaluation documents
    log_evaluation_artifacts(
        client, run_id, val_results_path, model_comparison_path
    )

    # 2. Log visualization artifacts
    note = (
        "Note: Phase 1 baseline checkpoint (models/gradient_boosting.pkl)\n"
        "trained on train+val achieved Test MAE=0.0739 kWh, R²=0.3723."
    )
    log_visualization_artifacts(
        client,
        run_id,
        model,
        X_test,
        y_test,
        metrics=metrics,
        feature_names=FEATURE_COLUMNS,
        title_info=f"MLflow Registry v1 (Run {run_id[:8]}...)",
        note_text=note,
    )

    # 3. Log dataset lineage tags
    train_md5, train_sha256 = compute_file_hashes(train_path)
    val_md5, _ = compute_file_hashes(val_path)
    test_md5, _ = compute_file_hashes(test_path)

    raw_dvc_md5 = "86f1e64db5d4387c6146e8063066b9fe"
    if dvc_file and dvc_file.exists():
        with open(dvc_file, "r") as f:
            for line in f:
                if "md5:" in line:
                    raw_dvc_md5 = line.split("md5:")[1].strip()
                    break

    tags = {
        "dataset.train.rows": str(len(train_df)),
        "dataset.train.md5": train_md5,
        "dataset.train.sha256": train_sha256,
        "dataset.validation.rows": str(len(val_df)),
        "dataset.validation.md5": val_md5,
        "dataset.test.rows": str(len(test_df)),
        "dataset.test.md5": test_md5,
        "dataset.split_strategy": "Chronological 70/15/15 time-series (no shuffle)",
        "dataset.raw_dvc_md5": raw_dvc_md5,
        "model.checkpoint_origin": model_checkpoint_origin,
        "evaluation.test_mae": f"{test_mae:.6f}",
        "evaluation.test_rmse": f"{test_rmse:.6f}",
        "evaluation.test_r2": f"{test_r2:.6f}",
        "evaluation.test_mean_bias": f"{test_bias:+.6f}",
    }
    log_dataset_lineage_tags(client, run_id, tags)

    return metrics


def main():
    """Main CLI entrypoint for local execution."""
    tracking_uri = DEFAULT_TRACKING_URI
    run_id = DEFAULT_RUN_ID
    model_uri = DEFAULT_MODEL_URI

    print(f"Connecting to MLflow Tracking Server at: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    run = client.get_run(run_id)
    print(f"Found existing run: {run.info.run_name} (ID: {run.info.run_id}, Status: {run.info.status})")

    print(f"\nLoading model from MLflow Registry: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    test_path = PROJECT_ROOT / "data" / "test.csv"
    train_path = PROJECT_ROOT / "data" / "train.csv"
    val_path = PROJECT_ROOT / "data" / "validation.csv"
    val_csv_path = PROJECT_ROOT / "reports" / "validation_results.csv"
    comp_md_path = PROJECT_ROOT / "reports" / "model_comparison.md"
    dvc_file = PROJECT_ROOT / "data" / "raw" / "SM Cleaned Data MH Aggregated.csv.dvc"

    metrics = log_demonstration_artifacts(
        client=client,
        run_id=run_id,
        model=model,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        val_results_path=val_csv_path,
        model_comparison_path=comp_md_path,
        dvc_file=dvc_file,
    )

    print(f"\nEvaluation on Held-Out Test Set:")
    print(f"  Test MAE:   {metrics['MAE']:.6f} kWh")
    print(f"  Test RMSE:  {metrics['RMSE']:.6f} kWh")
    print(f"  Test R²:    {metrics['R2']:.6f}")
    print(f"  Mean Bias:  {metrics['Bias']:+.6f} kWh")
    print("\nEnhancement complete!")


if __name__ == "__main__":
    main()
