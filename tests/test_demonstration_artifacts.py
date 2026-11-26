"""
Unit and integration tests for MLflow demonstration artifact logging.

Includes:
1. Portable tests: Test artifact and lineage tag logging using an isolated,
   temporary MLflow SQLite database and temporary artifact storage.
   These tests run cleanly in GitHub Actions on Ubuntu and any CI runner.
2. Local demonstration verification: An opt-in test suite verifying the actual
   local Gradient Boosting run (e9922ecbb3af4c029aebf2f19b755694), Model
   Registry version 1, and the complete 9-directory artifact hierarchy.
   Automatically skipped in CI when the local database is absent.
"""

import json
import os
import pathlib
import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Add project root to sys.path so scripts and src are reliably importable
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import mlflow
from mlflow.tracking import MlflowClient

from scripts.log_demonstration_artifacts import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    compute_file_hashes,
    generate_feature_importance_plot,
    generate_actual_vs_predicted_plot,
    generate_residuals_plot,
    build_dataset_artifacts,
    build_preprocessing_artifacts,
    build_configuration_artifacts,
    build_metadata_artifacts,
    build_deployment_artifacts,
    build_evaluation_artifacts,
    build_training_logs_artifacts,
    log_evaluation_artifacts,
    log_visualization_artifacts,
    log_dataset_lineage_tags,
    log_demonstration_artifacts,
    log_all_demonstration_artifacts,
)
from src.mlflow_config import PROJECT_ROOT as TRACKING_PROJECT_ROOT

# ---------------------------------------------------------------------------
# Dynamic Local Demonstration Detection
# ---------------------------------------------------------------------------
ENV_TRACKING_URI = os.getenv("LOCAL_MLFLOW_TRACKING_URI")
LOCAL_DB_CANDIDATE = TRACKING_PROJECT_ROOT / "mlflow.db"

if ENV_TRACKING_URI:
    LOCAL_TRACKING_URI = ENV_TRACKING_URI
    LOCAL_DEMO_AVAILABLE = True
elif LOCAL_DB_CANDIDATE.exists():
    LOCAL_TRACKING_URI = f"sqlite:///{LOCAL_DB_CANDIDATE.as_posix()}"
    LOCAL_DEMO_AVAILABLE = True
else:
    LOCAL_TRACKING_URI = None
    LOCAL_DEMO_AVAILABLE = False

LOCAL_RUN_ID = "e9922ecbb3af4c029aebf2f19b755694"
LOCAL_MODEL_NAME = "smart-home-energy-advisor"


# ---------------------------------------------------------------------------
# Portable Fixtures (Independent of Host Filesystem)
# ---------------------------------------------------------------------------
@pytest.fixture
def isolated_mlflow(tmp_path, monkeypatch):
    """Sets up an isolated, temporary SQLite MLflow tracking store for testing."""
    prev_tracking_uri = mlflow.get_tracking_uri()
    prev_registry_uri = mlflow.get_registry_uri()

    db_file = (tmp_path / "test_mlflow.db").as_posix()
    artifacts_dir = (tmp_path / "artifacts").as_posix()
    os.makedirs(artifacts_dir, exist_ok=True)
    tracking_uri = f"sqlite:///{db_file}"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment(
        "test-demonstration-exp", artifact_location=artifacts_dir
    )
    run = client.create_run(experiment_id, run_name="Test-Run")

    try:
        yield {
            "client": client,
            "tracking_uri": tracking_uri,
            "artifacts_dir": artifacts_dir,
            "run_id": run.info.run_id,
        }
    finally:
        client.set_terminated(run.info.run_id)
        if hasattr(client, "_tracking_client") and hasattr(
            client._tracking_client.store, "engine"
        ):
            client._tracking_client.store.engine.dispose()
        try:
            reg_client = client._get_registry_client()
            if hasattr(reg_client.store, "engine"):
                reg_client.store.engine.dispose()
        except Exception:
            pass

        mlflow.set_tracking_uri(prev_tracking_uri)
        mlflow.set_registry_uri(prev_registry_uri)


@pytest.fixture
def sample_model_and_data():
    """Provides a small deterministic synthetic dataset and fitted regressor."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.uniform(0.01, 1.0, size=(30, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )
    y = pd.Series(
        X["energy_current_hour"] * 0.75 + X["energy_previous_hour"] * 0.2 + 0.05,
        name=TARGET_COLUMN,
    )

    model = GradientBoostingRegressor(n_estimators=3, max_depth=2, random_state=42)
    model.fit(X, y)

    return {"X": X, "y": y, "model": model}


# ---------------------------------------------------------------------------
# Portable Unit & Integration Tests (Runs in GitHub Actions & Locally)
# ---------------------------------------------------------------------------
class TestPortableArtifactLogging:
    """Portable tests executed across all CI and local environments."""

    def test_compute_file_hashes(self, tmp_path):
        """Verify that MD5 and SHA256 checksums are computed accurately."""
        test_file = tmp_path / "sample.txt"
        test_file.write_text("mlops demonstration test payload\n")

        md5, sha256 = compute_file_hashes(test_file)
        assert len(md5) == 32
        assert len(sha256) == 64

    def test_log_evaluation_artifacts_portable(self, isolated_mlflow, tmp_path):
        """Verify evaluation documents are logged under evaluation/ directory."""
        client = isolated_mlflow["client"]
        run_id = isolated_mlflow["run_id"]

        val_csv = tmp_path / "validation_results.csv"
        comp_md = tmp_path / "model_comparison.md"
        val_csv.write_text("Model,MAE,RMSE,R2\nGradient Boosting,0.0712,0.1423,0.3397\n")
        comp_md.write_text("# Model Comparison\n\nGradient Boosting won.\n")

        log_evaluation_artifacts(client, run_id, val_csv, comp_md)

        artifacts = client.list_artifacts(run_id, path="evaluation")
        artifact_paths = {a.path for a in artifacts}
        assert "evaluation/validation_results.csv" in artifact_paths
        assert "evaluation/model_comparison.md" in artifact_paths

        for a in artifacts:
            assert a.file_size > 0

    def test_log_visualization_artifacts_portable(
        self, isolated_mlflow, sample_model_and_data
    ):
        """Verify diagnostic plots are created and logged under visualizations/ directory."""
        client = isolated_mlflow["client"]
        run_id = isolated_mlflow["run_id"]
        model = sample_model_and_data["model"]
        X = sample_model_and_data["X"]
        y = sample_model_and_data["y"]

        logged = log_visualization_artifacts(
            client=client,
            run_id=run_id,
            model=model,
            X_test=X,
            y_test=y,
            title_info="Test Checkpoint",
        )

        assert "feature_importance" in logged
        assert "actual_vs_predicted" in logged
        assert "residuals_plot" in logged

        artifacts = client.list_artifacts(run_id, path="visualizations")
        artifact_paths = {a.path for a in artifacts}
        assert "visualizations/feature_importance.png" in artifact_paths
        assert "visualizations/actual_vs_predicted.png" in artifact_paths
        assert "visualizations/residuals_plot.png" in artifact_paths

        for a in artifacts:
            assert a.file_size > 10_000

    def test_log_dataset_lineage_tags_portable(self, isolated_mlflow):
        """Verify dataset lineage metadata tags are recorded on the run."""
        client = isolated_mlflow["client"]
        run_id = isolated_mlflow["run_id"]

        tags = {
            "dataset.train.rows": "100",
            "dataset.train.md5": "abc123md5",
            "dataset.split_strategy": "Chronological 70/15/15",
            "evaluation.test_mae": "0.0712",
        }
        log_dataset_lineage_tags(client, run_id, tags)

        run = client.get_run(run_id)
        for key, val in tags.items():
            assert run.data.tags.get(key) == val

    def test_build_preprocessing_artifacts_portable(self, tmp_path):
        """Verify preprocessing artifacts generation and JSON/MD schemas."""
        out = build_preprocessing_artifacts(tmp_path)
        assert out["preprocessing_report"].exists()
        assert out["feature_definitions"].exists()

        defs = json.loads(out["feature_definitions"].read_text(encoding="utf-8"))
        assert "features" in defs
        assert len(defs["features"]) == 6
        assert defs["target"]["name"] == "target_next_hour"

    def test_build_deployment_artifacts_portable(self, tmp_path):
        """Verify deployment artifacts generation and OpenAPI JSON schema."""
        out = build_deployment_artifacts(tmp_path)
        assert out["inference_schema"].exists()
        assert out["deployment_config"].exists()

        schema = json.loads(out["inference_schema"].read_text(encoding="utf-8"))
        assert "POST /predict" in schema["endpoints"]
        assert "POST /feedback" in schema["endpoints"]

    def test_download_artifact_portable(self, isolated_mlflow, tmp_path):
        """Verify logged artifact can be retrieved and parsed via MLflow download API."""
        client = isolated_mlflow["client"]
        run_id = isolated_mlflow["run_id"]

        val_csv = tmp_path / "validation_results.csv"
        comp_md = tmp_path / "model_comparison.md"
        val_csv.write_text("Model,MAE\nGradient Boosting,0.0712\n")
        comp_md.write_text("# Summary\n")

        log_evaluation_artifacts(client, run_id, val_csv, comp_md)

        download_dest = tmp_path / "downloads"
        downloaded = client.download_artifacts(
            run_id, "evaluation/validation_results.csv", dst_path=str(download_dest)
        )
        assert os.path.exists(downloaded)
        content = pathlib.Path(downloaded).read_text()
        assert "Gradient Boosting" in content
        assert "0.0712" in content


# ---------------------------------------------------------------------------
# Local Demonstration Verification (Opt-In / Skipped when DB Unavailable)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def local_mlflow_client():
    """Returns client connected to the actual local demonstration database."""
    if not LOCAL_DEMO_AVAILABLE:
        pytest.skip("Local MLflow demonstration database not present")
    mlflow.set_tracking_uri(LOCAL_TRACKING_URI)
    return MlflowClient(tracking_uri=LOCAL_TRACKING_URI)


@pytest.mark.skipif(
    not LOCAL_DEMO_AVAILABLE,
    reason="Local MLflow demonstration database not present (local viva/demo verification only)",
)
class TestLocalDemonstrationVerification:
    """Verifies the actual local MLflow tracking database and artifacts for demonstration."""

    def test_local_existing_gradient_boosting_run_exists(self, local_mlflow_client):
        """Verify that the existing Gradient Boosting run exists and is FINISHED."""
        run = local_mlflow_client.get_run(LOCAL_RUN_ID)
        assert run is not None
        assert run.info.run_id == LOCAL_RUN_ID
        assert run.info.status == "FINISHED"
        assert run.data.params.get("model_type") == "Gradient Boosting"

    def test_local_registered_model_links_to_gradient_boosting_run(
        self, local_mlflow_client
    ):
        """Verify that smart-home-energy-advisor:1 links to run e9922ecbb3af4c029aebf2f19b755694."""
        mv = local_mlflow_client.get_model_version(LOCAL_MODEL_NAME, 1)
        assert mv is not None
        assert mv.run_id == LOCAL_RUN_ID
        assert mv.status == "READY"

    def test_local_all_nine_artifact_directories_exist(self, local_mlflow_client):
        """Verify that all 9 required artifact directories exist under the run."""
        artifacts = local_mlflow_client.list_artifacts(LOCAL_RUN_ID)
        dir_names = {a.path for a in artifacts if a.is_dir}
        expected_dirs = {
            "dataset",
            "preprocessing",
            "model",
            "configuration",
            "evaluation",
            "training_logs",
            "visualizations",
            "metadata",
            "deployment",
        }
        for ed in expected_dirs:
            assert ed in dir_names, f"Expected artifact directory '{ed}' missing from run"

    def test_local_dataset_artifacts_contents(self, local_mlflow_client):
        """Verify dataset_summary.json, data_split_summary.json, and data_lineage.md exist."""
        artifacts = local_mlflow_client.list_artifacts(LOCAL_RUN_ID, path="dataset")
        paths = {a.path.split("/")[-1] for a in artifacts}
        assert "dataset_summary.json" in paths
        assert "data_split_summary.json" in paths
        assert "data_lineage.md" in paths

    def test_local_preprocessing_artifacts_contents(self, local_mlflow_client):
        """Verify preprocessing_report.md and feature_definitions.json exist."""
        artifacts = local_mlflow_client.list_artifacts(LOCAL_RUN_ID, path="preprocessing")
        paths = {a.path.split("/")[-1] for a in artifacts}
        assert "preprocessing_report.md" in paths
        assert "feature_definitions.json" in paths

    def test_local_model_artifacts_contents(self, local_mlflow_client):
        """Verify MLmodel, model.pkl, requirements.txt, and input_example.json exist."""
        artifacts = local_mlflow_client.list_artifacts(LOCAL_RUN_ID, path="model")
        paths = {a.path.split("/")[-1] for a in artifacts}
        assert "MLmodel" in paths
        assert "model.pkl" in paths
        assert "requirements.txt" in paths
        assert "input_example.json" in paths

    def test_local_configuration_artifacts_contents(self, local_mlflow_client):
        """Verify model_config.json and training_config.json exist."""
        artifacts = local_mlflow_client.list_artifacts(LOCAL_RUN_ID, path="configuration")
        paths = {a.path.split("/")[-1] for a in artifacts}
        assert "model_config.json" in paths
        assert "training_config.json" in paths

    def test_local_evaluation_artifacts_exist(self, local_mlflow_client):
        """Verify validation and test evaluation files exist in evaluation/."""
        artifacts = local_mlflow_client.list_artifacts(LOCAL_RUN_ID, path="evaluation")
        paths = {a.path.split("/")[-1] for a in artifacts}
        assert "validation_results.csv" in paths
        assert "model_comparison.md" in paths
        assert "test_metrics.json" in paths
        assert "evaluation_report.md" in paths

    def test_local_training_logs_artifacts_contents(self, local_mlflow_client):
        """Verify training_summary.json exists in training_logs/."""
        artifacts = local_mlflow_client.list_artifacts(LOCAL_RUN_ID, path="training_logs")
        paths = {a.path.split("/")[-1] for a in artifacts}
        assert "training_summary.json" in paths

    def test_local_visualization_artifacts_exist(self, local_mlflow_client):
        """Verify feature_importance, actual_vs_predicted, and residuals exist in visualizations/."""
        artifacts = local_mlflow_client.list_artifacts(
            LOCAL_RUN_ID, path="visualizations"
        )
        paths = {a.path.split("/")[-1] for a in artifacts}
        assert "feature_importance.png" in paths
        assert "actual_vs_predicted.png" in paths
        assert "residuals_plot.png" in paths
        for a in artifacts:
            assert a.file_size > 10_000

    def test_local_metadata_artifacts_contents(self, local_mlflow_client):
        """Verify model_card.md and model_metadata.json exist in metadata/."""
        artifacts = local_mlflow_client.list_artifacts(LOCAL_RUN_ID, path="metadata")
        paths = {a.path.split("/")[-1] for a in artifacts}
        assert "model_card.md" in paths
        assert "model_metadata.json" in paths

    def test_local_deployment_artifacts_contents(self, local_mlflow_client):
        """Verify inference_schema.json and deployment_config.md exist in deployment/."""
        artifacts = local_mlflow_client.list_artifacts(LOCAL_RUN_ID, path="deployment")
        paths = {a.path.split("/")[-1] for a in artifacts}
        assert "inference_schema.json" in paths
        assert "deployment_config.md" in paths

    def test_local_recalculated_test_metrics_match_recorded(
        self, local_mlflow_client, tmp_path
    ):
        """Verify recalculated metrics on test.csv match test_metrics.json."""
        test_path = PROJECT_ROOT / "data" / "test.csv"
        assert test_path.exists(), "Held-out test dataset missing"
        test_df = pd.read_csv(test_path)
        assert len(test_df) == 4120

        model = mlflow.sklearn.load_model(f"models:/{LOCAL_MODEL_NAME}/1")
        preds = model.predict(test_df[FEATURE_COLUMNS])

        mae_calc = float(np.mean(np.abs(preds - test_df[TARGET_COLUMN])))
        bias_calc = float(np.mean(preds - test_df[TARGET_COLUMN]))

        dl = local_mlflow_client.download_artifacts(
            LOCAL_RUN_ID, "evaluation/test_metrics.json", dst_path=str(tmp_path)
        )
        recorded = json.loads(pathlib.Path(dl).read_text(encoding="utf-8"))["metrics"]

        assert abs(mae_calc - recorded["mae_kwh"]) < 1e-4
        assert abs(bias_calc - recorded["prediction_bias_kwh"]) < 1e-4
