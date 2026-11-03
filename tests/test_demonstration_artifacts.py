"""
Unit and integration tests for MLflow demonstration artifact logging.

Includes:
1. Portable tests: Test artifact and lineage tag logging using an isolated,
   temporary MLflow SQLite database and temporary artifact storage.
   These tests run cleanly in GitHub Actions on Ubuntu and any CI runner.
2. Local demonstration verification: An opt-in test suite verifying the actual
   local Gradient Boosting run (e9922ecbb3af4c029aebf2f19b755694) and Model
   Registry version 1. Automatically skipped in CI when the local database is absent.
"""

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
    log_evaluation_artifacts,
    log_visualization_artifacts,
    log_dataset_lineage_tags,
    log_demonstration_artifacts,
)

# ---------------------------------------------------------------------------
# Dynamic Local Demonstration Detection
# ---------------------------------------------------------------------------
ENV_TRACKING_URI = os.getenv("LOCAL_MLFLOW_TRACKING_URI")
LOCAL_DB_CANDIDATE = pathlib.Path("E:/mlops%20phase%202/smart-home-energy-advisor/mlflow.db")

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
        # Dispose connection pools so SQLite files unlock cleanly on Windows
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

    def test_end_to_end_demonstration_logger_portable(
        self, isolated_mlflow, sample_model_and_data, tmp_path
    ):
        """Verify end-to-end demonstration logger orchestrator on isolated files."""
        client = isolated_mlflow["client"]
        run_id = isolated_mlflow["run_id"]
        model = sample_model_and_data["model"]
        X = sample_model_and_data["X"]
        y = sample_model_and_data["y"]

        # Prepare small synthetic CSV splits
        df = X.copy()
        df[TARGET_COLUMN] = y
        train_csv = tmp_path / "train.csv"
        val_csv = tmp_path / "val.csv"
        test_csv = tmp_path / "test.csv"
        rep_val_csv = tmp_path / "validation_results.csv"
        rep_comp_md = tmp_path / "model_comparison.md"

        df.to_csv(train_csv, index=False)
        df.to_csv(val_csv, index=False)
        df.to_csv(test_csv, index=False)
        rep_val_csv.write_text("Model,MAE\nGradient Boosting,0.0712\n")
        rep_comp_md.write_text("# Report\n")

        metrics = log_demonstration_artifacts(
            client=client,
            run_id=run_id,
            model=model,
            train_path=train_csv,
            val_path=val_csv,
            test_path=test_csv,
            val_results_path=rep_val_csv,
            model_comparison_path=rep_comp_md,
            model_checkpoint_origin="Isolated Test Checkpoint",
        )

        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "R2" in metrics

        # Verify all directories exist
        root_artifacts = {a.path for a in client.list_artifacts(run_id)}
        assert "evaluation" in root_artifacts
        assert "visualizations" in root_artifacts


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

    def test_local_evaluation_artifacts_exist(self, local_mlflow_client):
        """Verify validation_results.csv and model_comparison.md exist in evaluation/."""
        artifacts = local_mlflow_client.list_artifacts(LOCAL_RUN_ID, path="evaluation")
        paths = {a.path for a in artifacts}
        assert "evaluation/validation_results.csv" in paths
        assert "evaluation/model_comparison.md" in paths
        for a in artifacts:
            assert a.file_size > 0

    def test_local_visualization_artifacts_exist(self, local_mlflow_client):
        """Verify feature_importance, actual_vs_predicted, and residuals exist in visualizations/."""
        artifacts = local_mlflow_client.list_artifacts(
            LOCAL_RUN_ID, path="visualizations"
        )
        paths = {a.path for a in artifacts}
        assert "visualizations/feature_importance.png" in paths
        assert "visualizations/actual_vs_predicted.png" in paths
        assert "visualizations/residuals_plot.png" in paths
        for a in artifacts:
            assert a.file_size > 10_000

    def test_local_dataset_lineage_tags_present(self, local_mlflow_client):
        """Verify dataset lineage metadata tags are present and match verified checksums."""
        run = local_mlflow_client.get_run(LOCAL_RUN_ID)
        tags = run.data.tags
        assert tags.get("dataset.train.rows") == "18696"
        assert tags.get("dataset.train.md5") == "dd11d57bc366c3c5af9047d34cbe33af"
        assert tags.get("dataset.validation.rows") == "4085"
        assert tags.get("dataset.validation.md5") == "0b188dc415882a2cd20d6c14710daa17"
        assert tags.get("dataset.test.rows") == "4120"
        assert tags.get("dataset.test.md5") == "d9b92a33fe45dfc3459cb6356c004b2e"
        assert "Chronological" in tags.get("dataset.split_strategy", "")
        assert "Registry" in tags.get("model.checkpoint_origin", "")

    def test_local_download_artifact_via_api(self, local_mlflow_client, tmp_path):
        """Verify that an artifact can be downloaded from the local demonstration run."""
        downloaded = local_mlflow_client.download_artifacts(
            LOCAL_RUN_ID, "evaluation/validation_results.csv", dst_path=str(tmp_path)
        )
        assert os.path.exists(downloaded)
        content = pathlib.Path(downloaded).read_text()
        assert "Gradient Boosting" in content
        assert "MAE" in content
