"""Reproduce registry v1, train v2, and compare both on one locked holdout."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.mlflow_config import SKOPS_TRUSTED_TYPES, configure_mlflow
from src.model_registry import DEFAULT_MODEL_NAME

TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
EVALUATION_PATH = PROJECT_ROOT / "data" / "test.csv"
EXPERIMENT_NAME = "smart-home-energy-advisor"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_code_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), PROJECT_ROOT / "src" / "features.py"):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def metrics(y_true, predictions) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(mean_squared_error(y_true, predictions) ** 0.5),
        "R2": float(r2_score(y_true, predictions)),
    }


def validate_dataset(frame: pd.DataFrame, label: str) -> None:
    missing = [name for name in FEATURE_COLUMNS + [TARGET_COLUMN] if name not in frame]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    if frame[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any().any():
        raise ValueError(f"{label} contains missing model inputs or targets")


def log_and_register_v1(client, train, lineage):
    """Keep a real existing v1 untouched; recreate its documented recipe if absent."""
    registered = client.search_model_versions(f"name = '{DEFAULT_MODEL_NAME}'")
    existing_v1 = next((item for item in registered if str(item.version) == "1"), None)
    if existing_v1 is not None:
        version = existing_v1
        model = mlflow.sklearn.load_model(f"models:/{DEFAULT_MODEL_NAME}/1")
        print(f"Retaining existing registry v1 (run {version.run_id}).")
        return model, version
    if registered:
        raise RuntimeError(
            f"Registry '{DEFAULT_MODEL_NAME}' has versions but no v1; "
            "refusing to create a duplicate or relabel an existing lifecycle."
        )

    # README records this checkpoint as GradientBoostingRegressor trained on
    # train.csv, with the sklearn defaults made explicit for reproducibility.
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    with mlflow.start_run(run_name="v1 - Gradient Boosting baseline") as run:
        mlflow.log_params(
            {
                "lifecycle_role": "v1",
                "model_type": "GradientBoostingRegressor",
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.1,
                "random_state": 42,
                "training_rows": len(train),
                "training_dataset_sha256": lineage["training_dataset_sha256"],
                "evaluation_dataset_sha256": lineage["evaluation_dataset_sha256"],
                "target": TARGET_COLUMN,
                "feature_count": len(FEATURE_COLUMNS),
            }
        )
        for key, value in lineage.items():
            mlflow.set_tag(key, str(value))
        sample = train[FEATURE_COLUMNS].head(5)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=infer_signature(sample, model.predict(sample)),
            input_example=sample,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        version = mlflow.register_model(f"runs:/{run.info.run_id}/model", DEFAULT_MODEL_NAME)
        client.set_model_version_tag(DEFAULT_MODEL_NAME, version.version, "lifecycle_role", "v1")
        print(f"Created reproducible v1 as registry version {version.version}.")
    return model, version


def main() -> None:
    tracking_uri = configure_mlflow()
    tracking_store = (
        "environment-configured MLFLOW_TRACKING_URI (value redacted)"
        if os.getenv("MLFLOW_TRACKING_URI")
        else "repository-local SQLite file (mlflow.db)"
    )
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()
    train = pd.read_csv(TRAIN_PATH)
    evaluation = pd.read_csv(EVALUATION_PATH)
    validate_dataset(train, "Training data")
    validate_dataset(evaluation, "Locked evaluation data")
    X_eval = evaluation[FEATURE_COLUMNS]
    y_eval = evaluation[TARGET_COLUMN]

    try:
        revision = code_revision()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", True
    lineage = {
        "code_revision": revision,
        "code_worktree_dirty": str(dirty).lower(),
        "training_code_sha256": training_code_sha256(),
        "training_dataset": "data/train.csv",
        "training_dataset_sha256": sha256(TRAIN_PATH),
        "evaluation_dataset": "data/test.csv (one-time final holdout comparison)",
        "evaluation_dataset_sha256": sha256(EVALUATION_PATH),
        "evaluation_protocol": "Same held-out test.csv for both models; no tuning on this set",
    }

    v1_model, v1 = log_and_register_v1(client, train, lineage)
    if not isinstance(v1_model, GradientBoostingRegressor):
        raise TypeError(
            "Registry v1 is not the documented GradientBoostingRegressor; "
            "preserve it and investigate the registry before comparison."
        )
    v1_results = metrics(y_eval, v1_model.predict(X_eval))
    # Capture this comparison's frozen holdout evidence on both the registry
    # record and its training run, without rewriting historical training lineage.
    v1_run = client.get_run(v1.run_id)
    v1_training_hash = (
        v1_run.data.params.get("training_dataset_sha256")
        or v1_run.data.tags.get("training_dataset_sha256")
        or v1_run.data.tags.get("train_dataset_sha256")
        or "unknown_historical_lineage"
    )
    v1_comparison = {
        "comparison_code_revision": revision,
        "comparison_evaluation_dataset": lineage["evaluation_dataset"],
        "comparison_evaluation_dataset_sha256": lineage["evaluation_dataset_sha256"],
        "comparison_protocol": lineage["evaluation_protocol"],
        "comparison_training_lineage_status": (
            "training_dataset_sha256_recorded" if v1_training_hash != "unknown_historical_lineage"
            else "historical_training_dataset_hash_unavailable"
        ),
        **{f"comparison_test_{k}": v for k, v in v1_results.items()},
    }
    for key, value in v1_comparison.items():
        client.set_model_version_tag(DEFAULT_MODEL_NAME, v1.version, key, str(value))
    client.set_model_version_tag(DEFAULT_MODEL_NAME, v1.version, "model_type", "GradientBoostingRegressor")
    for key, value in v1_comparison.items():
        client.set_tag(v1.run_id, key, str(value))
    for key, value in v1_results.items():
        client.log_metric(v1.run_id, f"comparison_test_{key}", value)
    candidate_spec = {
        "model_type": "RandomForestRegressor",
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
        "training_dataset_sha256": lineage["training_dataset_sha256"],
        "training_code_sha256": lineage["training_code_sha256"],
        "code_revision": revision,
    }
    candidate_fingerprint = hashlib.sha256(
        json.dumps(candidate_spec, sort_keys=True).encode("utf-8")
    ).hexdigest()
    existing_candidate = None
    for item in client.search_model_versions(f"name = '{DEFAULT_MODEL_NAME}'"):
        if item.tags.get("candidate_fingerprint") == candidate_fingerprint:
            existing_candidate = item
            break
        # Backward-compatible idempotency for a model registered by this
        # workflow before a reporting-only edit changed its full source hash.
        if item.tags.get("lifecycle_role") != "v2":
            continue
        run = client.get_run(item.run_id)
        params = run.data.params
        if (
            item.tags.get("training_dataset_sha256") == lineage["training_dataset_sha256"]
            and params.get("model_type") == "RandomForestRegressor"
            and params.get("n_estimators") == "100"
            and params.get("random_state") == "42"
            and params.get("n_jobs") == "-1"
            and params.get("training_dataset_sha256") == lineage["training_dataset_sha256"]
        ):
            existing_candidate = item
            break

    # v2 is an intentionally different, validation-informed tree ensemble:
    # Random Forest. The validation comparison is documented in
    # model_comparison.md. The test set is used only for final evaluation.
    if existing_candidate:
        v2 = existing_candidate
        client.set_model_version_tag(
            DEFAULT_MODEL_NAME, v2.version, "candidate_fingerprint", candidate_fingerprint
        )
        v2_model = mlflow.sklearn.load_model(
            f"models:/{DEFAULT_MODEL_NAME}/{v2.version}"
        )
        print(f"Reusing matching v2 registry version {v2.version} (run {v2.run_id}).")
    else:
        v2_model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        v2_model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])

    v2_results = metrics(y_eval, v2_model.predict(X_eval))
    delta = {key: v2_results[key] - v1_results[key] for key in v1_results}
    results = {
        "model_name": DEFAULT_MODEL_NAME,
        "tracking_store": tracking_store,
        "training_rows": len(train),
        "evaluation_rows": len(evaluation),
        "lineage": lineage,
        "versions": {
            "v1": {
                "version": str(v1.version), "run_id": v1.run_id,
                "training_dataset_sha256": v1_training_hash,
                **v1_results,
            },
            "v2": {},
        },
        "v2_minus_v1": delta,
        "decision": "Review the shared-holdout metrics; promotion is an explicit operator action.",
    }

    if not existing_candidate:
        with mlflow.start_run(run_name="v2 - Random Forest alternative") as run:
            mlflow.log_params(
                {
                    "lifecycle_role": "v2",
                    "model_type": "RandomForestRegressor",
                    "n_estimators": 100,
                    "random_state": 42,
                    "n_jobs": -1,
                    "training_rows": len(train),
                    "training_dataset_sha256": lineage["training_dataset_sha256"],
                    "evaluation_dataset_sha256": lineage["evaluation_dataset_sha256"],
                    "training_code_sha256": lineage["training_code_sha256"],
                    "target": TARGET_COLUMN,
                    "feature_count": len(FEATURE_COLUMNS),
                }
            )
            for key, value in lineage.items():
                mlflow.set_tag(key, str(value))
            mlflow.set_tag("candidate_fingerprint", candidate_fingerprint)
            mlflow.log_metrics({f"test_{k}": v for k, v in v2_results.items()})
            sample = train[FEATURE_COLUMNS].head(5)
            mlflow.sklearn.log_model(
                v2_model,
                artifact_path="model",
                signature=infer_signature(sample, v2_model.predict(sample)),
                input_example=sample,
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
            v2 = mlflow.register_model(f"runs:/{run.info.run_id}/model", DEFAULT_MODEL_NAME)
            client.set_model_version_tag(DEFAULT_MODEL_NAME, v2.version, "lifecycle_role", "v2")
            client.set_model_version_tag(DEFAULT_MODEL_NAME, v2.version, "candidate_fingerprint", candidate_fingerprint)
            client.set_model_version_tag(DEFAULT_MODEL_NAME, v2.version, "code_revision", revision)
            client.set_model_version_tag(
                DEFAULT_MODEL_NAME, v2.version, "training_dataset_sha256", lineage["training_dataset_sha256"]
            )
            client.set_model_version_tag(
                DEFAULT_MODEL_NAME, v2.version, "evaluation_dataset_sha256", lineage["evaluation_dataset_sha256"]
            )
    results["versions"]["v2"] = {
        "version": str(v2.version), "run_id": v2.run_id, **v2_results
    }
    v2_run = client.get_run(v2.run_id)
    results["versions"]["v2"].update(
        {
            "code_revision": v2_run.data.tags.get("code_revision", "unknown"),
            "training_code_sha256": (
                v2_run.data.params.get("training_code_sha256")
                or v2_run.data.tags.get("training_code_sha256", "unknown")
            ),
            "training_dataset_sha256": v2_run.data.params.get(
                "training_dataset_sha256", "unknown"
            ),
        }
    )
    for key, value in {
        "comparison_evaluation_dataset_sha256": lineage["evaluation_dataset_sha256"],
        "comparison_evaluation_protocol": lineage["evaluation_protocol"],
        **{f"comparison_test_{k}": v for k, v in v2_results.items()},
    }.items():
        client.set_model_version_tag(DEFAULT_MODEL_NAME, v2.version, key, str(value))

    output_csv = PROJECT_ROOT / "reports" / "model_comparison_v1_v2.csv"
    output_json = PROJECT_ROOT / "reports" / "model_comparison_v1_v2.json"
    output_md = PROJECT_ROOT / "reports" / "model_comparison_v1_v2.md"
    comparison = pd.DataFrame(
        [
            {"version": "v1", "registry_version": v1.version, "run_id": v1.run_id, **v1_results},
            {"version": "v2", "registry_version": v2.version, "run_id": v2.run_id, **v2_results},
        ]
    )
    comparison.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Registry v1 vs v2 comparison", "",
        "## Protocol", "",
        f"- Training data: `{lineage['training_dataset']}` ({len(train)} rows), SHA-256 `{lineage['training_dataset_sha256']}`.",
        f"- Shared final evaluation data: `{lineage['evaluation_dataset']}` ({len(evaluation)} rows), SHA-256 `{lineage['evaluation_dataset_sha256']}`.",
        "- The held-out evaluation set was not used to select hyperparameters or tune either model.",
        "- Both versions use the same six features and target. The validation-set model selection is documented in `model_comparison.md`.",
        "", "## Registered artifacts", "",
        f"- v1: registry version `{v1.version}`, run `{v1.run_id}`; retained unchanged if it already existed.",
        f"  Historical training data SHA-256: `{v1_training_hash}` (unknown means it was not recorded by the original run).",
        f"- v2: registry version `{v2.version}`, run `{v2.run_id}`; RandomForestRegressor (100 trees, seed 42).",
        "- v2 is a meaningful alternative architecture to GradientBoostingRegressor; it is not promoted automatically.",
        "", "## Shared evaluation metrics", "",
        "| Version | Model | MAE (kWh) | RMSE (kWh) | R² |",
        "|---|---|---:|---:|---:|",
        f"| v1 | GradientBoostingRegressor | {v1_results['MAE']:.6f} | {v1_results['RMSE']:.6f} | {v1_results['R2']:.6f} |",
        f"| v2 | RandomForestRegressor | {v2_results['MAE']:.6f} | {v2_results['RMSE']:.6f} | {v2_results['R2']:.6f} |",
        "", "## Decision", "",
        f"v2 minus v1: MAE {delta['MAE']:+.6f} kWh, RMSE {delta['RMSE']:+.6f} kWh, R² {delta['R2']:+.6f}. Lower MAE/RMSE and higher R² are preferable. Review this evidence before explicitly promoting a version.",
        "", "## Lineage", "",
        f"- v2 training run code revision: `{v2_run.data.tags.get('code_revision', 'unknown')}`; training code SHA-256 `{results['versions']['v2']['training_code_sha256']}`.",
        f"- MLflow tracking store: `{tracking_store}`.",
        "- v1 comparison results and the evaluation dataset identity are attached as comparison metadata; missing historical training lineage is reported as unknown.",
        "- Complete v2 training parameters, hashes, metrics, signature, and comparison JSON are logged with the v2 run.",
    ]
    output_md.write_text("\n".join(report) + "\n", encoding="utf-8")
    client.log_dict(v2.run_id, results, "evaluation/model_comparison.json")
    client.log_artifact(v2.run_id, str(output_csv), artifact_path="evaluation")
    client.log_artifact(v2.run_id, str(output_md), artifact_path="evaluation")
    print(comparison.to_string(index=False))
    print("\nSaved comparison report: reports/model_comparison_v1_v2.md")
    print("v2 is registered as a candidate and is not automatically deployed.")


if __name__ == "__main__":
    main()
