# Model versioning and lifecycle demonstration

This guide covers only the model lifecycle workstream. It keeps the registered
v1 artifact, creates and compares a reproducible v2, and makes the serving
version and its MLflow run traceable from every new prediction.

## Tracking store

All project training, registry, and artifact scripts use `src/mlflow_config.py`.
By default it stores the SQLite database at the repository root as `mlflow.db`
and leaves MLflow artifact files in MLflow's local artifact directory. The URI
is made from a filesystem path directly, so a folder containing spaces is not
silently changed into a directory whose name contains the literal text `%20`.

Set `MLFLOW_TRACKING_URI` to use another store. `MLFLOW_REGISTRY_URI` may point
the registry at a separate MLflow store; when omitted, it uses the tracking
URI. For a path with spaces, provide the real path as a SQLite URI, for example
`sqlite:///C:/mlops project/mlflow.db` (do not pre-encode the space as `%20`).
Keep the database and artifact directory together when backing up a local
registry; the database alone does not contain model files.

## Reproduce and compare v1 and v2

From the repository root:

```powershell
python scripts/train_and_compare_model_versions.py
```

The workflow leaves a registered v1 version untouched when it exists. On a
fresh local registry, it recreates the documented v1 recipe: a
`GradientBoostingRegressor` with 100 estimators, depth 3, learning rate 0.1,
and seed 42, trained on `data/train.csv`. It then trains v2 as a
`RandomForestRegressor` with 100 trees and seed 42, using that same training
file. The model change is architectural and meaningful; it is not an
unexplained retuning of v1.

Both artifacts are evaluated once on the same held-out `data/test.csv`. That
test set is not used to tune either model. The existing
`reports/model_comparison.md` records the validation comparison that motivated
the Random Forest candidate. The new run records the final shared-holdout
metrics, dataset SHA-256 hashes, row counts, training parameters, feature and
target definitions, code revision, working-tree state, registry versions, and
run IDs. Outputs are written to `reports/model_comparison_v1_v2.{csv,json,md}`
and comparison JSON/Markdown are logged with v2's MLflow run.

The workflow registers v2 as a candidate and does not promote it automatically.
Compare MAE and RMSE (lower is better) and R² (higher is better), consider the
documented validation results, then make an explicit deployment decision.
It fingerprints the v2 recipe, training data, and training code, so rerunning
the same candidate reuses its registry version instead of adding duplicates.
For MLflow 3 `skops` serialization, the training workflow explicitly allows
the one scikit-learn tree storage type used by its own Gradient Boosting and
Random Forest models. It does not trust arbitrary reported types.

## Select, promote, inspect, and roll back

The deployment pointer is `models/deployment.json` (or the path in
`MODEL_DEPLOYMENT_FILE`). It records the selected candidate, deployed version,
and previous version. It is written atomically and ignored by Git so each
environment keeps its own deployment state. Selection validates that the
registered artifact can load but does not change serving traffic.

```powershell
python -m src.model_registry status
python -m src.model_registry select --version 2
python -m src.model_registry promote
python -m src.model_registry status
```

After promotion, the API sees the pointer change and loads the new registered
version on the next request. `/ready` reports `model_version` and
`model_run_id`; `/predict` returns both fields and retains `model` as the
estimator class name. The prediction row in SQLite stores `model_version` and
`model_run_id`, and `v_monitoring_records` exposes both for monitoring.

To roll back to the prior deployment:

```powershell
python -m src.model_registry rollback
python -m src.model_registry status
```

Rollback swaps the current and previous versions, preserving the newly active
version as the next rollback target. Promotion is local to the deployment
pointer: it does not delete, overwrite, or mutate the registered model
artifacts. If `MODEL_VERSION` is set in the API environment, it explicitly
overrides the pointer; remove that override when using the promotion/rollback
workflow.

## What to show

1. Show the comparison report and explain why v2 was trained and whether the
   evidence supports promoting it.
2. Run `status`, select the approved version, and promote it.
3. Request `/ready` and `/predict`; point to the registry version and MLflow
   run ID in the response and SQLite prediction record.
4. Roll back, then show `/ready` reporting the restored version and run ID.

The version is the immutable MLflow registry version number. The run ID points
to its training run and artifacts. Local fallback predictions are explicitly
marked `model_version: "local"` and have no MLflow run ID; they must not be
presented as registry-backed predictions.
