# Registry v1 vs v2 comparison

## Protocol

- Training data: `data/train.csv` (18696 rows), SHA-256 `5ada97c4a287a235279522af2968e16706756f27bf915387a793534d7fbe73b2`.
- Shared final evaluation data: `data/test.csv (one-time final holdout comparison)` (4120 rows), SHA-256 `883121b6c8b78e49f0e01085f9bdae16dad86124efb9bfc2842b9b491d80a5b2`.
- The held-out evaluation set was not used to select hyperparameters or tune either model.
- Both versions use the same six features and target. The validation-set model selection is documented in `model_comparison.md`.

## Registered artifacts

- v1: registry version `1`, run `93b515d19d2a49aebd7edacac361d14a`; retained unchanged if it already existed.
  Historical training data SHA-256: `5ada97c4a287a235279522af2968e16706756f27bf915387a793534d7fbe73b2` (unknown means it was not recorded by the original run).
- v2: registry version `2`, run `395050cc42294a408e3102c47a6d25de`; RandomForestRegressor (100 trees, seed 42).
- v2 is a meaningful alternative architecture to GradientBoostingRegressor; it is not promoted automatically.

## Shared evaluation metrics

| Version | Model | MAE (kWh) | RMSE (kWh) | R² |
|---|---|---:|---:|---:|
| v1 | GradientBoostingRegressor | 0.073797 | 0.151036 | 0.373649 |
| v2 | RandomForestRegressor | 0.075547 | 0.152520 | 0.361275 |

## Decision

v2 minus v1: MAE +0.001750 kWh, RMSE +0.001485 kWh, R² -0.012374. Lower MAE/RMSE and higher R² are preferable. Review this evidence before explicitly promoting a version.

## Lineage

- v2 training run code revision: `42dc21fc38f710349a4a950388434565537c470d`; training code SHA-256 `230843d43ae537828a5ee44395532c4fdf2774c6db33d6868da1cb1867ed7a00`.
- MLflow tracking store: `environment-configured MLFLOW_TRACKING_URI (value redacted)`.
- v1 comparison results and the evaluation dataset identity are attached as comparison metadata; missing historical training lineage is reported as unknown.
- Complete v2 training parameters, hashes, metrics, signature, and comparison JSON are logged with the v2 run.
