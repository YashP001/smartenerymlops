# Phase 1 Report: AI-Assisted Smart Home Energy Advisor

## 1. Problem statement and objective

This project explores whether historical smart-meter readings can predict a household meter's energy consumption for the next hour. A reliable forecast could later help inform energy-saving recommendations. Electrical anomaly detection has been discussed as a possible extension, but it is not part of the forecasting feature implemented in Phase 1.

The Phase 1 objective is to establish a repeatable data and modeling workflow: inspect the data, prepare hourly examples, compare candidate models, evaluate on later time periods, and record model experiments.

## 2. Dataset

The project uses the CEEW Smart Meter Energy Consumption dataset for Mathura, 2021. The inspected raw dataset contains 550,560 rows, six columns, 35 meters, and nominal three-minute sampling intervals. The columns are `x_Timestamp`, `t_kWh`, `z_Avg Voltage (Volt)`, `z_Avg Current (Amp)`, `y_Freq (Hz)`, and `meter`.

The raw-data inspection found no missing values and no duplicate `(meter, timestamp)` pairs. Some voltage and frequency values are zero or unusual and require further investigation; the team has not claimed to correct them. The forecasting model does not use voltage, current, or frequency.

A separate verification compared daily sums of the raw `t_kWh` values with the supplied aggregated dataset. All 1,147 matching meter-day records agreed within 0.001 kWh. This supports summing readings over a larger interval, while the dataset documentation does not explicitly define `t_kWh` semantics and the comparison does not prove every interval is complete.

## 3. Data preparation and features

Three-minute readings were aggregated into hourly energy consumption by meter. An hour was retained when it had at least 20 distinct readings at the expected three-minute intervals. Incomplete or non-consecutive hourly sequences were excluded rather than imputed.

The model-ready data contains 26,901 examples across 35 meters. Its input features are `energy_current_hour`, `energy_previous_hour`, `energy_two_hours_ago`, `target_hour`, `target_day_of_week`, and `target_is_weekend`. The target, `target_next_hour`, is the next hour's energy consumption.

The chronological splits contain 18,696 training samples, 4,085 validation samples, and 4,120 test samples. After selecting the best model on validation data, Gradient Boosting was retrained on the combined training and validation data and evaluated on the test set.

## 4. Models and results

The comparison includes a persistence baseline, which predicts that next-hour energy will equal current-hour energy, plus Linear Regression, Random Forest, and Gradient Boosting.

| Model | Validation MAE (kWh) | Validation RMSE (kWh) | Validation R² |
|---|---:|---:|---:|
| Gradient Boosting | 0.0712 | 0.1423 | 0.3397 |
| Random Forest | 0.0734 | 0.1472 | 0.2942 |
| Persistence baseline | 0.0778 | 0.1827 | -0.0885 |
| Linear Regression | 0.0821 | 0.1554 | 0.2125 |

Gradient Boosting performed best on validation and was selected for final evaluation. On the chronological test set, after retraining on training plus validation data, it achieved MAE **0.0739 kWh**, RMSE **0.1512 kWh**, and R² **0.3723**. These results indicate moderate predictive performance, with room for improvement.

## 5. MLflow tracking

The training integration uses the MLflow experiment name `smart-home-energy-advisor`. It reads the training and validation CSV files, creates a persistence-baseline run and one run for each of the three sklearn models, and records validation MAE, RMSE, and R². The reported MLflow validation results match the checked-in `reports/validation_results.csv` values.

Each run records the model type, target name, and feature count. The sklearn runs also log primitive model parameters returned by `get_params()`. The configured Random Forest uses 100 trees, `random_state=42`, and `n_jobs=-1`; Gradient Boosting uses `random_state=42`; Linear Regression uses its sklearn defaults. The script logs a model artifact for each sklearn model. The persistence baseline is a direct copy of the current-hour feature and logs parameters and metrics, but no separate model artifact.

The script does not set a tracking URI explicitly; MLflow uses the environment's configured/default tracking location, and the script prints the resolved URI. Therefore, the storage location depends on how the experiment is run. The final test metrics are produced by the separate evaluation script after refitting on training plus validation data; the current MLflow script logs validation runs, not a final test-evaluation run.

### Input signature and schema warning review

The model logging call passes the first five training rows as an input example and now explicitly infers and logs a signature from those six feature columns and the fitted model predictions. This makes the expected input and output schema part of the artifact. The exact warning text was not available in the reviewed files, and the experiment was not rerun after this code change, so the warning resolution is not confirmed. Check the full warning output and verify that a loaded model accepts the expected column names and types.

MLflow improves development reproducibility by keeping parameters, validation metrics, and model artifacts together by run. It does not by itself mean that the model has been registered, served, deployed, or monitored in production.

## 6. Limitations and future improvements

Some voltage and frequency readings require investigation. The current forecasting model uses historical hourly energy and calendar features only. Excluding incomplete or non-consecutive hourly sequences reduces the number of usable examples and means forecasts are not produced for every possible hour in the raw series.

Three-minute sampling may miss brief electrical events. The proposed anomaly-detection extension could flag potential electrical anomalies or unusual consumption patterns, but these readings alone cannot establish an appliance failure. Anomaly detection is not implemented in this phase.

Possible next steps include investigating unusual readings with domain context, comparing additional forecasting features and methods, evaluating performance across meters and time periods, and checking behavior when readings are missing or irregular. The team should also review the actual MLflow warning output and verify the logged model input schema before using artifacts downstream.

## 7. Phase status

**Completed or reported as completed for Phase 1:** raw dataset inspection and verification; hourly aggregation and feature preparation; chronological splits; validation comparison of the persistence baseline and three models; final Gradient Boosting test evaluation; MLflow validation runs with parameters, metrics, and sklearn model artifacts.

**Proposed or not confirmed as completed:** electrical anomaly detection, MLflow model registration, FastAPI serving, Docker packaging, CI/CD, and production monitoring. These are future-phase possibilities, not completed Phase 1 components.
