# AI-Assisted Smart Home Energy Advisor

An end-to-end Machine Learning Operations (MLOps) project for household energy forecasting. The project develops, tracks, and evaluates machine learning models to predict a smart meter's electricity consumption for the next hour based on historical readings and calendar features.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Details](#dataset-details)
- [System Architecture & Pipeline](#system-architecture--pipeline)
- [Model Performance & Evaluation](#model-performance--evaluation)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Repository Structure](#repository-structure)
- [Branching & Collaboration Strategy](#branching--collaboration-strategy)
- [Getting Started & Reproducibility](#getting-started--reproducibility)
- [Team Roles & Contributions](#team-roles--contributions)
- [Roadmap & Next Steps](#roadmap--next-steps)

---

## Project Overview

The primary objective of this project is to explore whether historical smart-meter readings can reliably forecast a household meter's energy consumption ($t_{\text{kWh}}$) for the next hour. Accurate hour-ahead energy prediction empowers consumers with actionable energy-saving insights and grid operators with demand-forecasting data.

### Phase 1 Focus: Development and Reproducibility
- **Empirical Data Verification:** Confirm consistency between raw multi-minute readings and daily aggregations.
- **Data Engineering:** Aggregate 3-minute readings into strict hourly consumption windows and formulate lag/calendar features.
- **Baseline & ML Models:** Benchmark Persistence, Linear Regression, Random Forest, and Gradient Boosting Regressors using chronological train/validation/test splits.
- **Experiment Tracking:** Track model parameters, validation metrics (MAE, RMSE, $R^2$), and model artifacts with MLflow (including explicit signatures).
- **Version Control & Data Management:** Track source code via Git/GitHub feature branches and initialize data management via DVC.

---

## Dataset Details

The project utilizes the **CEEW (Council on Energy, Environment and Water) Smart Meter Energy Consumption Dataset** for Mathura, India (2021).

- **Raw Dimensions:** 550,560 rows, 6 columns across 35 unique smart meters.
- **Nominal Sampling Interval:** Every 3 minutes.
- **Feature Attributes:**
  - `x_Timestamp`: Reading timestamp
  - `t_kWh`: Electricity consumption for the sampling interval
  - `z_Avg Voltage (Volt)`: Average voltage
  - `z_Avg Current (Amp)`: Average current
  - `y_Freq (Hz)`: Frequency
  - `meter`: Smart meter identifier
- **Data Quality & Verification (`reports/data_verification.md`):**
  - Zero missing values and zero duplicate `(meter, timestamp)` pairs in the inspected raw dataset.
  - Aggregated 3-minute sums were matched against the official CEEW daily aggregated dataset (`SM Cleaned Data MH Aggregated.csv`): all 1,147 matching meter-days agreed within floating-point precision ($< 0.001\text{ kWh}$ error, mean difference $\approx 9.83 \times 10^{-15}\text{ kWh}$), proving additive consistency of `t_kWh`.
  - Voltage/frequency outliers and zeros were analyzed; the current energy forecasting pipeline strictly isolates consumption features (`t_kWh` and temporal features) without relying on electrical voltage/current/frequency columns.

---

## System Architecture & Pipeline

The pipeline processes raw smart-meter time series into cleaned, model-ready chronological datasets:

```
Raw Readings (3-min intervals)
          │
          ▼
[ src/build_hourly_data.py ]
   - Validate exactly 20 readings/hr at {00, 03, ..., 57} min
   - Sum t_kWh into hourly consumption (hourly_energy.csv)
          │
          ▼
[ src/prepare_features.py ]
   - Verify consecutive hour sequences
   - Construct lag features (t, t-1, t-2)
   - Add calendar features (hour, day of week, is_weekend)
   - Formulate target (t+1) -> model_dataset.csv (26,901 rows)
          │
          ▼
[ src/split_data.py ]
   - Chronological split (70% Train, 15% Validation, 15% Test)
   - Train: 18,696 rows | Validation: 4,085 rows | Test: 4,120 rows
          │
          ▼
[ src/train_with_mlflow.py ] & [ src/evaluate_model.py ]
   - Train candidates, log metrics & signatures to MLflow
   - Refit best model on Train + Validation, evaluate on Test
```

### Feature Set
| Feature Name | Description |
|---|---|
| `energy_current_hour` | Energy consumption in the current hour ($t$) |
| `energy_previous_hour` | Energy consumption in the previous hour ($t-1$) |
| `energy_two_hours_ago` | Energy consumption two hours ago ($t-2$) |
| `target_hour` | Hour of the day for the prediction target ($0 - 23$) |
| `target_day_of_week` | Day of week for target ($0 = \text{Monday}, 6 = \text{Sunday}$) |
| `target_is_weekend` | Binary flag ($1$ if Saturday/Sunday, $0$ otherwise) |
| **`target_next_hour`** | **Target variable**: Energy consumption in hour $t+1$ |

---

## Model Performance & Evaluation

Models were evaluated across chronological splits using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$ score.

### Validation Set Comparison (`reports/validation_results.csv`)

| Model | Validation MAE (kWh) | Validation RMSE (kWh) | Validation $R^2$ |
|---|---:|---:|---:|
| **Gradient Boosting** | **0.0712** | **0.1423** | **0.3397** |
| Random Forest | 0.0734 | 0.1472 | 0.2942 |
| Persistence Baseline | 0.0778 | 0.1827 | -0.0885 |
| Linear Regression | 0.0821 | 0.1554 | 0.2125 |

### Final Evaluation on Chronological Test Set

Following model selection on validation data, the winning **Gradient Boosting Regressor** was refitted on combined training + validation data (`development`) and evaluated on the unseen chronological test set (`reports/test_predictions.csv`):

- **Test Samples:** 4,120 rows
- **Test MAE:** **0.0739 kWh**
- **Test RMSE:** **0.1512 kWh**
- **Test $R^2$:** **0.3723**

---

## MLflow & Weights & Biases Experiment Tracking

The project uses MLflow for model registry and tracking, and Weights & Biases (W&B) for interactive dashboards:
- **MLflow Experiment Name:** `smart-home-energy-advisor` (stored in persistent SQLite DB `mlflow.db`)
- **Logged Parameters:** Model type, feature count, target column, and scikit-learn hyperparameters via `get_params()`.
- **Logged Metrics:** Validation MAE, RMSE, and $R^2$.
- **Model Signatures & Artifacts:** Logged with explicit input/output schemas (`infer_signature`) and cloudpickle serialization.
- **Model Registry:** Best run (`GradientBoostingRegressor`) registered under `smart-home-energy-advisor` (Version 1) with idempotency protection.

### Viewing MLflow Locally
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Navigate to `http://127.0.0.1:5000` to inspect experiment runs, comparative metric charts, registered models, and logged model artifacts.

### Synchronizing Weights & Biases (W&B) Dashboard
When running without an active W&B login, experiments are recorded in offline mode (`wandb/offline-run-...`). To sync your runs to the cloud:
1. Log in to [https://wandb.ai/authorize](https://wandb.ai/authorize) to get your API key.
2. Authenticate in your terminal: `wandb login` (or set `$env:WANDB_API_KEY="..."`).
3. Push offline runs to your W&B project:
   ```bash
   wandb sync wandb/offline-run-20261031_230321-rugud7li
   ```
4. Access the comparative run dashboards and `model_comparison` tables on your W&B dashboard.

---

## Docker Containerization (Member 4)

The FastAPI inference service is packaged into a self-contained, production-ready Docker image based on `python:3.13-slim`.

For complete instructions, architecture explanations, and fallback behaviors, see the dedicated [Docker Containerization Guide](file:///e:/mlops%20phase%202/smart-home-energy-advisor/docs/docker_guide.md).

### Quickstart Commands (PowerShell)

```powershell
# 1. Build the Docker image:
docker build -t smart-home-energy-advisor .

# 2. Run the container:
docker run --rm -p 8000:8000 --name smart-home-advisor-app smart-home-energy-advisor

# 3. Check health status:
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get

# 4. Predict next-hour consumption:
$body = @{
    energy_current_hour = 0.05
    energy_previous_hour = 0.04
    energy_two_hours_ago = 0.06
    target_hour = 14
    target_day_of_week = 2
    target_is_weekend = 0
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body
```

---

## Repository Structure

```
├── .dockerignore           # Excludes local caches, datasets, and secrets from Docker
├── .dvc/                   # DVC configuration and tracking metadata
├── .gitignore              # Git ignore rules for data, models, and caches
├── Dockerfile              # Container definition for FastAPI inference service
├── requirements.txt        # Full development & training dependencies
├── requirements-inference.txt # Dedicated lightweight runtime inference dependencies
├── README.md               # Project documentation and workflow guide
├── api/
│   └── main.py             # FastAPI service with lifespan and /health, /predict endpoints
├── docs/
│   └── docker_guide.md     # Detailed containerization and deployment guide
├── data/
│   ├── raw/                # Raw smart-meter CSV files (DVC tracked)
│   └── processed/          # Cleaned and processed model datasets
├── models/                 # Saved model serialization artifacts (.pkl/.joblib)
├── notebooks/              # Exploratory Data Analysis notebooks
├── reports/                # Validation results, metrics, and markdown reports
├── src/                    # Feature pipeline, training, registry, and prediction modules
└── tests/                  # Pytest automated test suites (108 tests passing)
```

---

## Branching & Collaboration Strategy

The repository follows a collaborative branch-per-feature workflow:

- **`main`**: Production-ready branch containing merged, verified pipelines, reports, and tracking scripts.
- **`member1-eda`**: Dedicated to exploratory data analysis, dataset profiling, and EDA reporting (`notebooks/01_eda.ipynb`, `reports/eda_summary.md`, `data/README.md`).
- **`member3-ml-pipeline`**: Established the end-to-end data processing, feature engineering, and model training modules.
- **`member4-mlflow-report`**: Integrated MLflow experiment tracking, explicit model signatures, validation logging, and authored the comprehensive `reports/phase1_report.md`. (Merged into `main` via PR #2).

---

## Getting Started & Reproducibility

### 1. Prerequisites
- Python 3.9+ installed
- Git configured

### 2. Setup Environment
```bash
# Clone repository
git clone https://github.com/simran-s-patil/smart-home-energy-advisor.git
cd smart-home-energy-advisor

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Pipeline & Verification
Place raw CEEW smart-meter data (`SM Cleaned Data MH2021.csv`) into `data/`.
```bash
# 1. Verify daily aggregations against reference dataset
python src/verify_daily_totals.py

# 2. Build verified hourly consumption dataset
python src/build_hourly_data.py

# 3. Generate lag features and target
python src/prepare_features.py

# 4. Generate chronological data splits
python src/split_data.py
```

### 4. Model Training & MLflow Tracking
```bash
# Run MLflow-tracked experiments
python src/train_with_mlflow.py

# Evaluate best model on test split
python src/evaluate_model.py

# Launch MLflow UI
mlflow ui
```

---

## Team Roles & Contributions

This project is collaboratively developed by a four-member team:

- **Member 1 (EDA & Data Quality - `member1-eda`):** Exploratory data analysis, statistical distributions, handling null/zero readings, and preliminary reporting.
- **Member 2 (Project Setup & Data Versioning - `main`):** Git repository structure, environment configuration, dependency management, and DVC initialization.
- **Member 3 (ML Pipeline & Feature Engineering - `member3-ml-pipeline`):** Hourly window aggregations, lag feature generation, chronological data splitting, and model benchmarking.
- **Member 4 (MLflow Tracking & Reporting - `member4-mlflow-report`):** MLflow experiment tracking, model signatures/artifacts logging, Phase 1 documentation, and test-set verification.

---

## Roadmap & Next Steps

- [ ] Incorporate weather features (temperature, humidity) and outdoor ambient conditions.
- [ ] Investigate outlier voltage/frequency intervals for potential electrical anomaly detection.
- [ ] Implement MLflow Model Registry and staging transitions.
- [ ] Package model inference into a RESTful API using FastAPI.
- [ ] Containerize application with Docker and establish automated CI/CD testing pipelines.