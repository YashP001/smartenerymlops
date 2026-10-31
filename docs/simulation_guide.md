# Production Data Simulation Guide

## Overview

The **Production Data Simulator** (`src/simulation.py`, version `1.0.0`) generates synthetic smart-meter observations to test downstream MLOps monitoring pipelines (such as Evidently drift detection, performance tracking, and retraining triggers) under controlled operational conditions.

Simulated datasets reproduce the canonical model schema from `src/features.py` and historical distribution characteristics from `data/train.csv`, without copying rows or modifying baseline historical datasets.

---

## Architecture & Data Flow

```
                      +-----------------------------+
                      | Reference Data: train.csv   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Simulation Engine         |
                      |   (src/simulation.py v1.0.0)|
                      |   - Time series generator   |
                      |   - Hour/day matching       |
                      |   - Autoregressive chaining |
                      +--------------+--------------+
                                     |
       +-----------------------------+-----------------------------+
       |                             |                             |
       v                             v                             v
+--------------+              +--------------+              +--------------+
|   Scenario:  |              |   Scenario:  |              |   Scenario:  |
|    NORMAL    |              |  HVAC_SHIFT  |              | SENSOR_DRIFT |
+--------------+              +--------------+              +--------------+
       |                             |                             |
       +-----------------------------+-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Schema & Boundary Validator |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Isolated Output Directory   |
                      | data/simulated/*.csv        |
                      | data/simulated/*.json       |
                      +-----------------------------+
```

---

## Simulation Scenarios

### 1. Normal Operation (`normal`)
- **Objective:** Emulates typical operational data reflecting normal seasonal and diurnal consumption patterns without introducing intentional drift.
- **Methodology:**
  - Matches requested hours and days of the week to historical reference subsets from `data/train.csv`.
  - Samples real observations and injects subtle, zero-mean Gaussian jitter ($\sigma = 0.05 \times \sigma_{\text{ref}}$) to prevent identical row duplication.
  - Preserves natural temporal cross-correlations across lag features (`energy_current_hour`, `energy_previous_hour`, `energy_two_hours_ago`, `target_next_hour`).
- **Statistical Interpretation:**
  - Distribution consistency is evaluated jointly using standardized effect sizes (Cohen's $|d| < 0.15$) and low Kolmogorov-Smirnov distance statistics ($\text{KS} < 0.15$).
  - **Important Note on $p$-values:** A KS test $p$-value above $0.05$ does *not* prove distribution equivalence, and in finite samples across random seeds, $p$ may occasionally dip below $0.05$ (e.g., $0.015 - 0.895$). True operational equivalence is demonstrated by negligible effect size ($d \approx 0.01$) and aligned distribution moments.

### 2. HVAC Consumption Shift (`hvac_shift`)
- **Objective:** Simulates covariate shift caused by increased thermal cooling loads (e.g., intense air conditioning usage during summer heatwaves).
- **Methodology:**
  - Concentrates elevated consumption during afternoon and evening hours (12:00 to 20:00).
  - Injects realistic non-linear additive increments ($+0.12$ to $+0.27$ kWh modulated by severity) combined with multiplicative scaling ($1.0 + 0.45 \times \text{severity}$).
  - Smoothly propagates the shift across adjacent lag features and target hours to maintain realistic autoregressive inertia.
- **Statistical Interpretation:**
  - Statistically detectable distribution shift with medium-to-large effect size (Cohen's $d > 0.30$, observed mean $d \approx 0.48$), large KS distance ($\text{KS} > 0.20$), and $p < 0.001$.

### 3. Sensor Drift (`sensor_drift`)
- **Objective:** Simulates gradual sensor degradation, systematic hardware calibration bias, and measurement noise disturbances.
- **Telemetry Scope & Sensor Terminology:**
  - **Scope Limitation:** The simulator introduces disturbances into **derived active energy measurements** ($\text{kWh}$). It does **not** synthesize raw independent voltage ($V$) or current ($I$) telemetry channels, because the production model feature pipeline (`src/features.py`) ingests active hourly energy consumption and calendar variables.
  - Disturbances are modeled directly on the energy readings via positive calibration offset ($+0.05 \times \text{severity}$ kWh), analog measurement noise ($\sigma = 0.04 \times \text{severity}$ kWh), and intermittent measurement disturbance spikes.
  - Physical non-negativity and upper saturation limits ($0.0 \le \text{energy} \le 15.0$ kWh) are strictly enforced.
- **Statistical Interpretation:**
  - Statistically detectable positive bias with medium effect size (Cohen's $d > 0.25$, observed mean $d \approx 0.41$), KS distance $> 0.20$, and $p < 0.01$.
  - **Distinguishability:** Sensor drift shifts the entire 24-hour diurnal baseline uniformly (peak/off-peak ratio $\approx 0.83$, matching normal baseline $\approx 0.76$), whereas HVAC shift is concentrated strictly in hours 12:00–20:00 (peak/off-peak ratio $\approx 2.16$).

---

## Canonical Feature Schema

All generated datasets contain the exact six model input features and the actual target reading:

| Column Name | Type | Description | Range / Constraints |
|:---|:---|:---|:---|
| `energy_current_hour` | `float64` | Energy consumed in current hour ($t$) | $\ge 0.0$ kWh, finite |
| `energy_previous_hour` | `float64` | Energy consumed in previous hour ($t-1$) | $\ge 0.0$ kWh, finite |
| `energy_two_hours_ago` | `float64` | Energy consumed two hours ago ($t-2$) | $\ge 0.0$ kWh, finite |
| `target_hour` | `int64` | Hour of the day | $0 \le \text{hour} \le 23$ |
| `target_day_of_week` | `int64` | Day of week (0=Monday, 6=Sunday) | $0 \le \text{dow} \le 6$ |
| `target_is_weekend` | `int64` | Weekend indicator | $1$ if Saturday/Sunday, else $0$ |
| `target_next_hour` | `float64` | Actual energy consumed in next hour ($t+1$) | $\ge 0.0$ kWh, finite |

### Provenance & Audit Metadata

Every generated row includes explicit audit tags to prevent synthetic records from being confused with historical ground truth:

| Metadata Column | Type | Description | Example |
|:---|:---|:---|:---|
| `timestamp` | `datetime64[ns]` | Chronological hourly timestamp | `2026-11-01 00:00:00` |
| `meter` | `object` / `string` | Meter identifier | `MH01` or `SIM_MH01` |
| `simulation_scenario` | `object` / `string` | Scenario identifier | `normal`, `hvac_shift`, `sensor_drift` |
| `generation_timestamp` | `object` / `string` | UTC generation time (ISO-8601) | `2026-11-01T04:30:00Z` |
| `is_simulated` | `bool` | Explicit synthetic data flag | `True` |

### Companion JSON Manifest

Each CSV output is accompanied by a companion JSON metadata manifest containing complete provenance:

```json
{
  "simulator_version": "1.0.0",
  "reference_dataset": "data/train.csv",
  "simulation_scenario": "hvac_shift",
  "n_samples": 500,
  "random_state": 42,
  "severity": 1.0,
  "generation_timestamp": "2026-11-01T04:30:00Z",
  "columns": [...],
  "summary_statistics": {
    "energy_current_hour_mean": 0.2654,
    "energy_current_hour_std": 0.2741,
    "energy_current_hour_min": 0.0105,
    "energy_current_hour_max": 2.1402,
    "target_next_hour_mean": 0.2641
  }
}
```

- **`simulator_version`:** Explicit version of the simulator engine (`1.0.0`).
- **`reference_dataset`:** Path or identifier of the baseline dataset used for distribution conditioning.

---

## File Storage and Isolation

Generated outputs are strictly stored in `data/simulated/` and excluded from Git tracking:
- **Data File:** `data/simulated/simulated_<scenario>.csv`
- **Metadata Manifest:** `data/simulated/simulated_<scenario>_metadata.json`
- **Git Tracking:** Only `data/simulated/.gitkeep` is tracked in version control. All `.csv` and `.json` artifacts in this folder are ignored via `.gitignore`.

---

## CLI & Python API Usage

### Command Line Interface

```bash
# Generate 168 hours (1 week) of normal baseline data
.venv\Scripts\python.exe src/simulation.py --scenario normal --n-samples 168 --seed 42

# Generate HVAC consumption shift with high severity
.venv\Scripts\python.exe src/simulation.py --scenario hvac_shift --n-samples 168 --severity 1.5 --seed 101

# Generate sensor drift dataset
.venv\Scripts\python.exe src/simulation.py --scenario sensor_drift --n-samples 72 --severity 1.0 --seed 202
```

### Python API

```python
from src.simulation import generate_simulated_data, save_simulated_dataset

# 1. Generate DataFrame
df = generate_simulated_data(
    scenario="hvac_shift",
    n_samples=168,
    random_state=42,
    severity=1.2,
    meter_id="SIM_MH01",
)

# 2. Persist with provenance metadata
csv_path, json_path = save_simulated_dataset(
    df=df,
    scenario="hvac_shift",
    output_dir="data/simulated",
    random_state=42,
    severity=1.2,
    reference_dataset="data/train.csv",
)

print(f"Saved dataset: {csv_path}")
print(f"Saved metadata: {json_path}")
```

---

## Verification & Testing

Run the simulation test suite:

```bash
.venv\Scripts\python.exe -m pytest tests/test_simulation.py -v
```

The test suite validates:
1. **Scenario Coverage:** All 3 scenarios (`normal`, `hvac_shift`, `sensor_drift`) execute and validate.
2. **Schema Integrity:** Exact matching of 6 model features, target column, and metadata flags.
3. **Reproducibility:** Identical random seeds produce byte-identical DataFrames; distinct seeds produce divergent observations.
4. **Physical Boundaries:** All energy values are finite and non-negative ($\ge 0.0$).
5. **Statistical Verification:**
   - Normal operation exhibits negligible effect size (Cohen's $|d| < 0.15$) and low KS distance ($< 0.15$).
   - HVAC shift exhibits substantial effect size ($d > 0.30$) and significant KS drift ($p < 0.001$, $\text{KS} > 0.20$).
   - Sensor drift exhibits noticeable effect size ($d > 0.25$) and significant KS drift ($p < 0.01$, $\text{KS} > 0.20$).
   - Drift scenarios remain robustly distinguishable from normal operations across multiple random seeds.
6. **Provenance Tracking:** Manifest JSON contains `simulator_version` (`1.0.0`) and `reference_dataset` (`data/train.csv`).
