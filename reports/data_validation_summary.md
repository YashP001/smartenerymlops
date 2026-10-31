# Dataset Quality and Validation Report

**Generated:** 2026-10-31T20:39:23.226243  
**Dataset Rows:** 550,560  
**Dataset Columns:** 6  
**Overall Status:** `[WARN - ANOMALIES DETECTED]`  

---

## 1. Executive Summary

| Check Category | Status | Details |
| --- | :---: | --- |
| Schema & Column Types | [PASS] | All 6 required columns present |
| Missing Values | [PASS] | 0 missing / null / inf entries |
| Duplicate Records | [PASS] | 0 full row duplicates |
| Meter/Timestamp Uniqueness | [PASS] | 0 duplicate meter-timestamp pairs |
| Timestamp Validity | [PASS] | 0 unparseable timestamps |
| Timestamp Cadence (3-min) | [WARN] | 261 transmission gaps > 3 min |
| Physical Value Validity | [PASS] | 0 negative or physically impossible values |
| Domain Anomalies | [WARN] | 661 suspicious domain readings |

---

## 2. Confirmed Invalid Data (Critical Errors)

Confirmed invalid data are records that violate physical laws, break database schema, or disrupt model pipelines:

- **Schema Integrity:** 0 missing columns.
- **Missing Values:** 0 null / NaN / Inf values across the entire dataset.
- **Exact Duplicate Rows:** 0 duplicates found.
- **Meter/Timestamp Collisions:** 0 duplicate timestamp collisions.
- **Unparseable Timestamps:** 0 timestamps failed ISO datetime parsing.
- **Physically Impossible Values:** 0 negative values in energy, voltage, or current.

**Conclusion:** The dataset has **0 critical physical violations or schema errors**.

---

## 3. Suspicious Values and Domain Anomalies

Domain anomalies represent valid physical measurements under abnormal conditions (e.g. grid outages, surges) or sensor calibration edge-cases. These records are **preserved without silent modification**.

| Anomaly Type | Count | Threshold / Condition | Description |
| --- | :---: | --- | --- |
| `zero_voltage_with_consumption` | 45 | N/A | Readings where voltage is 0.0V but consumption t_kWh > 0 |
| `zero_frequency_with_consumption` | 45 | N/A | Readings where grid frequency is 0.0Hz but consumption t_kWh > 0 |
| `zero_current_with_consumption` | 215 | N/A | Readings where current is 0.0A but consumption t_kWh > 0 |
| `overvoltage_spikes` | 132 | > 300.0V | Readings with line voltage exceeding 300.0V |
| `brownout_undervoltage` | 212 | 0V < V < 150.0V | Readings with brownout voltage between 0V and 150.0V |
| `zero_voltage_outages` | 66,595 | N/A | Total readings with voltage = 0.0V (indicates power outages / offline) |
| `frequency_overfrequency` | 8 | > 55.0Hz | Readings with frequency exceeding 55.0Hz |
| `frequency_underfrequency` | 4 | 0Hz < F < 45.0Hz | Readings with non-zero frequency below 45.0Hz |
| `energy_consumption_spikes` | 0 | > 2.5 kWh / 3min | Readings where 3-minute energy exceeds 2.5 kWh |

### Key Findings:
- **Zero Voltage during Consumption:** 45 records reported `t_kWh > 0` while line voltage read `0.0V`. This represents a known smart-meter sensor logging lag when power resumes.
- **Overvoltage Surges:** 132 readings exceeded 300V (max recorded: 654.73V), consistent with rural grid surge events in Uttar Pradesh.
- **Grid Blackouts:** 66,595 readings (12.1%) show 0V line voltage, reflecting expected power supply outages.

---

## 4. Timestamp Interval and Gap Analysis

- **Nominal Interval:** 3 minutes
- **Total Intervals Evaluated:** 550,525
- **Exact 3-Minute Intervals:** 550,264 (99.95%)
- **Transmission Gaps (> 3 min):** 261
- **Estimated Missing Reading Intervals:** 254,400
- **Largest Recorded Gap:** 29 days 00:03:00

### Top 5 Largest Transmission Gaps:

| Meter | Timestamp Resumed | Gap Duration | Est. Missing Slots |
| --- | --- | --- | :---: |
| `MH41` | `2021-02-14 00:00:00` | 29 days 00:03:00 | 13,920 |
| `MH26` | `2021-02-09 00:00:00` | 9 days 00:03:00 | 4,320 |
| `MH20` | `2021-01-17 00:00:00` | 8 days 00:03:00 | 3,840 |
| `MH33` | `2021-01-24 00:00:00` | 8 days 00:03:00 | 3,840 |
| `MH34` | `2021-02-03 00:00:00` | 8 days 00:03:00 | 3,840 |

---

## 5. Statistical Summary of Numeric Features

| Metric | t_kWh (Consumption) | z_Avg Voltage (Volt) | z_Avg Current (Amp) | y_Freq (Hz) |
| --- | :---: | :---: | :---: | :---: |
| Count | 550,560 | 550,560 | 550,560 | 550,560 |
| Mean | 0.0069 | 223.7199 | 0.708 | 43.9553 |
| Std Dev | 0.0138 | 83.8543 | 1.1921 | 16.3054 |
| Min | 0.0 | 0.0 | 0.0 | 0.0 |
| 25% (Q1) | 0.001 | 242.65 | 0.09 | 49.94 |
| Median (50%) | 0.003 | 253.89 | 0.37 | 50.0 |
| 75% (Q3) | 0.008 | 261.57 | 0.89 | 50.04 |
| 95% | 0.018 | 271.45 | 2.07 | 50.08 |
| 99% | 0.085 | 276.74 | 7.0 | 50.13 |
| Max | 0.232 | 654.73 | 18.29 | 74.76 |
| Zero Count (%) | 20.79% | 12.1% | 14.45% | 12.1% |

### Meter Coverage Summary:
- **Unique Smart Meters:** 35
- **Readings per Meter (Min / Mean / Max):** 6,240 / 15,730.3 / 24,000

---

## 6. Recommendations for Downstream MLOps Modules

1. **Hourly Aggregation (Feature Engineering):**
   - Since transmission gaps exist (261 gaps > 3 min), hourly aggregation must require at least 15 valid readings per hour (out of 20 expected) before computing hourly sums, or apply linear interpolation for single missing 3-min slots.
2. **Outlier Handling:**
   - Clamp overvoltage readings (> 300V) or use robust scaling rather than deleting records, preventing data leakage.
3. **Zero Voltage Handling:**
   - Power outages (0V) represent true zero consumption periods and should not be imputed with non-zero averages.

---

## 7. Dataset Limitations and Known Gotchas

- **Transmission Gaps:** The dataset contains 261 non-contiguous intervals where smart meters lost cellular/RF connectivity. The longest gap spans 29 days for meter `MH41`.
- **Sensor Outage Lag:** In 45 instances, meters logged non-zero consumption (`t_kWh > 0`) while voltage registered `0.0V`. These occur at the boundary of grid restoration.
- **Single-Phase Voltage Surges:** Extreme voltages reaching up to 654.73V exist in the raw CEEW readings, reflecting grid transient spikes in rural distribution networks.
- **No Ground Truth Cleaning in Raw Data:** In accordance with MLOps best practices, raw datasets are preserved immutably under DVC without destructive filtering.

---

## 8. Reproduction Commands

To reproduce these data quality checks and verify the dataset independently:

```bash
# 1. Ensure DVC tracked dataset is retrieved
dvc pull data/raw/SM\ Cleaned\ Data\ MH2021.csv.dvc

# 2. Run the automated data validation module
python src/data_validation.py

# 3. Run the automated PyTest test suite (35 tests)
python -m pytest tests/test_data_validation.py -v

# 4. Verify DVC tracking status remains clean
dvc status
```
