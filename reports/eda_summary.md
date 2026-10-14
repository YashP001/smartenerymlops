# Initial EDA Summary: `SM Cleaned Data MH2021.csv`

## Dataset Details

- Source file used: `E:\MLOPS PROJECT\dataverse_files\SM Cleaned Data MH2021.csv`
- Notebook uses a project-relative lookup for `data/SM Cleaned Data MH2021.csv` or `dataverse_files/SM Cleaned Data MH2021.csv`, with a local fallback path and a clear error if the file cannot be found.
- Original file was not modified, moved, or duplicated.
- Dataset shape: 550,560 rows x 6 columns
- Time range after parsing `x_Timestamp`: 2021-01-01 00:00:00 to 2021-02-20 23:57:00
- Unique meters: 35

## Columns and Data Types

| Column | Original dtype |
| --- | --- |
| `x_Timestamp` | object |
| `t_kWh` | float64 |
| `z_Avg Voltage (Volt)` | float64 |
| `z_Avg Current (Amp)` | float64 |
| `y_Freq (Hz)` | float64 |
| `meter` | object |

`x_Timestamp` was converted to datetime in the notebook with `errors="coerce"`. No invalid timestamps were found.

## Data Quality Checks

- Missing values: 0 in every original column
- Full duplicate rows: 0
- Duplicate `(meter, x_Timestamp)` rows: 0
- Duplicate `(meter, x_Timestamp)` pairs: 0
- Invalid/unparseable timestamps: 0
- Zero readings were retained for analysis and were not deleted or imputed.

## Numeric Descriptive Statistics

| Measurement | Count | Mean | Std | Min | 25% | Median | 75% | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `t_kWh` | 550,560 | 0.006852 | 0.013761 | 0.000000 | 0.001000 | 0.003000 | 0.008000 | 0.232000 |
| `z_Avg Voltage (Volt)` | 550,560 | 223.719934 | 83.854298 | 0.000000 | 242.650000 | 253.890000 | 261.570000 | 654.730000 |
| `z_Avg Current (Amp)` | 550,560 | 0.708009 | 1.192096 | 0.000000 | 0.090000 | 0.370000 | 0.890000 | 18.290000 |
| `y_Freq (Hz)` | 550,560 | 43.955283 | 16.305434 | 0.000000 | 49.940000 | 50.000000 | 50.040000 | 74.760000 |

## Zero Reading Percentages

| Numeric column | Zero reading percentage |
| --- | ---: |
| `t_kWh` | 20.7926% |
| `z_Avg Current (Amp)` | 14.4529% |
| `z_Avg Voltage (Volt)` | 12.0959% |
| `y_Freq (Hz)` | 12.0959% |

## Meter-Level Data Integrity Checks

### Timestamp Coverage and Gaps

The expected interval checked in the notebook is 3 minutes. Coverage was evaluated within each meter's own observed first and last timestamps, rather than assuming all meters should have identical source coverage.

- Meters checked: 35
- Record counts per meter range from 6,240 to 24,000.
- Earliest meter first timestamp: 2021-01-01 00:00:00
- Latest meter first timestamp: 2021-01-18 00:00:00
- Earliest meter last timestamp: 2021-02-16 23:57:00
- Latest meter last timestamp: 2021-02-20 23:57:00
- Meters with at least one gap greater than 3 minutes: 35
- Total gaps greater than 3 minutes: 261
- Largest gap: 29 days 00:03:00 for meter `MH41`

Top gap-count examples:

| Meter | Records | First timestamp | Last timestamp | Gaps > 3 min | Largest gap |
| --- | ---: | --- | --- | ---: | --- |
| `MH47` | 12,000 | 2021-01-01 00:00:00 | 2021-02-18 23:57:00 | 14 | 5 days 00:03:00 |
| `MH07` | 14,880 | 2021-01-01 00:00:00 | 2021-02-18 23:57:00 | 14 | 3 days 00:03:00 |
| `MH27` | 10,560 | 2021-01-06 00:00:00 | 2021-02-17 23:57:00 | 14 | 3 days 00:03:00 |
| `MH31` | 12,480 | 2021-01-05 00:00:00 | 2021-02-18 23:57:00 | 14 | 2 days 00:03:00 |
| `MH24` | 11,520 | 2021-01-03 00:00:00 | 2021-02-19 23:57:00 | 13 | 6 days 00:03:00 |

Record-count extremes:

| Meter | Records | First timestamp | Last timestamp | Gaps > 3 min | Largest gap |
| --- | ---: | --- | --- | ---: | --- |
| `MH35` | 6,240 | 2021-01-18 00:00:00 | 2021-02-17 23:57:00 | 7 | 5 days 00:03:00 |
| `MH39` | 6,240 | 2021-01-07 00:00:00 | 2021-02-19 23:57:00 | 9 | 7 days 00:03:00 |
| `MH33` | 6,720 | 2021-01-09 00:00:00 | 2021-02-16 23:57:00 | 10 | 8 days 00:03:00 |
| `MH23` | 24,000 | 2021-01-01 00:00:00 | 2021-02-20 23:57:00 | 1 | 1 days 00:03:00 |
| `MH25` | 24,000 | 2021-01-01 00:00:00 | 2021-02-20 23:57:00 | 1 | 1 days 00:03:00 |
| `MH43` | 24,000 | 2021-01-01 00:00:00 | 2021-02-20 23:57:00 | 1 | 1 days 00:03:00 |

### Consecutive Zero-Reading Runs

Longest `t_kWh` zero-consumption runs:

| Meter | Run length | Start timestamp | End timestamp | Duration minutes |
| --- | ---: | --- | --- | ---: |
| `MH28` | 457 | 2021-01-12 19:12:00 | 2021-01-13 18:00:00 | 1,371 |
| `MH24` | 237 | 2021-02-17 16:18:00 | 2021-02-19 04:06:00 | 711 |
| `MH33` | 236 | 2021-01-24 16:12:00 | 2021-01-25 03:57:00 | 708 |
| `MH18` | 230 | 2021-01-20 17:24:00 | 2021-01-22 04:51:00 | 690 |
| `MH07` | 225 | 2021-01-24 16:54:00 | 2021-01-26 04:06:00 | 675 |

Longest zero-voltage runs:

| Meter | Run length | Start timestamp | End timestamp | Duration minutes |
| --- | ---: | --- | --- | ---: |
| `MH24` | 237 | 2021-02-17 16:18:00 | 2021-02-19 04:06:00 | 711 |
| `MH33` | 236 | 2021-01-24 16:12:00 | 2021-01-25 03:57:00 | 708 |
| `MH18` | 230 | 2021-01-20 17:24:00 | 2021-01-22 04:51:00 | 690 |
| `MH07` | 225 | 2021-01-24 16:54:00 | 2021-01-26 04:06:00 | 675 |
| `MH01` | 221 | 2021-01-13 17:00:00 | 2021-01-14 04:00:00 | 663 |

Longest zero-frequency runs match the zero-voltage run leaders exactly for the top five listed above.

### Zero Voltage, Zero Frequency, and Zero Consumption Overlap

| Condition | Records | Percent of dataset | Zero-consumption records | Zero-consumption percent within condition |
| --- | ---: | ---: | ---: | ---: |
| zero voltage | 66,595 | 12.0959% | 66,550 | 99.9324% |
| zero frequency | 66,595 | 12.0959% | 66,550 | 99.9324% |
| zero voltage and zero frequency | 66,595 | 12.0959% | 66,550 | 99.9324% |
| zero voltage or zero frequency | 66,595 | 12.0959% | 66,550 | 99.9324% |

Confirmed: the zero-voltage and zero-frequency masks are identical in this dataset. There are 45 records where voltage and frequency are both zero but `t_kWh` is non-zero.

### Per-Meter Daily Consumption

Top five meters by total observed `t_kWh`:

| Meter | Observed days | Mean daily t_kWh | Median daily t_kWh | Min daily t_kWh | Max daily t_kWh | Total t_kWh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MH25` | 50 | 5.0997 | 4.9290 | 2.5050 | 8.0070 | 254.9870 |
| `MH10` | 48 | 4.7404 | 4.6565 | 1.3780 | 7.7060 | 227.5390 |
| `MH43` | 50 | 4.4278 | 4.2845 | 3.3770 | 5.8340 | 221.3880 |
| `MH36` | 47 | 4.6116 | 4.6670 | 3.5110 | 5.5070 | 216.7440 |
| `MH23` | 50 | 4.3278 | 4.2250 | 3.1860 | 8.2880 | 216.3890 |

Lowest five meters by total observed `t_kWh`:

| Meter | Observed days | Mean daily t_kWh | Median daily t_kWh | Min daily t_kWh | Max daily t_kWh | Total t_kWh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MH41` | 15 | 1.0180 | 0.8270 | 0.5560 | 1.5760 | 15.2700 |
| `MH20` | 21 | 0.7008 | 0.6800 | 0.3330 | 1.1490 | 14.7160 |
| `MH26` | 17 | 0.7839 | 0.7940 | 0.4200 | 1.1760 | 13.3260 |
| `MH47` | 25 | 0.4182 | 0.4310 | 0.1970 | 0.7110 | 10.4550 |
| `MH28` | 37 | 0.2525 | 0.2310 | 0.0600 | 0.6260 | 9.3420 |

The notebook adds a selected-meter daily consumption plot using high-, mid-, and low-total meters to compare daily patterns without overcrowding the chart.

## Visualization Observations

1. Distribution of household energy consumption (`t_kWh`):
   - The distribution is strongly right-skewed.
   - The median reading is 0.003 kWh, while the 99th percentile is 0.085 kWh and the maximum is 0.232 kWh.
   - 20.7926% of `t_kWh` readings are zero, so zero consumption periods are a major feature of the dataset and should be investigated rather than automatically removed.

2. Energy consumption over time:
   - The notebook plots daily summed `t_kWh` across meters for readability.
   - Daily totals cover 51 days.
   - Mean daily total is 73.969 kWh, median daily total is 74.791 kWh.
   - Lowest daily total is 26.712 kWh on 2021-02-20.
   - Highest daily total is 101.837 kWh on 2021-02-17.

3. Correlation heatmap:
   - `t_kWh` is highly correlated with `z_Avg Current (Amp)` at 0.9761.
   - `z_Avg Voltage (Volt)` and `y_Freq (Hz)` are highly correlated at 0.9897, influenced by their identical zero-reading masks.
   - `t_kWh` has weaker correlation with voltage at 0.1659 and frequency at 0.1847.

## Confirmed Data-Quality Findings

- No missing values, full duplicate rows, duplicate `(meter, x_Timestamp)` pairs, or invalid timestamps were found.
- Every meter has at least one timestamp gap greater than the expected 3-minute interval.
- Per-meter record counts and observed date coverage vary substantially.
- Zero voltage and zero frequency occur together for exactly the same 66,595 records.
- Most zero-voltage/zero-frequency records also have zero consumption, but 45 such records have non-zero `t_kWh`.
- Some voltage and frequency values are outside typical expected ranges for household electricity data, including voltage up to 654.73 V and frequency up to 74.76 Hz.

## Possible Explanations to Validate

- Multi-day timestamp gaps could reflect meter onboarding/offboarding, source extraction windows, outages, communication gaps, or intentional filtering in the cleaned dataset.
- Simultaneous zero voltage and zero frequency may represent power outages, meter outage states, communication behavior, or encoded unavailable readings.
- Non-zero `t_kWh` when voltage and frequency are both zero may indicate interval energy accumulation semantics, lagged measurement fields, rounding/timing effects, or data-quality issues.
- High voltage and frequency values may be true anomalies, sensor artifacts, scaling issues, or transient measurement errors. They need domain validation before modeling.

## Targeted Investigation Update

### Source Documentation

The inspected file `CEEW -  ReadMe.txt` describes the smart-meter files as containing electricity consumption (`t_kWh`), average voltage (`z_Avg Voltage (Volt)`), average current (`z_Avg Current (Amp)`), frequency (`y_Freq (Hz)`), meter ID (`meter`) and timestamp (`x_Timestamp`) at every three-minute interval.

Confirmed from documentation:

- `t_kWh`: electricity consumption
- `z_Avg Voltage (Volt)`: average voltage
- `z_Avg Current (Amp)`: average current
- `y_Freq (Hz)`: frequency
- `x_Timestamp`: three-minute interval timestamp
- Missing-value conventions: not documented in the inspected ReadMe

### Largest Timestamp Gaps

The largest same-meter gap is confirmed for `MH41`: the row before the gap is `2021-01-15 23:57:00`, and the next observed row is `2021-02-14 00:00:00`, giving a gap of `29 days 00:03:00`.

Top selected gap boundaries:

| Meter | Previous timestamp | Next timestamp | Gap | Next-row `t_kWh` | Next-row voltage | Next-row current | Next-row frequency |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `MH41` | 2021-01-15 23:57:00 | 2021-02-14 00:00:00 | 29 days 00:03:00 | 0.000 | 253.12 | 0.02 | 50.05 |
| `MH26` | 2021-01-30 23:57:00 | 2021-02-09 00:00:00 | 9 days 00:03:00 | 0.000 | 0.00 | 0.00 | 0.00 |
| `MH20` | 2021-01-08 23:57:00 | 2021-01-17 00:00:00 | 8 days 00:03:00 | 0.000 | 0.00 | 0.00 | 0.00 |
| `MH33` | 2021-01-15 23:57:00 | 2021-01-24 00:00:00 | 8 days 00:03:00 | 0.003 | 258.07 | 0.25 | 49.98 |
| `MH34` | 2021-01-25 23:57:00 | 2021-02-03 00:00:00 | 8 days 00:03:00 | 0.000 | 0.00 | 0.00 | 0.00 |

The notebook prints the exact row immediately before and after each selected large gap.

### Verification of Gap and Zero-Run Calculations

Timestamp gaps are calculated after sorting by `meter` and `x_Timestamp`; each gap is the current timestamp minus the previous timestamp for the same meter. The first row for each meter has no prior timestamp and is not counted as a gap.

Zero-run duration has two related interpretations:

- For `N` contiguous three-minute readings, elapsed time from first timestamp to last timestamp is `(N - 1) * 3` minutes.
- The covered reading window is `N * 3` minutes.

The earlier zero-run table counted consecutive zero rows in sorted observed records. The new investigation adds a stricter gap-aware calculation that breaks runs when the timestamp gap is not exactly three minutes.

Top gap-aware contiguous zero runs:

| Measurement | Meter | Run length | Start timestamp | End timestamp | First-to-last elapsed minutes | Covered interval minutes |
| --- | --- | ---: | --- | --- | ---: | ---: |
| `t_kWh` | `MH28` | 457 | 2021-01-12 19:12:00 | 2021-01-13 18:00:00 | 1,368 | 1,371 |
| `t_kWh` | `MH33` | 236 | 2021-01-24 16:12:00 | 2021-01-25 03:57:00 | 705 | 708 |
| `t_kWh` | `MH30` | 225 | 2021-02-16 19:24:00 | 2021-02-17 06:36:00 | 672 | 675 |
| `z_Avg Voltage (Volt)` | `MH33` | 236 | 2021-01-24 16:12:00 | 2021-01-25 03:57:00 | 705 | 708 |
| `z_Avg Voltage (Volt)` | `MH01` | 221 | 2021-01-13 17:00:00 | 2021-01-14 04:00:00 | 660 | 663 |
| `y_Freq (Hz)` | `MH33` | 236 | 2021-01-24 16:12:00 | 2021-01-25 03:57:00 | 705 | 708 |
| `y_Freq (Hz)` | `MH01` | 221 | 2021-01-13 17:00:00 | 2021-01-14 04:00:00 | 660 | 663 |

### Zero Voltage/Frequency with Non-Zero Consumption

The count remains 45 records where voltage and frequency are both zero but `t_kWh` is non-zero. These records occur across multiple meters; `MH36` has the highest count with 6 records.

First examples:

| Meter | Timestamp | `t_kWh` | Voltage | Current | Frequency |
| --- | --- | ---: | ---: | ---: | ---: |
| `MH01` | 2021-02-08 06:42:00 | 0.002 | 0.00 | 0.00 | 0.00 |
| `MH01` | 2021-02-17 20:27:00 | 0.001 | 0.00 | 0.00 | 0.00 |
| `MH03` | 2021-01-11 17:54:00 | 0.001 | 0.00 | 0.00 | 0.00 |
| `MH03` | 2021-01-29 09:54:00 | 0.003 | 0.00 | 0.00 | 0.00 |
| `MH06` | 2021-02-02 08:30:00 | 0.001 | 0.00 | 0.00 | 0.00 |

The notebook displays nearby readings within +/- 6 minutes for the first 10 such events. Several examples show normal voltage/current readings immediately before the zero-voltage/frequency row, followed by zero or normal readings afterward. This is a confirmed local pattern, not proof of cause.

### Largest Voltage and Frequency Values

Top voltage readings are concentrated in `MH37` in this dataset:

| Meter | Timestamp | `t_kWh` | Voltage | Current | Frequency |
| --- | --- | ---: | ---: | ---: | ---: |
| `MH37` | 2021-01-26 16:12:00 | 0.009 | 654.73 | 0.87 | 49.98 |
| `MH37` | 2021-01-26 16:33:00 | 0.005 | 654.39 | 0.48 | 50.02 |
| `MH37` | 2021-01-25 00:18:00 | 0.007 | 653.97 | 0.60 | 50.04 |
| `MH37` | 2021-01-26 16:36:00 | 0.006 | 652.35 | 0.49 | 50.01 |
| `MH37` | 2021-01-26 16:39:00 | 0.005 | 651.23 | 0.48 | 50.00 |

Top frequency readings occur across multiple meters:

| Meter | Timestamp | `t_kWh` | Voltage | Current | Frequency |
| --- | --- | ---: | ---: | ---: | ---: |
| `MH47` | 2021-02-07 19:18:00 | 0.000 | 1.69 | 0.00 | 74.76 |
| `MH33` | 2021-01-25 13:45:00 | 0.001 | 1.01 | 0.33 | 72.04 |
| `MH33` | 2021-01-27 16:54:00 | 0.003 | 224.90 | 0.19 | 70.63 |
| `MH46` | 2021-01-13 17:12:00 | 0.000 | 221.82 | 0.06 | 66.08 |
| `MH07` | 2021-01-23 03:30:00 | 0.004 | 272.47 | 0.48 | 64.08 |

The notebook displays surrounding records within +/- 6 minutes for the top voltage and frequency events.

### Confirmed Patterns vs Hypotheses

Confirmed:

- The ReadMe defines the fields and states a three-minute interval.
- The inspected ReadMe does not document missing-value conventions.
- The 29-day `MH41` timestamp gap is real in the file based on adjacent observed rows.
- The 45 zero-voltage/frequency but non-zero-consumption records are real in the file.
- The largest voltage values are concentrated in meter `MH37`.
- The largest frequency values occur across multiple meters and often appear near low-voltage, zero-voltage, or otherwise abrupt neighboring readings.

Hypotheses requiring validation:

- Large gaps may reflect meter coverage windows, communication gaps, outages, extraction windows, or cleaning/filtering in the source dataset.
- Zero voltage/frequency with non-zero `t_kWh` may reflect interval accumulation semantics, timing lag between fields, meter behavior, or data-quality artifacts.
- Extreme voltage/frequency values may be true electrical events, device artifacts, scaling issues, or transient measurement errors.

## Recommended Next Steps

- Confirm source documentation for meter coverage windows before treating missing 3-minute slots as data loss.
- Inspect exact large-gap intervals per meter and compare them with meter installation or data-collection metadata if available.
- Build meter-level outage or zero-run features, but do not delete or impute zero readings until their meaning is validated.
- Investigate the 45 records with zero voltage/frequency but non-zero `t_kWh`.
- Review high-voltage and high-frequency records by meter and timestamp.
- Add per-meter timestamp completeness charts and daily heatmaps before modeling.
- Use time-aware train/test splits only after timestamp continuity and meter coverage decisions are documented.

## Execution Status

- Notebook updated: `E:\MLOPS PROJECT\notebooks\01_eda.ipynb`
- Notebook execution: successful from start to finish
- Code cells executed: all cells
- Error outputs: 0
- Summary report updated: `E:\MLOPS PROJECT\reports\eda_summary.md`
- Original CSV unchanged
