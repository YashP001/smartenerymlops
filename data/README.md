# Dataset Notes

## Source

This project uses CEEW household smart-meter electricity data for Mathura and Bareilly, India. The local primary EDA file is:

- `dataverse_files/SM Cleaned Data MH2021.csv`

The source CSV is not modified, cleaned, imputed, or duplicated by the EDA notebook.

## Primary EDA Dataset

`SM Cleaned Data MH2021.csv` contains Mathura smart-meter readings for 2021 in the local workspace. The inspected CEEW ReadMe describes the smart-meter files as containing readings at every three-minute interval.

## Column Descriptions

| Column | Description from available documentation |
| --- | --- |
| `x_Timestamp` | Timestamp for the reading |
| `t_kWh` | Electricity consumption |
| `z_Avg Voltage (Volt)` | Average voltage |
| `z_Avg Current (Amp)` | Average current |
| `y_Freq (Hz)` | Frequency |
| `meter` | Smart meter identifier |

## Expected Sampling Interval

The available ReadMe describes the smart-meter measurements as recorded at every three-minute interval.

## Known Limitations

- Missing-value conventions are not documented in the inspected ReadMe.
- The EDA found per-meter timestamp gaps greater than the expected three-minute interval.
- The EDA found zero voltage and zero frequency readings, including a small number of rows where `t_kWh` is non-zero.
- Some voltage and frequency values are outside typical household electricity ranges and require domain validation before modeling.
- Large source CSV files should not be committed directly to ordinary Git. Use the team's agreed data-sharing workflow, such as the existing DVC setup if configured by the Git/DVC owner.
