# Dataset Verification: Daily Energy Aggregation

## Objective
To empirically verify whether summing the raw three-minute `t_kWh` readings produces daily energy totals consistent with the supplied aggregated dataset.

## Datasets
- Raw dataset: `SM Cleaned Data MH2021.csv`
- Aggregated dataset: `SM Cleaned Data MH Aggregated.csv`
- Raw columns used: `x_Timestamp`, `meter`, `t_kWh`
- Aggregated columns used: `meter`, `Date`, `t_kWh`

## Method
1. Extracted the calendar date from each raw timestamp.
2. Grouped raw readings by `meter` and date.
3. Summed `t_kWh` for each meter-day.
4. Joined the resulting daily totals with the aggregated dataset using `meter` and `Date`.
5. Calculated absolute differences for matching records.

## Results
- Matching meter-day records: 1,147
- Mean absolute difference: approximately 9.83 × 10⁻¹⁵ kWh
- Maximum absolute difference: approximately 3.38 × 10⁻¹⁴ kWh
- Records agreeing within 0.001 kWh: 1,147 of 1,147

## Conclusion and Limitations
The raw daily sums agree with the supplied aggregated daily values to floating-point precision for all 1,147 matching meter-day records. This empirical result supports summing `t_kWh` readings to calculate energy over a larger time interval.

The dataset README does not explicitly define the measurement semantics of `t_kWh`. Therefore, this comparison is empirical evidence rather than a direct confirmation from the documentation. The comparison covers only matching records and does not establish that every interval is complete.

Before hourly aggregation, missing readings, missing hours, and incomplete hours must be checked and handled explicitly.
