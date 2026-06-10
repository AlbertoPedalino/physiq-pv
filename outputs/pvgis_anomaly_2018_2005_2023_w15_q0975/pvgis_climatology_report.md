# PVGIS climatology anomaly report

Target PVGIS year compared against a multi-year PVGIS climatology. PVGIS is used as a consistent physical reference. This analysis does **not** inspect real plants, does **not** use observed plant energy, and does **not** touch the training pipeline.

## Parameters

- Target year: **2018**
- Climatology years used: 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2019, 2020, 2021, 2022, 2023
- Quantile band: **[0.025, 0.975]** (quantile=0.975)
- Min climatology years per bin: **3**
- Climatology window: **±15 days**
- Min score denominator: **1.0**
- Locations: **1149**  |  Target timesteps: **8760**
- Variables analyzed: solar_irradiance_poa, pv_power_output, temperature_2m, wind_speed_10m
- Total flagged anomalies: **984578**
- Generated (UTC): 2026-06-08T10:32:24+00:00

## Per-variable summary

| variable | valid | normal | insufficient | flagged | high | low | % flagged | max |score| |
|---|---|---|---|---|---|---|---|---|
| solar_irradiance_poa | 10065240 | 9949793 | 0 | 115447 | 4784 | 110663 | 1.147% | 2.59 |
| pv_power_output | 10065240 | 9938049 | 0 | 127191 | 8383 | 118808 | 1.264% | 2.48 |
| temperature_2m | 10065240 | 9746512 | 0 | 318728 | 183823 | 134905 | 3.167% | 2.47 |
| wind_speed_10m | 10065240 | 9642028 | 0 | 423212 | 213323 | 209889 | 4.205% | 9.02 |

## Top 20 most extreme conditions

| timestamp | location | variable | value | median | band | score | label |
|---|---|---|---|---|---|---|---|
| 2018-10-30 01:10:00 | 574 | wind_speed_10m | 10.3 | 1.24 | [0.475, 2.48] | 9.02 | extreme_wind_condition |
| 2018-10-30 02:10:00 | 574 | wind_speed_10m | 10.3 | 1.24 | [0.41, 2.42] | 9.02 | extreme_wind_condition |
| 2018-10-30 02:10:00 | 537 | wind_speed_10m | 10.3 | 1.24 | [0.41, 2.42] | 9.02 | extreme_wind_condition |
| 2018-10-30 02:10:00 | 536 | wind_speed_10m | 10.3 | 1.24 | [0.41, 2.42] | 9.02 | extreme_wind_condition |
| 2018-10-30 01:10:00 | 536 | wind_speed_10m | 10.3 | 1.24 | [0.475, 2.48] | 9.02 | extreme_wind_condition |
| 2018-10-30 01:10:00 | 537 | wind_speed_10m | 10.3 | 1.24 | [0.475, 2.48] | 9.02 | extreme_wind_condition |
| 2018-10-30 01:10:00 | 573 | wind_speed_10m | 10.3 | 1.24 | [0.475, 2.48] | 9.02 | extreme_wind_condition |
| 2018-10-30 02:10:00 | 573 | wind_speed_10m | 10.3 | 1.24 | [0.41, 2.42] | 9.02 | extreme_wind_condition |
| 2018-10-30 02:10:00 | 499 | wind_speed_10m | 10.3 | 1.45 | [0.615, 2.63] | 8.76 | extreme_wind_condition |
| 2018-10-30 02:10:00 | 462 | wind_speed_10m | 10.3 | 1.45 | [0.615, 2.63] | 8.76 | extreme_wind_condition |
| 2018-10-30 02:10:00 | 424 | wind_speed_10m | 10.3 | 1.45 | [0.615, 2.63] | 8.76 | extreme_wind_condition |
| 2018-10-30 02:10:00 | 498 | wind_speed_10m | 10.3 | 1.45 | [0.615, 2.63] | 8.76 | extreme_wind_condition |
| 2018-10-30 02:10:00 | 461 | wind_speed_10m | 10.3 | 1.45 | [0.615, 2.63] | 8.76 | extreme_wind_condition |
| 2018-10-30 02:10:00 | 423 | wind_speed_10m | 10.3 | 1.45 | [0.615, 2.63] | 8.76 | extreme_wind_condition |
| 2018-10-30 01:10:00 | 576 | wind_speed_10m | 10.4 | 1.1 | [0.34, 2.49] | 8.68 | extreme_wind_condition |
| 2018-10-30 01:10:00 | 539 | wind_speed_10m | 10.4 | 1.1 | [0.34, 2.49] | 8.68 | extreme_wind_condition |
| 2018-10-30 01:10:00 | 538 | wind_speed_10m | 10.4 | 1.1 | [0.34, 2.49] | 8.68 | extreme_wind_condition |
| 2018-10-30 01:10:00 | 575 | wind_speed_10m | 10.4 | 1.1 | [0.34, 2.49] | 8.68 | extreme_wind_condition |
| 2018-10-30 02:10:00 | 572 | wind_speed_10m | 9.79 | 1.24 | [0.48, 2.48] | 8.55 | extreme_wind_condition |
| 2018-10-30 02:10:00 | 535 | wind_speed_10m | 9.79 | 1.24 | [0.48, 2.48] | 8.55 | extreme_wind_condition |

_Note: 1276 flagged points with band width < 1.0 (near-zero bands: night / sunrise / sunset) are excluded from this table; they remain in `scores.csv`._

## Notes

- Climatology pools (location, hour) over a ±15-day calendar window across all climatology years.
- A point is flagged when its value falls outside the [q_low, q_high] band of its bin.
- `anomaly_score = (value - median) / max(|q_high - q_low| / 2, min_score_denominator)`; sign encodes direction.
- `--quantile` is an exploratory threshold, not a definitive scientific choice; with few years the extreme quantiles are fragile.
- `--min-score-denominator` stabilises the score in near-zero bands and gates the top table above.
- Bins with fewer than the minimum number of years are labelled `insufficient_climatology`.
