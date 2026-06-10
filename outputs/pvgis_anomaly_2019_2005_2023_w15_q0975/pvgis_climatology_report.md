# PVGIS climatology anomaly report

Target PVGIS year compared against a multi-year PVGIS climatology. PVGIS is used as a consistent physical reference. This analysis does **not** inspect real plants, does **not** use observed plant energy, and does **not** touch the training pipeline.

## Parameters

- Target year: **2019**
- Climatology years used: 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2020, 2021, 2022, 2023
- Quantile band: **[0.025, 0.975]** (quantile=0.975)
- Min climatology years per bin: **3**
- Climatology window: **±15 days**
- Min score denominator: **1.0**
- Locations: **1149**  |  Target timesteps: **8760**
- Variables analyzed: solar_irradiance_poa, pv_power_output, temperature_2m, wind_speed_10m
- Total flagged anomalies: **1169926**
- Generated (UTC): 2026-06-04T14:20:41+00:00

## Per-variable summary

| variable | valid | normal | insufficient | flagged | high | low | % flagged | max |score| |
|---|---|---|---|---|---|---|---|---|
| solar_irradiance_poa | 10065240 | 9938102 | 0 | 127138 | 19973 | 107165 | 1.263% | 2.71 |
| pv_power_output | 10065240 | 9918404 | 0 | 146836 | 30879 | 115957 | 1.459% | 2.72 |
| temperature_2m | 10065240 | 9628714 | 0 | 436526 | 314767 | 121759 | 4.337% | 2.54 |
| wind_speed_10m | 10065240 | 9605814 | 0 | 459426 | 287473 | 171953 | 4.564% | 5.23 |

## Top 20 most extreme conditions

| timestamp | location | variable | value | median | band | score | label |
|---|---|---|---|---|---|---|---|
| 2019-07-15 06:10:00 | 456 | wind_speed_10m | 6.48 | 1.07 | [0.34, 2.41] | 5.23 | extreme_wind_condition |
| 2019-07-15 06:10:00 | 493 | wind_speed_10m | 6.48 | 1.07 | [0.34, 2.41] | 5.23 | extreme_wind_condition |
| 2019-07-15 06:10:00 | 418 | wind_speed_10m | 6.48 | 1.07 | [0.34, 2.41] | 5.23 | extreme_wind_condition |
| 2019-07-15 06:10:00 | 455 | wind_speed_10m | 6.48 | 1.07 | [0.34, 2.41] | 5.23 | extreme_wind_condition |
| 2019-07-15 06:10:00 | 417 | wind_speed_10m | 6.48 | 1.07 | [0.34, 2.41] | 5.23 | extreme_wind_condition |
| 2019-07-15 06:10:00 | 492 | wind_speed_10m | 6.48 | 1.07 | [0.34, 2.41] | 5.23 | extreme_wind_condition |
| 2019-07-15 06:10:00 | 494 | wind_speed_10m | 6.34 | 1.1 | [0.41, 2.48] | 5.06 | extreme_wind_condition |
| 2019-07-15 06:10:00 | 457 | wind_speed_10m | 6.34 | 1.1 | [0.41, 2.48] | 5.06 | extreme_wind_condition |
| 2019-07-15 06:10:00 | 419 | wind_speed_10m | 6.34 | 1.1 | [0.41, 2.48] | 5.06 | extreme_wind_condition |
| 2019-07-15 07:10:00 | 569 | wind_speed_10m | 7.52 | 1.1 | [0.21, 2.76] | 5.04 | extreme_wind_condition |
| 2019-07-15 07:10:00 | 530 | wind_speed_10m | 7.45 | 1.03 | [0.21, 2.76] | 5.04 | extreme_wind_condition |
| 2019-07-15 07:10:00 | 532 | wind_speed_10m | 7.52 | 1.1 | [0.21, 2.76] | 5.04 | extreme_wind_condition |
| 2019-07-15 07:10:00 | 568 | wind_speed_10m | 7.45 | 1.03 | [0.21, 2.76] | 5.04 | extreme_wind_condition |
| 2019-07-15 07:10:00 | 531 | wind_speed_10m | 7.45 | 1.03 | [0.21, 2.76] | 5.04 | extreme_wind_condition |
| 2019-07-15 07:10:00 | 567 | wind_speed_10m | 7.45 | 1.03 | [0.21, 2.76] | 5.04 | extreme_wind_condition |
| 2019-07-15 08:10:00 | 448 | wind_speed_10m | 5.72 | 0.69 | [0.07, 2] | 5.03 | extreme_wind_condition |
| 2019-07-15 08:10:00 | 410 | wind_speed_10m | 5.72 | 0.69 | [0.07, 2] | 5.03 | extreme_wind_condition |
| 2019-07-15 08:10:00 | 485 | wind_speed_10m | 5.72 | 0.69 | [0.07, 2] | 5.03 | extreme_wind_condition |
| 2019-07-15 08:10:00 | 484 | wind_speed_10m | 5.72 | 0.69 | [0.07, 2] | 5.03 | extreme_wind_condition |
| 2019-07-15 08:10:00 | 409 | wind_speed_10m | 5.72 | 0.69 | [0.07, 2] | 5.03 | extreme_wind_condition |

_Note: 1887 flagged points with band width < 1.0 (near-zero bands: night / sunrise / sunset) are excluded from this table; they remain in `scores.csv`._

## Notes

- Climatology pools (location, hour) over a ±15-day calendar window across all climatology years.
- A point is flagged when its value falls outside the [q_low, q_high] band of its bin.
- `anomaly_score = (value - median) / max(|q_high - q_low| / 2, min_score_denominator)`; sign encodes direction.
- `--quantile` is an exploratory threshold, not a definitive scientific choice; with few years the extreme quantiles are fragile.
- `--min-score-denominator` stabilises the score in near-zero bands and gates the top table above.
- Bins with fewer than the minimum number of years are labelled `insufficient_climatology`.
