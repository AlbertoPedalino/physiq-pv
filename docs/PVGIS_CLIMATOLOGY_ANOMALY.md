# PVGIS climatology anomaly

A minimal, self-contained pipeline that compares a single PVGIS **target year**
against a **multi-year PVGIS climatology** built in memory from separate annual
PVGIS NetCDF files.

## What it does

- Loads the target-year PVGIS file and the annual PVGIS files in a year range.
- Builds an in-memory climatology binned by `(location, month, day, hour)`.
  Binning by `(month, day)` instead of `day_of_year` keeps the same calendar
  day aligned across leap and non-leap years. One sample per climatology year
  per bin (simple version, no rolling window).
- For each bin stores the `median` and the band `[q_low, q_high]` with
  `q_low = 1 - quantile`, `q_high = quantile`.
- Flags target points whose value falls outside their bin band, with an
  `anomaly_score = (value - median) / (|q_high - q_low| / 2 + eps)`.
- Writes CSV outputs and a markdown report.

Climatology-level labels only:

```
normal
unusually_low_solar_potential
unusually_high_solar_potential
extreme_temperature_condition
extreme_wind_condition
insufficient_climatology
```

## Scope and boundaries

- PVGIS is treated as a **consistent physical reference**.
- This pipeline does **not** analyse real plants and does **not** use observed
  plant production.
- It does **not** train any model and does **not** modify the training pipeline.
- It does **not** download from the PVGIS API and does **not** write multi-year
  NetCDF files: the climatology lives only in memory.
- By default the target year is excluded from the climatology (leave-one-out);
  pass `--include-target-year-in-climatology` to override.
- `--quantile` (e.g. `0.975`, `0.99`) is an **exploratory** threshold to select
  the tail of the distribution, not a definitive scientific value.

## Variables

Considered if present in both target and climatology files:

```
solar_irradiance_poa
pv_power_output
temperature_2m
wind_speed_10m
```

## Outputs

```
outputs/pvgis_anomaly/pvgis_climatology_scores.csv    # one row per flagged anomaly
outputs/pvgis_anomaly/pvgis_climatology_labels.csv    # per (location, variable, label) tally
outputs/pvgis_anomaly/pvgis_climatology_summary.csv   # per-variable summary
outputs/pvgis_anomaly/pvgis_climatology_report.md     # human-readable report
```

`scores.csv` contains only flagged points (the climatological tail); normal
points are counted in the summary but not enumerated, keeping output bounded.

## Example (server)

```bash
PYTHONPATH=$PWD python scripts/run_pvgis_climatology_anomaly.py \
  --year 2019 \
  --pvgis-path /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance/piedmont_pvgis_2019.nc \
  --pvgis-climatology-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance \
  --climatology-start-year 2015 \
  --climatology-end-year 2023 \
  --out-dir outputs/pvgis_anomaly \
  --quantile 0.975
```
