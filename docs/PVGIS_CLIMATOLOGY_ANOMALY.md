# PVGIS climatology anomaly

A minimal, self-contained pipeline that compares a single PVGIS **target year**
against a **multi-year PVGIS climatology** built in memory from separate annual
PVGIS NetCDF files.

## What it does

- Loads the target-year PVGIS file and the annual PVGIS files in a year range.
- Builds an in-memory climatology keyed by `(location, calendar_day, hour)`.
  For each target day it pools samples over a `±--climatology-window-days`
  calendar-day window, across **all** climatology years. The calendar-day index
  is leap-consistent (the same calendar date aligns across leap / non-leap
  years), so there is no `day_of_year` drift after Feb 29 and the window is
  contiguous. With `W` years and a `±D`-day window a bin pools up to
  `W * (2D + 1)` samples instead of a few exact-day samples.
- For each bin stores the `median` and the band `[q_low, q_high]` with
  `q_low = 1 - quantile`, `q_high = quantile`.
- Flags target points whose value falls outside their bin band, with
  `anomaly_score = (value - median) / max(|q_high - q_low| / 2, --min-score-denominator)`.
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
- Prefer using **all available years** (`2005–2023` if present) for the
  climatology. `2015–2023` is fine for a light/debug run, but more years make
  the quantiles more stable. The target year is always excluded by default.
- `--climatology-window-days` (default `15`) widens each bin by pooling nearby
  calendar days, raising the per-bin sample count. Without a window, with few
  years, the extreme quantiles `0.025 / 0.975` collapse to near min/max and are
  fragile.
- `--min-score-denominator` (default `1.0`) floors the score denominator to
  avoid huge `anomaly_score` values in near-zero climatology bands (night /
  sunrise / sunset). It also gates the report's "top extreme conditions" table,
  which excludes bins whose band width is below this floor (those rows still
  appear in `scores.csv`).
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

## Recommended command (server)

Uses all available years `2005–2023` (target `2019` excluded by default) with a
±15-day window and score stabilisation:

```bash
PYTHONPATH=$PWD python scripts/run_pvgis_climatology_anomaly.py \
  --year 2019 \
  --pvgis-path /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance/piedmont_pvgis_2019.nc \
  --pvgis-climatology-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance \
  --climatology-start-year 2005 \
  --climatology-end-year 2023 \
  --out-dir outputs/pvgis_anomaly_2019_2005_2023_w15 \
  --quantile 0.975 \
  --climatology-window-days 15 \
  --min-score-denominator 1.0
```

### Light / debug run

Narrower year range, same logic:

```bash
PYTHONPATH=$PWD python scripts/run_pvgis_climatology_anomaly.py \
  --year 2019 \
  --pvgis-path /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance/piedmont_pvgis_2019.nc \
  --pvgis-climatology-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance \
  --climatology-start-year 2015 \
  --climatology-end-year 2023 \
  --out-dir outputs/pvgis_anomaly \
  --quantile 0.975 \
  --climatology-window-days 15 \
  --min-score-denominator 1.0
```
