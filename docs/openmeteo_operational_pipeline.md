# Open-Meteo Operational Pipeline

## Current status

This branch trains with Open-Meteo weather. `main.py` defaults to
`openmeteo_historical_forecast` and `openmeteo_operational`.

## Weather source

**Open-Meteo operational**: Historical Forecast and live Forecast APIs.
Available in real time, updated hourly, suitable for operational retraining and forecasting.

## Required Open-Meteo variables

| Variable | Unit | Required | Notes |
|----------|------|----------|-------|
| `temperature_2m` | C | Yes | |
| `wind_speed_10m` | m/s | Yes | |
| `shortwave_radiation` | W/m2 | Yes | GHI, maps to `solar_irradiance_poa` |
| `direct_normal_irradiance` | W/m2 | Recommended | Used directly if present, Erbs fallback otherwise |
| `diffuse_radiation` | W/m2 | Recommended | Used directly if present, Erbs fallback otherwise |

## How to download Open-Meteo Historical Forecast

```bash
python scripts/setup_openmeteo_data.py

# Dry run (shows what would be downloaded):
python scripts/download_openmeteo_historical_forecast.py \
    --plants-path data/energy_with_coordinates.csv \
    --start-date 2019-03-01 --end-date 2019-12-31 \
    --source historical_forecast \
    --dry-run

# Real download:
python scripts/download_openmeteo_historical_forecast.py \
    --plants-path data/energy_with_coordinates.csv \
    --start-date 2019-03-01 --end-date 2019-12-31 \
    --out data/openmeteo_piedmont_2019.nc \
    --source historical_forecast
```

## How to validate the pipeline

```bash
python scripts/check_openmeteo_pipeline.py \
    --openmeteo-path data/openmeteo_piedmont_2019.nc \
    --max-plants 5 \
    --max-time-steps 200
```

## `solar_irradiance_poa`

In Open-Meteo operational, `shortwave_radiation` (GHI, W/m2) is mapped to
`solar_irradiance_poa` for downstream compatibility. The feature name in
`FEATURE_NAMES_OPENMETEO_OPERATIONAL` is `shortwave_radiation_ghi_proxy`.

## Why this is not true POA

POA (Plane of Array) irradiance requires tilt and azimuth of each panel.
We do not have tilt/azimuth for the 1022 Piedmont plants.
Using GHI (horizontal) is the safe default that works without plant metadata.

## Why `global_tilted_irradiance` is not default

Open-Meteo provides GTI but requires tilt/azimuth parameters.
Without reliable per-plant geometry, GTI would be based on wrong assumptions.

## DNI/DHI: direct vs Erbs fallback

**Open-Meteo operational**: If `direct_normal_irradiance` AND `diffuse_radiation`
are both present, they are used directly (no Erbs). If either is absent,
Erbs fallback activates automatically.

Direct is preferred: Erbs is statistical, Open-Meteo DNI/DHI come from NWP models.

## Feature pipeline (N_FEATURES=16)

| # | Feature | Open-Meteo source |
|---|---------|-------------------|
| 0 | temperature_2m | `temperature_2m` |
| 1 | solar_resource | `shortwave_radiation` (GHI) |
| 2 | wind_speed_10m | `wind_speed_10m` |
| 3 | sin_solar_elev | pvlib |
| 4 | cos_solar_elev | pvlib |
| 5-9 | m1..m5 | QS(ENERGIA, solar) |
| 10 | pv_lag | ENERGIA |
| 11 | kt | solar/ghi_cs |
| 12 | kt_std_3h | rolling(kt) |
| 13 | dghi_dt | diff(solar) |
| 14 | dni_norm | direct DNI or Erbs fallback |
| 15 | dhi_norm | direct DHI or Erbs fallback |

## CLI

### Training offline (main.py)

```bash
python main.py
```

## Training

```bash
python main.py
```
