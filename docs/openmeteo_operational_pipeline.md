# Open-Meteo Operational Pipeline

## Current status

Branch `feat/openmeteo-training-cl` adds Open-Meteo support alongside PVGIS legacy.
PVGIS remains the default. Open-Meteo is activated only when explicitly requested.

## Why two weather sources

**PVGIS legacy** (`piedmont_pvgis_2019.nc`): ERA5-derived reanalysis from JRC PVGIS.
Used for all thesis experiments. Not available in real time.

**Open-Meteo operational**: Historical Forecast and live Forecast APIs.
Available in real time, updated hourly, suitable for operational CL.

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

## How to inspect NetCDF

```bash
python scripts/inspect_weather_netcdf.py --path data/piedmont_pvgis_2019.nc
python scripts/inspect_weather_netcdf.py --path data/openmeteo_piedmont_2019.nc
```

## How to run feature comparison

```bash
python scripts/compare_weather_features.py \
    --pvgis-path data/piedmont_pvgis_2019.nc \
    --openmeteo-path data/openmeteo_piedmont_2019.nc \
    --out reports/weather_feature_comparison.md
```

## What changes in `solar_irradiance_poa`

In PVGIS legacy, `solar_irradiance_poa` comes from the PVGIS NetCDF.
Despite the name suggesting Plane of Array, the code treats it as GHI proxy.

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

**PVGIS legacy**: DNI and DHI estimated via Erbs from GHI proxy. Always.

**Open-Meteo operational**: If `direct_normal_irradiance` AND `diffuse_radiation`
are both present, they are used directly (no Erbs). If either is absent,
Erbs fallback activates automatically.

Direct is preferred: Erbs is statistical, Open-Meteo DNI/DHI come from NWP models.

## Feature pipeline (N_FEATURES=16)

| # | Feature | PVGIS legacy | Open-Meteo operational |
|---|---------|-------------|----------------------|
| 0 | temperature_2m | PVGIS NetCDF | Open-Meteo |
| 1 | solar_resource | `solar_irradiance_poa` | `shortwave_radiation` (GHI) |
| 2 | wind_speed_10m | PVGIS NetCDF | Open-Meteo |
| 3 | sin_solar_elev | pvlib | pvlib |
| 4 | cos_solar_elev | pvlib | pvlib |
| 5-9 | m1..m5 | QS(ENERGIA, solar) | QS(ENERGIA, solar) |
| 10 | pv_lag | ENERGIA | ENERGIA |
| 11 | kt | solar/ghi_cs | solar/ghi_cs |
| 12 | kt_std_3h | rolling(kt) | rolling(kt) |
| 13 | dghi_dt | diff(solar) | diff(solar) |
| 14 | dni_norm | Erbs | Direct or Erbs |
| 15 | dhi_norm | Erbs | Direct or Erbs |

## Continual Learning

CL uses the same `PVDataset` and `feature_set`. Each window rebuilds
features with the configured weather source.

### Replay policy

Do not mix PVGIS and Open-Meteo samples in the same replay buffer.
When switching `weather_source`, start with an empty buffer.
The buffer stores raw tensors without source metadata.

## CLI

### Training offline (main.py)

```bash
# PVGIS legacy (default)
python main.py

# Open-Meteo retrain
WEATHER_SOURCE=openmeteo_historical_forecast \
FEATURE_SET=openmeteo_operational \
OPENMETEO_PATH=data/openmeteo_piedmont_2019.nc \
python main.py
```

### Continual Learning

```bash
# PVGIS legacy (default)
python -m physiq_pv.continual.train_replay_continual --data-mode real

# Open-Meteo operational
python -m physiq_pv.continual.train_replay_continual \
    --data-mode real \
    --weather-source openmeteo_historical_forecast \
    --feature-set openmeteo_operational \
    --openmeteo-path data/openmeteo_piedmont_2019.nc
```

## Experiments to run

1. **PVGIS legacy baseline**: existing sweeps (done)
2. **Open-Meteo retrain**: `bash scripts/experiments/train_openmeteo_retrain.sh`
3. **CL Open-Meteo replay**: `bash scripts/experiments/cl_openmeteo_replay.sh`
4. **PVGIS pretrain + Open-Meteo fine-tune**: `bash scripts/experiments/finetune_pvgis_to_openmeteo.sh`
5. **Feature comparison**: `bash scripts/experiments/compare_pvgis_openmeteo_features.sh`
