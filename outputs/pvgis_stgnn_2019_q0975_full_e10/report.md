# PVGIS-only ST-GNN forecasting report

Reuses the existing STGNN architecture on a **PVGIS-only** input. No real plant production, no ENERGIA, no quality score, no kWp/UPN. Anomaly labels are used **only** for stratified evaluation.

## Experiment

- Mode: **pvgis_stgnn**
- Model type: **stgnn**
- Feature set: **full**
- W&B enabled: **False**
- MC Dropout: **disabled** (not implemented yet — reserved flag)

## Parameters

- Target variable: **pv_power_output**
- Selected features (11): temperature_2m, solar_irradiance_poa, wind_speed_10m, sin_elev, cos_elev, kt, kt_std_3h, dghi_dt, dni_norm, dhi_norm, pv_lag_pvgis
- seq_len: **24**  |  horizon: **1**
- Train years: 2016,2017,2018
- Test year: **2019**
- Nodes (locations): **1149**  |  epochs: **10**
- batch_size: 8  |  lr: 0.001
- Anomaly scores: outputs/pvgis_anomaly_2019_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Predictions: **10037664**
- Device: cuda  |  Generated (UTC): 2026-06-04T15:35:56+00:00

## Global metrics

| stratum | count | MAE | RMSE |
|---|---|---|---|
| all | 10037664 | 17.5763 | 43.8297 |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE |
|---|---|---|---|
| group:normal | 9054285 | 16.5619 | 40.8823 |
| group:rare_or_extreme | 983379 | 26.9162 | 64.9613 |
| label:unusually_low_solar_potential | 118147 | 95.0536 | 146.4789 |
| label:unusually_high_solar_potential | 38401 | 38.0118 | 64.8045 |
| label:extreme_temperature_condition | 436524 | 14.1482 | 36.9415 |
| label:extreme_wind_condition | 458693 | 21.4685 | 51.0519 |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

- MAE normal: 16.5619  |  MAE rare/extreme: 26.9162  |  ratio: **1.63×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

