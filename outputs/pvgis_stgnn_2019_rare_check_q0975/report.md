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
- Nodes (locations): **1149**  |  epochs: **2**
- batch_size: 8  |  lr: 0.001
- Anomaly scores: outputs/pvgis_anomaly_2019_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Predictions: **10037664**
- Device: cuda  |  Generated (UTC): 2026-06-04T14:31:04+00:00

## Global metrics

| stratum | count | MAE | RMSE |
|---|---|---|---|
| all | 10037664 | 25.5708 | 51.7168 |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE |
|---|---|---|---|
| group:normal | 9054285 | 24.3161 | 48.8575 |
| group:rare_or_extreme | 983379 | 37.1235 | 72.9543 |
| label:unusually_low_solar_potential | 118147 | 108.4215 | 150.7040 |
| label:unusually_high_solar_potential | 38401 | 74.3804 | 96.3170 |
| label:extreme_temperature_condition | 436524 | 25.0423 | 50.8125 |
| label:extreme_wind_condition | 458693 | 29.0393 | 58.0043 |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

- MAE normal: 24.3161  |  MAE rare/extreme: 37.1235  |  ratio: **1.53×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

