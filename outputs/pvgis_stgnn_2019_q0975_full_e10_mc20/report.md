# PVGIS-only ST-GNN forecasting report

Reuses the existing STGNN architecture on a **PVGIS-only** input. No real plant production, no ENERGIA, no quality score, no kWp/UPN. Anomaly labels are used **only** for stratified evaluation.

## Experiment

- Mode: **pvgis_stgnn**
- Model type: **stgnn**
- Feature set: **full**
- W&B enabled: **False**
- MC Dropout: **enabled** (experimental)

## Parameters

- Target variable: **pv_power_output**
- Selected features (11): temperature_2m, solar_irradiance_poa, wind_speed_10m, sin_elev, cos_elev, kt, kt_std_3h, dghi_dt, dni_norm, dhi_norm, pv_lag_pvgis
- MC samples: **20**
- seq_len: **24**  |  horizon: **1**
- Train years: 2016,2017,2018
- Test year: **2019**
- Nodes (locations): **1149**  |  epochs: **10**
- batch_size: 8  |  lr: 0.001
- Anomaly scores: outputs/pvgis_anomaly_2019_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Predictions: **10037664**
- Device: cuda  |  Generated (UTC): 2026-06-05T16:06:01+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95 |
|---|---|---|---|---|---|---|---|
| all | 10037664 | 17.5929 | 43.8817 | 1.0505 | 0.0267 | 3.0643 | 0.249 |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95 |
|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 16.5701 | 40.9186 | 1.0294 | 0.0201 | 3.0332 | 0.254 |
| group:rare_or_extreme | 983379 | 27.0108 | 65.1084 | 1.2452 | 0.6135 | 3.3515 | 0.207 |
| label:unusually_low_solar_potential | 118147 | 95.2772 | 146.9717 | 2.1557 | 1.3493 | 4.8323 | 0.004 |
| label:unusually_high_solar_potential | 38401 | 38.0770 | 64.6405 | 2.5066 | 1.7700 | 4.9349 | 0.067 |
| label:extreme_temperature_condition | 436524 | 14.3056 | 36.9651 | 1.0060 | 0.5160 | 2.6186 | 0.244 |
| label:extreme_wind_condition | 458693 | 21.4787 | 51.0994 | 1.1571 | 0.0419 | 3.4270 | 0.228 |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

- MAE normal: 16.5701  |  MAE rare/extreme: 27.0108  |  ratio: **1.63×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 16.5701  |  MAE rare/extreme: 27.0108  |  rare/normal MAE ratio: **1.63×**
- Mean uncertainty (std) normal: 1.0294  |  rare/extreme: 1.2452  |  rare/normal uncertainty ratio: **1.21×**
- Coverage@95 normal: 0.254  |  rare/extreme: 0.207

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.63×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.21×).

