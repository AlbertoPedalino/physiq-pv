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
- MC samples: **3**
- seq_len: **24**  |  horizon: **1**
- Train years: 2016,2017,2018
- Test year: **2019**
- Nodes (locations): **1149**  |  epochs: **2**
- batch_size: 8  |  lr: 0.001
- Anomaly scores: outputs/pvgis_anomaly_2019_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Predictions: **5745000**
- Device: cuda  |  Generated (UTC): 2026-06-05T15:23:57+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95 |
|---|---|---|---|---|---|---|---|
| all | 5745000 | 25.0366 | 50.6671 | 0.6560 | 0.0560 | 1.9795 | 0.016 |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95 |
|---|---|---|---|---|---|---|---|
| group:normal | 5186644 | 23.8765 | 47.9292 | 0.6412 | 0.0505 | 1.9617 | 0.016 |
| group:rare_or_extreme | 558356 | 35.8132 | 71.2372 | 0.7935 | 0.3723 | 2.1270 | 0.018 |
| label:unusually_low_solar_potential | 65663 | 106.9241 | 151.2603 | 1.2535 | 1.0348 | 2.4594 | 0.002 |
| label:unusually_high_solar_potential | 18622 | 78.4098 | 100.8890 | 1.5923 | 1.2819 | 3.0607 | 0.007 |
| label:extreme_temperature_condition | 257981 | 25.0601 | 48.3880 | 0.7675 | 0.3562 | 2.0490 | 0.024 |
| label:extreme_wind_condition | 255133 | 26.7313 | 54.4509 | 0.6577 | 0.0557 | 1.9969 | 0.015 |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

- MAE normal: 23.8765  |  MAE rare/extreme: 35.8132  |  ratio: **1.50×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **3**
- MAE normal: 23.8765  |  MAE rare/extreme: 35.8132  |  rare/normal MAE ratio: **1.50×**
- Mean uncertainty (std) normal: 0.6412  |  rare/extreme: 0.7935  |  rare/normal uncertainty ratio: **1.24×**
- Coverage@95 normal: 0.016  |  rare/extreme: 0.018

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.50×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.24×).

