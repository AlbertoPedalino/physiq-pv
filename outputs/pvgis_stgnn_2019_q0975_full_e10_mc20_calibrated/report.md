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
- Train years: 2016,2017
- Test year: **2019**
- Nodes (locations): **1149**  |  epochs: **10**
- batch_size: 8  |  lr: 0.001
- Anomaly scores: outputs/pvgis_anomaly_2019_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Predictions: **10037664**
- Device: cuda  |  Generated (UTC): 2026-06-05T18:41:05+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 18.1047 | 44.5370 | 0.9539 | 0.0052 | 2.7614 | 0.042 | 0.951 |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 16.9432 | 41.4434 | 0.9325 | 0.0042 | 2.7347 | 0.042 | 0.954 |
| group:rare_or_extreme | 983379 | 28.7984 | 66.5775 | 1.1513 | 0.7322 | 3.0063 | 0.044 | 0.918 |
| label:unusually_low_solar_potential | 118147 | 105.4744 | 150.9227 | 2.2436 | 1.5774 | 4.5989 | 0.001 | 0.659 |
| label:unusually_high_solar_potential | 38401 | 39.5718 | 64.4802 | 1.9180 | 1.3549 | 3.6350 | 0.050 | 0.898 |
| label:extreme_temperature_condition | 436524 | 15.1046 | 37.9136 | 0.9257 | 0.5482 | 2.4214 | 0.061 | 0.960 |
| label:extreme_wind_condition | 458693 | 22.6254 | 52.3973 | 1.0415 | 0.0141 | 2.9902 | 0.038 | 0.936 |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

- MAE normal: 16.9432  |  MAE rare/extreme: 28.7984  |  ratio: **1.70×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty calibration

MC Dropout std is a relative uncertainty measure. Raw intervals `mean ± 1.96 std` are not guaranteed to be calibrated. Post-hoc calibration scales std with a factor estimated on a separate calibration set; the test year is used only for evaluation.

- Calibration years: 2018
- Coverage target: **0.950**
- Calibration factor: **49.4703**
- Calibration predictions: **10037664**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| global | 0.042 | 0.951 |
| normal | 0.042 | 0.954 |
| rare/extreme | 0.044 | 0.918 |

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 16.9432  |  MAE rare/extreme: 28.7984  |  rare/normal MAE ratio: **1.70×**
- Mean uncertainty (std) normal: 0.9325  |  rare/extreme: 1.1513  |  rare/normal uncertainty ratio: **1.23×**
- Raw coverage@95 normal: 0.042  |  rare/extreme: 0.044
- Calibrated coverage@95 normal: 0.954  |  rare/extreme: 0.918

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.70×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.23×).

