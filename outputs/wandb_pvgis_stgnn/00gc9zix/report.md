# PVGIS-only ST-GNN forecasting report

Reuses the existing STGNN architecture on a **PVGIS-only** input. No real plant production, no ENERGIA, no quality score, no kWp/UPN. Anomaly labels are used **only** for stratified evaluation.

## Experiment

- Mode: **pvgis_stgnn**
- Model type: **stgnn**
- Feature set: **full**
- W&B enabled: **True**
- MC Dropout: **enabled** (experimental)

## Parameters

- Target variable: **pv_power_output**
- Selected features (11): temperature_2m, solar_irradiance_poa, wind_speed_10m, sin_elev, cos_elev, kt, kt_std_3h, dghi_dt, dni_norm, dhi_norm, pv_lag_pvgis
- MC samples: **20**
- seq_len: **24**  |  horizon: **1**
- Train years: 2016,2017
- Test year: **2019**
- Nodes (locations): **1149**  |  epochs: **5**
- batch_size: 16  |  lr: 0.001
- Anomaly scores: outputs/pvgis_anomaly_2019_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Predictions: **10037664**
- Device: cuda  |  Generated (UTC): 2026-06-08T15:16:06+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 21.6657 | 47.7534 | 0.7886 | 0.0321 | 2.4014 | 0.021 | 0.949 |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 20.5244 | 44.9506 | 0.7751 | 0.0289 | 2.3913 | 0.021 | 0.950 |
| group:rare_or_extreme | 983379 | 32.1741 | 68.3568 | 0.9128 | 0.4765 | 2.4915 | 0.021 | 0.934 |
| label:unusually_low_solar_potential | 118147 | 98.8393 | 145.4177 | 1.7400 | 1.5086 | 3.2110 | 0.012 | 0.835 |
| label:unusually_high_solar_potential | 38401 | 62.6869 | 86.3961 | 1.4033 | 0.8435 | 3.2542 | 0.015 | 0.695 |
| label:extreme_temperature_condition | 436524 | 20.1450 | 44.2070 | 0.7214 | 0.3583 | 2.0367 | 0.025 | 0.956 |
| label:extreme_wind_condition | 458693 | 25.5875 | 54.8340 | 0.8577 | 0.0379 | 2.5627 | 0.020 | 0.955 |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

- MAE normal: 20.5244  |  MAE rare/extreme: 32.1741  |  ratio: **1.57×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty calibration

MC Dropout std is a relative uncertainty measure. Raw intervals `mean ± 1.96 std` are not guaranteed to be calibrated. Post-hoc calibration scales std with a factor estimated on a separate calibration set; the test year is used only for evaluation.

- Calibration years: 2018
- Coverage target: **0.950**
- Calibration factor: **86.5394**
- Calibration predictions: **10037664**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| global | 0.021 | 0.949 |
| normal | 0.021 | 0.950 |
| rare/extreme | 0.021 | 0.934 |

## Stratified uncertainty calibration

Global calibration uses a single factor for every test row; **group**/**label** strategies estimate separate factors on the calibration year's anomaly strata so rare/extreme bands are not under-covered. A stratum with fewer than `min_samples` calibration points falls back (label → rare/extreme group → global).

- Calibration strategy: **group**
- Calibration anomaly scores: outputs/pvgis_anomaly_2018_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Min samples per stratum: **1000**
- Global factor (k_global): **86.5394**
- Factor normal: **85.5784 (n=9207477)**
- Factor rare_or_extreme: **96.8136 (n=830187)**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| normal | 0.021 | 0.950 |
| rare/extreme | 0.021 | 0.934 |

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 20.5244  |  MAE rare/extreme: 32.1741  |  rare/normal MAE ratio: **1.57×**
- Mean uncertainty (std) normal: 0.7751  |  rare/extreme: 0.9128  |  rare/normal uncertainty ratio: **1.18×**
- Raw coverage@95 normal: 0.021  |  rare/extreme: 0.021
- Calibrated coverage@95 normal: 0.950  |  rare/extreme: 0.934

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.57×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.18×).

