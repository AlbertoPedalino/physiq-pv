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
- Device: cuda  |  Generated (UTC): 2026-06-08T16:54:28+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 21.3539 | 48.3347 | 0.8669 | 0.0654 | 2.5715 | 0.029 | 0.947 |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 20.0252 | 44.8090 | 0.8527 | 0.0459 | 2.5595 | 0.029 | 0.948 |
| group:rare_or_extreme | 983379 | 33.5873 | 73.2114 | 0.9979 | 0.6421 | 2.6821 | 0.026 | 0.943 |
| label:unusually_low_solar_potential | 118147 | 118.0998 | 167.2229 | 1.6806 | 1.3023 | 3.2387 | 0.002 | 0.711 |
| label:unusually_high_solar_potential | 38401 | 34.5961 | 62.6296 | 1.9945 | 1.5251 | 3.8932 | 0.082 | 0.984 |
| label:extreme_temperature_condition | 436524 | 20.1951 | 43.6699 | 0.8187 | 0.5202 | 2.2143 | 0.029 | 0.974 |
| label:extreme_wind_condition | 458693 | 25.6157 | 55.6044 | 0.9291 | 0.1702 | 2.7367 | 0.025 | 0.966 |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

- MAE normal: 20.0252  |  MAE rare/extreme: 33.5873  |  ratio: **1.68×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty calibration

MC Dropout std is a relative uncertainty measure. Raw intervals `mean ± 1.96 std` are not guaranteed to be calibrated. Post-hoc calibration scales std with a factor estimated on a separate calibration set; the test year is used only for evaluation.

- Calibration years: 2018
- Coverage target: **0.950**
- Calibration factor: **72.8796**
- Calibration predictions: **10037664**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| global | 0.029 | 0.947 |
| normal | 0.029 | 0.948 |
| rare/extreme | 0.026 | 0.943 |

## Stratified uncertainty calibration

Global calibration uses a single factor for every test row; **group**/**label** strategies estimate separate factors on the calibration year's anomaly strata so rare/extreme bands are not under-covered. A stratum with fewer than `min_samples` calibration points falls back (label → rare/extreme group → global).

- Calibration strategy: **group**
- Calibration anomaly scores: outputs/pvgis_anomaly_2018_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Min samples per stratum: **1000**
- Global factor (k_global): **72.8796**
- Factor normal: **71.9511 (n=9207477)**
- Factor rare_or_extreme: **87.3917 (n=830187)**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| normal | 0.029 | 0.948 |
| rare/extreme | 0.026 | 0.943 |

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 20.0252  |  MAE rare/extreme: 33.5873  |  rare/normal MAE ratio: **1.68×**
- Mean uncertainty (std) normal: 0.8527  |  rare/extreme: 0.9979  |  rare/normal uncertainty ratio: **1.17×**
- Raw coverage@95 normal: 0.029  |  rare/extreme: 0.026
- Calibrated coverage@95 normal: 0.948  |  rare/extreme: 0.943

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.68×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.17×).

