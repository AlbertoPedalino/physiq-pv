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
- Device: cuda  |  Generated (UTC): 2026-06-08T17:20:27+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 19.8882 | 46.4985 | 0.8960 | 0.0293 | 2.6880 | 0.032 | 0.949 |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 18.8429 | 43.6298 | 0.8801 | 0.0247 | 2.6759 | 0.032 | 0.949 |
| group:rare_or_extreme | 983379 | 29.5123 | 67.3995 | 1.0422 | 0.6501 | 2.7985 | 0.032 | 0.944 |
| label:unusually_low_solar_potential | 118147 | 95.0065 | 147.5616 | 1.8791 | 1.5491 | 3.4752 | 0.007 | 0.871 |
| label:unusually_high_solar_potential | 38401 | 47.2063 | 74.7810 | 1.7672 | 1.1603 | 3.7962 | 0.038 | 0.961 |
| label:extreme_temperature_condition | 436524 | 17.5404 | 41.2078 | 0.8619 | 0.4638 | 2.3595 | 0.040 | 0.958 |
| label:extreme_wind_condition | 458693 | 23.5962 | 53.4905 | 0.9641 | 0.0399 | 2.8495 | 0.030 | 0.948 |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

- MAE normal: 18.8429  |  MAE rare/extreme: 29.5123  |  ratio: **1.57×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty calibration

MC Dropout std is a relative uncertainty measure. Raw intervals `mean ± 1.96 std` are not guaranteed to be calibrated. Post-hoc calibration scales std with a factor estimated on a separate calibration set; the test year is used only for evaluation.

- Calibration years: 2018
- Coverage target: **0.950**
- Calibration factor: **85.4765**
- Calibration predictions: **10037664**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| global | 0.032 | 0.949 |
| normal | 0.032 | 0.949 |
| rare/extreme | 0.032 | 0.944 |

## Stratified uncertainty calibration

Global calibration uses a single factor for every test row; **group**/**label** strategies estimate separate factors on the calibration year's anomaly strata so rare/extreme bands are not under-covered. A stratum with fewer than `min_samples` calibration points falls back (label → rare/extreme group → global).

- Calibration strategy: **group**
- Calibration anomaly scores: outputs/pvgis_anomaly_2018_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Min samples per stratum: **1000**
- Global factor (k_global): **85.4765**
- Factor normal: **85.3291 (n=9207477)**
- Factor rare_or_extreme: **87.3320 (n=830187)**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| normal | 0.032 | 0.949 |
| rare/extreme | 0.032 | 0.944 |

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 18.8429  |  MAE rare/extreme: 29.5123  |  rare/normal MAE ratio: **1.57×**
- Mean uncertainty (std) normal: 0.8801  |  rare/extreme: 1.0422  |  rare/normal uncertainty ratio: **1.18×**
- Raw coverage@95 normal: 0.032  |  rare/extreme: 0.032
- Calibrated coverage@95 normal: 0.949  |  rare/extreme: 0.944

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.57×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.18×).

