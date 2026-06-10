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
- Device: cuda  |  Generated (UTC): 2026-06-08T16:01:55+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 21.6637 | 47.9497 | 0.8541 | 0.0443 | 2.4671 | 0.027 | 0.948 |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 20.5061 | 45.1663 | 0.8390 | 0.0385 | 2.4554 | 0.027 | 0.948 |
| group:rare_or_extreme | 983379 | 32.3219 | 68.4507 | 0.9931 | 0.7245 | 2.5749 | 0.028 | 0.945 |
| label:unusually_low_solar_potential | 118147 | 102.6080 | 146.1299 | 1.6340 | 1.3073 | 3.0134 | 0.002 | 0.794 |
| label:unusually_high_solar_potential | 38401 | 57.7614 | 84.3568 | 1.9102 | 1.1237 | 4.3237 | 0.023 | 0.941 |
| label:extreme_temperature_condition | 436524 | 20.1310 | 44.3698 | 0.8421 | 0.5689 | 2.1917 | 0.036 | 0.971 |
| label:extreme_wind_condition | 458693 | 25.5033 | 54.8534 | 0.9175 | 0.0609 | 2.6278 | 0.026 | 0.957 |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

- MAE normal: 20.5061  |  MAE rare/extreme: 32.3219  |  ratio: **1.58×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty calibration

MC Dropout std is a relative uncertainty measure. Raw intervals `mean ± 1.96 std` are not guaranteed to be calibrated. Post-hoc calibration scales std with a factor estimated on a separate calibration set; the test year is used only for evaluation.

- Calibration years: 2018
- Coverage target: **0.950**
- Calibration factor: **78.3707**
- Calibration predictions: **10037664**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| global | 0.027 | 0.948 |
| normal | 0.027 | 0.948 |
| rare/extreme | 0.028 | 0.945 |

## Stratified uncertainty calibration

Global calibration uses a single factor for every test row; **group**/**label** strategies estimate separate factors on the calibration year's anomaly strata so rare/extreme bands are not under-covered. A stratum with fewer than `min_samples` calibration points falls back (label → rare/extreme group → global).

- Calibration strategy: **group**
- Calibration anomaly scores: outputs/pvgis_anomaly_2018_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Min samples per stratum: **1000**
- Global factor (k_global): **78.3707**
- Factor normal: **77.9072 (n=9207477)**
- Factor rare_or_extreme: **84.3883 (n=830187)**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| normal | 0.027 | 0.948 |
| rare/extreme | 0.028 | 0.945 |

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 20.5061  |  MAE rare/extreme: 32.3219  |  rare/normal MAE ratio: **1.58×**
- Mean uncertainty (std) normal: 0.8390  |  rare/extreme: 0.9931  |  rare/normal uncertainty ratio: **1.18×**
- Raw coverage@95 normal: 0.027  |  rare/extreme: 0.028
- Calibrated coverage@95 normal: 0.948  |  rare/extreme: 0.945

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.58×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.18×).

