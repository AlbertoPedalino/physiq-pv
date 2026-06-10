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
- Device: cuda  |  Generated (UTC): 2026-06-08T11:09:55+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 18.1081 | 44.5420 | 0.9680 | 0.0047 | 2.8873 | 0.040 | 0.951 |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 16.9976 | 41.5335 | 0.9472 | 0.0040 | 2.8568 | 0.039 | 0.951 |
| group:rare_or_extreme | 983379 | 28.3325 | 66.0934 | 1.1602 | 0.5442 | 3.1629 | 0.041 | 0.943 |
| label:unusually_low_solar_potential | 118147 | 102.8737 | 149.1723 | 2.3345 | 1.7363 | 4.7508 | 0.002 | 0.785 |
| label:unusually_high_solar_potential | 38401 | 40.8893 | 65.6824 | 1.9349 | 1.2432 | 3.9078 | 0.037 | 0.930 |
| label:extreme_temperature_condition | 436524 | 14.7142 | 37.5271 | 0.9010 | 0.4100 | 2.4345 | 0.056 | 0.971 |
| label:extreme_wind_condition | 458693 | 22.2936 | 52.1428 | 1.0773 | 0.0076 | 3.1698 | 0.037 | 0.957 |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

- MAE normal: 16.9976  |  MAE rare/extreme: 28.3325  |  ratio: **1.67×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty calibration

MC Dropout std is a relative uncertainty measure. Raw intervals `mean ± 1.96 std` are not guaranteed to be calibrated. Post-hoc calibration scales std with a factor estimated on a separate calibration set; the test year is used only for evaluation.

- Calibration years: 2018
- Coverage target: **0.950**
- Calibration factor: **50.3668**
- Calibration predictions: **10037664**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| global | 0.040 | 0.951 |
| normal | 0.039 | 0.951 |
| rare/extreme | 0.041 | 0.943 |

## Stratified uncertainty calibration

Global calibration uses a single factor for every test row; **group**/**label** strategies estimate separate factors on the calibration year's anomaly strata so rare/extreme bands are not under-covered. A stratum with fewer than `min_samples` calibration points falls back (label → rare/extreme group → global).

- Calibration strategy: **group**
- Calibration anomaly scores: outputs/pvgis_anomaly_2018_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Min samples per stratum: **1000**
- Global factor (k_global): **50.3668**
- Factor normal: **49.2274 (n=9207477)**
- Factor rare_or_extreme: **60.6686 (n=830187)**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| normal | 0.039 | 0.951 |
| rare/extreme | 0.041 | 0.943 |

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 16.9976  |  MAE rare/extreme: 28.3325  |  rare/normal MAE ratio: **1.67×**
- Mean uncertainty (std) normal: 0.9472  |  rare/extreme: 1.1602  |  rare/normal uncertainty ratio: **1.22×**
- Raw coverage@95 normal: 0.039  |  rare/extreme: 0.041
- Calibrated coverage@95 normal: 0.951  |  rare/extreme: 0.943

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.67×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.22×).

