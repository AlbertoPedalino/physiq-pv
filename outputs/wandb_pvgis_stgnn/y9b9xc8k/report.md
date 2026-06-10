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
- Device: cuda  |  Generated (UTC): 2026-06-08T16:28:29+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 19.9306 | 46.4347 | 0.8945 | 0.0753 | 2.5766 | 0.033 | 0.950 |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 18.7909 | 43.2494 | 0.8785 | 0.0638 | 2.5647 | 0.033 | 0.951 |
| group:rare_or_extreme | 983379 | 30.4245 | 69.1842 | 1.0419 | 0.6896 | 2.6894 | 0.034 | 0.945 |
| label:unusually_low_solar_potential | 118147 | 103.7447 | 155.3164 | 1.7168 | 1.3916 | 3.1885 | 0.005 | 0.731 |
| label:unusually_high_solar_potential | 38401 | 43.7730 | 71.7873 | 2.0573 | 1.5567 | 4.1120 | 0.052 | 0.967 |
| label:extreme_temperature_condition | 436524 | 17.4072 | 41.0100 | 0.8582 | 0.5051 | 2.2564 | 0.043 | 0.980 |
| label:extreme_wind_condition | 458693 | 24.0708 | 53.8329 | 0.9774 | 0.1033 | 2.7788 | 0.031 | 0.964 |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

- MAE normal: 18.7909  |  MAE rare/extreme: 30.4245  |  ratio: **1.62×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty calibration

MC Dropout std is a relative uncertainty measure. Raw intervals `mean ± 1.96 std` are not guaranteed to be calibrated. Post-hoc calibration scales std with a factor estimated on a separate calibration set; the test year is used only for evaluation.

- Calibration years: 2018
- Coverage target: **0.950**
- Calibration factor: **57.1224**
- Calibration predictions: **10037664**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| global | 0.033 | 0.950 |
| normal | 0.033 | 0.951 |
| rare/extreme | 0.034 | 0.945 |

## Stratified uncertainty calibration

Global calibration uses a single factor for every test row; **group**/**label** strategies estimate separate factors on the calibration year's anomaly strata so rare/extreme bands are not under-covered. A stratum with fewer than `min_samples` calibration points falls back (label → rare/extreme group → global).

- Calibration strategy: **group**
- Calibration anomaly scores: outputs/pvgis_anomaly_2018_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Min samples per stratum: **1000**
- Global factor (k_global): **57.1224**
- Factor normal: **55.6789 (n=9207477)**
- Factor rare_or_extreme: **76.9376 (n=830187)**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| normal | 0.033 | 0.951 |
| rare/extreme | 0.034 | 0.945 |

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 18.7909  |  MAE rare/extreme: 30.4245  |  rare/normal MAE ratio: **1.62×**
- Mean uncertainty (std) normal: 0.8785  |  rare/extreme: 1.0419  |  rare/normal uncertainty ratio: **1.19×**
- Raw coverage@95 normal: 0.033  |  rare/extreme: 0.034
- Calibrated coverage@95 normal: 0.951  |  rare/extreme: 0.945

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.62×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.19×).

