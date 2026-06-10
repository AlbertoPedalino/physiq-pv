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
- Device: cuda  |  Generated (UTC): 2026-06-09T12:32:51+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 21.1016 | 47.9283 | 0.8734 | 0.0499 | 2.5973 | 0.029 | 0.947 |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 19.8319 | 44.5020 | 0.8593 | 0.0387 | 2.5861 | 0.030 | 0.948 |
| group:rare_or_extreme | 983379 | 32.7918 | 72.2014 | 1.0036 | 0.6414 | 2.7024 | 0.028 | 0.945 |
| label:unusually_low_solar_potential | 118147 | 113.8368 | 164.0873 | 1.6458 | 1.2473 | 3.2305 | 0.001 | 0.749 |
| label:unusually_high_solar_potential | 38401 | 34.3677 | 63.2567 | 2.0987 | 1.6405 | 4.0086 | 0.093 | 0.988 |
| label:extreme_temperature_condition | 436524 | 19.7914 | 43.2885 | 0.8325 | 0.5189 | 2.2529 | 0.031 | 0.971 |
| label:extreme_wind_condition | 458693 | 25.2295 | 55.1592 | 0.9274 | 0.0896 | 2.7618 | 0.026 | 0.967 |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

- MAE normal: 19.8319  |  MAE rare/extreme: 32.7918  |  ratio: **1.65×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty calibration

**Raw MC Dropout std is an uncalibrated diagnostic score, not a predictive std.** The raw band `mean ± 1.96·std_raw` is diagnostic only — its `coverage_95_raw` is typically far below the 0.95 target and must NOT be read as a 95% predictive interval. The **primary** intervals are the calibrated ones `mean ± k·std_raw`, where k is the `coverage_target` quantile of `|y_true − y_pred_mean| / std_raw` estimated on a separate calibration set (k already absorbs the quantile — no extra 1.96 factor). The test year is used only for evaluation.

- `coverage_95_raw` = diagnostic coverage of uncalibrated MC Dropout intervals (not a nominal 95% predictive interval).
- `coverage_95_calibrated` = **primary** coverage metric (global / normal / rare_extreme).
- Calibration years: 2018
- Coverage target: **0.950**
- Calibration factor: **79.8284**
- Calibration predictions: **10037664**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| global | 0.029 | 0.947 |
| normal | 0.030 | 0.948 |
| rare/extreme | 0.028 | 0.945 |

## Stratified uncertainty calibration

Global calibration uses a single factor for every test row; **group**/**label** strategies estimate separate factors on the calibration year's anomaly strata so rare/extreme bands are not under-covered. A stratum with fewer than `min_samples` calibration points falls back (label → rare/extreme group → global).

- Calibration strategy: **group**
- Calibration anomaly scores: outputs/pvgis_anomaly_2018_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Min samples per stratum: **1000**
- Global factor (k_global): **79.8284**
- Factor normal: **78.7723 (n=9207477)**
- Factor rare_or_extreme: **93.3565 (n=830187)**

Raw MC std on the calibration set (sanity check that large k is not driven by near-zero std):
- std_raw global: min **0.004257**, max **39.44**, % < eps **0.00%** (n=10037664)
- std_raw normal: mean **0.8984**, median **0.04371** (n=9207477)
- std_raw rare/extreme: mean **0.9908**, median **0.5855** (n=830187)

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| normal | 0.030 | 0.948 |
| rare/extreme | 0.028 | 0.945 |

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 19.8319  |  MAE rare/extreme: 32.7918  |  rare/normal MAE ratio: **1.65×**
- Mean uncertainty (std) normal: 0.8593  |  rare/extreme: 1.0036  |  rare/normal uncertainty ratio: **1.17×**
- Raw coverage@95 normal: 0.030  |  rare/extreme: 0.028
- Calibrated coverage@95 normal: 0.948  |  rare/extreme: 0.945

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.65×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.17×).

## Interval reliability & sharpness (PICP / NMPIL / CLC)

Predictive-interval reliability/sharpness, inspired by uncertainty-aware rainfall prediction. **Raw intervals are diagnostic and uncalibrated; the calibrated intervals are the main predictive intervals.**

- **PICP** measures empirical coverage (fraction of y_true inside the interval).
- **NMPIL** measures normalized interval width (MPIW / target_range).
- **CLC** measures the sharpness/reliability trade-off: `CLC = NMPIL·(1 + exp(−η·(PICP − γ)))` (lower is better once PICP ≥ γ).
- γ (coverage target): **0.950**  |  η (clc_eta): **10.00**  |  target_range: **893.2700**

| stratum | PICP raw | PICP cal | MPIW raw | MPIW cal | NMPIL raw | NMPIL cal | CLC raw | CLC cal |
|---|---|---|---|---|---|---|---|---|
| global | 0.029 | 0.947 | 3.4238 | 140.4686 | 0.0038 | 0.1573 | 38.1742 | 0.3188 |
| normal | 0.030 | 0.948 | 3.3683 | 135.3727 | 0.0038 | 0.1515 | 37.4969 | 0.3069 |
| rare_extreme | 0.028 | 0.945 | 3.9342 | 187.3884 | 0.0044 | 0.2098 | 44.5068 | 0.4297 |

