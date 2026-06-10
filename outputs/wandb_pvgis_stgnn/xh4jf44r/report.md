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
- Device: cuda  |  Generated (UTC): 2026-06-09T11:14:31+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 21.7064 | 47.7481 | 0.7846 | 0.0385 | 2.3679 | 0.021 | 0.949 |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 20.5812 | 44.9788 | 0.7705 | 0.0348 | 2.3574 | 0.021 | 0.951 |
| group:rare_or_extreme | 983379 | 32.0665 | 68.1487 | 0.9148 | 0.4960 | 2.4641 | 0.022 | 0.936 |
| label:unusually_low_solar_potential | 118147 | 98.6574 | 144.8453 | 1.7431 | 1.5087 | 3.1841 | 0.011 | 0.831 |
| label:unusually_high_solar_potential | 38401 | 61.1441 | 85.7000 | 1.4485 | 0.8768 | 3.3339 | 0.013 | 0.722 |
| label:extreme_temperature_condition | 436524 | 20.0771 | 44.1886 | 0.7270 | 0.3894 | 2.0209 | 0.026 | 0.959 |
| label:extreme_wind_condition | 458693 | 25.5869 | 54.7651 | 0.8562 | 0.0451 | 2.5281 | 0.020 | 0.956 |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

- MAE normal: 20.5812  |  MAE rare/extreme: 32.0665  |  ratio: **1.56×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty calibration

**Raw MC Dropout std is an uncalibrated diagnostic score, not a predictive std.** The raw band `mean ± 1.96·std_raw` is diagnostic only — its `coverage_95_raw` is typically far below the 0.95 target and must NOT be read as a 95% predictive interval. The **primary** intervals are the calibrated ones `mean ± k·std_raw`, where k is the `coverage_target` quantile of `|y_true − y_pred_mean| / std_raw` estimated on a separate calibration set (k already absorbs the quantile — no extra 1.96 factor). The test year is used only for evaluation.

- `coverage_95_raw` = diagnostic coverage of uncalibrated MC Dropout intervals (not a nominal 95% predictive interval).
- `coverage_95_calibrated` = **primary** coverage metric (global / normal / rare_extreme).
- Calibration years: 2018
- Coverage target: **0.950**
- Calibration factor: **82.1479**
- Calibration predictions: **10037664**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| global | 0.021 | 0.949 |
| normal | 0.021 | 0.951 |
| rare/extreme | 0.022 | 0.936 |

## Stratified uncertainty calibration

Global calibration uses a single factor for every test row; **group**/**label** strategies estimate separate factors on the calibration year's anomaly strata so rare/extreme bands are not under-covered. A stratum with fewer than `min_samples` calibration points falls back (label → rare/extreme group → global).

- Calibration strategy: **group**
- Calibration anomaly scores: outputs/pvgis_anomaly_2018_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Min samples per stratum: **1000**
- Global factor (k_global): **82.1479**
- Factor normal: **81.2482 (n=9207477)**
- Factor rare_or_extreme: **91.7265 (n=830187)**

Raw MC std on the calibration set (sanity check that large k is not driven by near-zero std):
- std_raw global: min **0.002521**, max **26.53**, % < eps **0.00%** (n=10037664)
- std_raw normal: mean **0.7991**, median **0.03636** (n=9207477)
- std_raw rare/extreme: mean **0.8982**, median **0.4125** (n=830187)

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| normal | 0.021 | 0.951 |
| rare/extreme | 0.022 | 0.936 |

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 20.5812  |  MAE rare/extreme: 32.0665  |  rare/normal MAE ratio: **1.56×**
- Mean uncertainty (std) normal: 0.7705  |  rare/extreme: 0.9148  |  rare/normal uncertainty ratio: **1.19×**
- Raw coverage@95 normal: 0.021  |  rare/extreme: 0.022
- Calibrated coverage@95 normal: 0.951  |  rare/extreme: 0.936

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.56×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.19×).

## Interval reliability & sharpness (PICP / NMPIL / CLC)

Predictive-interval reliability/sharpness, inspired by uncertainty-aware rainfall prediction. **Raw intervals are diagnostic and uncalibrated; the calibrated intervals are the main predictive intervals.**

- **PICP** measures empirical coverage (fraction of y_true inside the interval).
- **NMPIL** measures normalized interval width (MPIW / target_range).
- **CLC** measures the sharpness/reliability trade-off: `CLC = NMPIL·(1 + exp(−η·(PICP − γ)))` (lower is better once PICP ≥ γ).
- γ (coverage target): **0.950**  |  η (clc_eta): **10.00**  |  target_range: **893.2700**

| stratum | PICP raw | PICP cal | MPIW raw | MPIW cal | NMPIL raw | NMPIL cal | CLC raw | CLC cal |
|---|---|---|---|---|---|---|---|---|
| global | 0.021 | 0.949 | 3.0758 | 129.3786 | 0.0034 | 0.1448 | 37.3535 | 0.2908 |
| normal | 0.021 | 0.951 | 3.0204 | 125.2038 | 0.0034 | 0.1402 | 36.7085 | 0.2794 |
| rare_extreme | 0.022 | 0.936 | 3.5859 | 167.8180 | 0.0040 | 0.1879 | 43.2444 | 0.4050 |

