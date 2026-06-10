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
- Device: cuda  |  Generated (UTC): 2026-06-09T12:06:46+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 19.9405 | 46.4496 | 0.8931 | 0.0764 | 2.5701 | 0.033 | 0.950 |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 18.7956 | 43.2510 | 0.8769 | 0.0645 | 2.5579 | 0.033 | 0.951 |
| group:rare_or_extreme | 983379 | 30.4812 | 69.2770 | 1.0419 | 0.6925 | 2.6845 | 0.034 | 0.946 |
| label:unusually_low_solar_potential | 118147 | 104.0303 | 155.5846 | 1.7222 | 1.4010 | 3.1836 | 0.005 | 0.732 |
| label:unusually_high_solar_potential | 38401 | 44.0881 | 71.9946 | 2.0660 | 1.5846 | 4.0980 | 0.052 | 0.967 |
| label:extreme_temperature_condition | 436524 | 17.4042 | 41.0071 | 0.8595 | 0.5164 | 2.2576 | 0.043 | 0.981 |
| label:extreme_wind_condition | 458693 | 24.1037 | 53.8882 | 0.9763 | 0.1039 | 2.7738 | 0.031 | 0.964 |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

- MAE normal: 18.7956  |  MAE rare/extreme: 30.4812  |  ratio: **1.62×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty calibration

**Raw MC Dropout std is an uncalibrated diagnostic score, not a predictive std.** The raw band `mean ± 1.96·std_raw` is diagnostic only — its `coverage_95_raw` is typically far below the 0.95 target and must NOT be read as a 95% predictive interval. The **primary** intervals are the calibrated ones `mean ± k·std_raw`, where k is the `coverage_target` quantile of `|y_true − y_pred_mean| / std_raw` estimated on a separate calibration set (k already absorbs the quantile — no extra 1.96 factor). The test year is used only for evaluation.

- `coverage_95_raw` = diagnostic coverage of uncalibrated MC Dropout intervals (not a nominal 95% predictive interval).
- `coverage_95_calibrated` = **primary** coverage metric (global / normal / rare_extreme).
- Calibration years: 2018
- Coverage target: **0.950**
- Calibration factor: **57.8773**
- Calibration predictions: **10037664**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| global | 0.033 | 0.950 |
| normal | 0.033 | 0.951 |
| rare/extreme | 0.034 | 0.946 |

## Stratified uncertainty calibration

Global calibration uses a single factor for every test row; **group**/**label** strategies estimate separate factors on the calibration year's anomaly strata so rare/extreme bands are not under-covered. A stratum with fewer than `min_samples` calibration points falls back (label → rare/extreme group → global).

- Calibration strategy: **group**
- Calibration anomaly scores: outputs/pvgis_anomaly_2018_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Min samples per stratum: **1000**
- Global factor (k_global): **57.8773**
- Factor normal: **56.6468 (n=9207477)**
- Factor rare_or_extreme: **76.4897 (n=830187)**

Raw MC std on the calibration set (sanity check that large k is not driven by near-zero std):
- std_raw global: min **0.006845**, max **34.94**, % < eps **0.00%** (n=10037664)
- std_raw normal: mean **0.9085**, median **0.06984** (n=9207477)
- std_raw rare/extreme: mean **1.014**, median **0.632** (n=830187)

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| normal | 0.033 | 0.951 |
| rare/extreme | 0.034 | 0.946 |

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 18.7956  |  MAE rare/extreme: 30.4812  |  rare/normal MAE ratio: **1.62×**
- Mean uncertainty (std) normal: 0.8769  |  rare/extreme: 1.0419  |  rare/normal uncertainty ratio: **1.19×**
- Raw coverage@95 normal: 0.033  |  rare/extreme: 0.034
- Calibrated coverage@95 normal: 0.951  |  rare/extreme: 0.946

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.62×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.19×).

## Interval reliability & sharpness (PICP / NMPIL / CLC)

Predictive-interval reliability/sharpness, inspired by uncertainty-aware rainfall prediction. **Raw intervals are diagnostic and uncalibrated; the calibrated intervals are the main predictive intervals.**

- **PICP** measures empirical coverage (fraction of y_true inside the interval).
- **NMPIL** measures normalized interval width (MPIW / target_range).
- **CLC** measures the sharpness/reliability trade-off: `CLC = NMPIL·(1 + exp(−η·(PICP − γ)))` (lower is better once PICP ≥ γ).
- γ (coverage target): **0.950**  |  η (clc_eta): **10.00**  |  target_range: **893.2700**

| stratum | PICP raw | PICP cal | MPIW raw | MPIW cal | NMPIL raw | NMPIL cal | CLC raw | CLC cal |
|---|---|---|---|---|---|---|---|---|
| global | 0.033 | 0.950 | 3.5008 | 105.2279 | 0.0039 | 0.1178 | 37.5411 | 0.2354 |
| normal | 0.033 | 0.951 | 3.4374 | 99.3459 | 0.0038 | 0.1112 | 36.9003 | 0.2216 |
| rare_extreme | 0.034 | 0.946 | 4.0841 | 159.3852 | 0.0046 | 0.1784 | 43.3765 | 0.3651 |

