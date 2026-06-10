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
- Device: cuda  |  Generated (UTC): 2026-06-09T12:59:00+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 19.9590 | 46.5049 | 0.8900 | 0.0321 | 2.6757 | 0.031 | 0.949 |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 18.9050 | 43.6246 | 0.8739 | 0.0267 | 2.6628 | 0.031 | 0.950 |
| group:rare_or_extreme | 983379 | 29.6626 | 67.4759 | 1.0381 | 0.6337 | 2.7933 | 0.031 | 0.943 |
| label:unusually_low_solar_potential | 118147 | 95.7755 | 147.8272 | 1.8997 | 1.5684 | 3.5167 | 0.007 | 0.862 |
| label:unusually_high_solar_potential | 38401 | 47.5893 | 74.9155 | 1.7369 | 1.1398 | 3.7353 | 0.036 | 0.954 |
| label:extreme_temperature_condition | 436524 | 17.6148 | 41.2330 | 0.8549 | 0.4529 | 2.3449 | 0.040 | 0.959 |
| label:extreme_wind_condition | 458693 | 23.7243 | 53.5723 | 0.9588 | 0.0447 | 2.8361 | 0.029 | 0.947 |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

- MAE normal: 18.9050  |  MAE rare/extreme: 29.6626  |  ratio: **1.57×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty calibration

**Raw MC Dropout std is an uncalibrated diagnostic score, not a predictive std.** The raw band `mean ± 1.96·std_raw` is diagnostic only — its `coverage_95_raw` is typically far below the 0.95 target and must NOT be read as a 95% predictive interval. The **primary** intervals are the calibrated ones `mean ± k·std_raw`, where k is the `coverage_target` quantile of `|y_true − y_pred_mean| / std_raw` estimated on a separate calibration set (k already absorbs the quantile — no extra 1.96 factor). The test year is used only for evaluation.

- `coverage_95_raw` = diagnostic coverage of uncalibrated MC Dropout intervals (not a nominal 95% predictive interval).
- `coverage_95_calibrated` = **primary** coverage metric (global / normal / rare_extreme).
- Calibration years: 2018
- Coverage target: **0.950**
- Calibration factor: **80.8427**
- Calibration predictions: **10037664**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| global | 0.031 | 0.949 |
| normal | 0.031 | 0.950 |
| rare/extreme | 0.031 | 0.943 |

## Stratified uncertainty calibration

Global calibration uses a single factor for every test row; **group**/**label** strategies estimate separate factors on the calibration year's anomaly strata so rare/extreme bands are not under-covered. A stratum with fewer than `min_samples` calibration points falls back (label → rare/extreme group → global).

- Calibration strategy: **group**
- Calibration anomaly scores: outputs/pvgis_anomaly_2018_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Min samples per stratum: **1000**
- Global factor (k_global): **80.8427**
- Factor normal: **80.6945 (n=9207477)**
- Factor rare_or_extreme: **82.7812 (n=830187)**

Raw MC std on the calibration set (sanity check that large k is not driven by near-zero std):
- std_raw global: min **0.003296**, max **26.9**, % < eps **0.00%** (n=10037664)
- std_raw normal: mean **0.9156**, median **0.02992** (n=9207477)
- std_raw rare/extreme: mean **1.035**, median **0.5628** (n=830187)

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| normal | 0.031 | 0.950 |
| rare/extreme | 0.031 | 0.943 |

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 18.9050  |  MAE rare/extreme: 29.6626  |  rare/normal MAE ratio: **1.57×**
- Mean uncertainty (std) normal: 0.8739  |  rare/extreme: 1.0381  |  rare/normal uncertainty ratio: **1.19×**
- Raw coverage@95 normal: 0.031  |  rare/extreme: 0.031
- Calibrated coverage@95 normal: 0.950  |  rare/extreme: 0.943

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.57×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.19×).

## Interval reliability & sharpness (PICP / NMPIL / CLC)

Predictive-interval reliability/sharpness, inspired by uncertainty-aware rainfall prediction. **Raw intervals are diagnostic and uncalibrated; the calibrated intervals are the main predictive intervals.**

- **PICP** measures empirical coverage (fraction of y_true inside the interval).
- **NMPIL** measures normalized interval width (MPIW / target_range).
- **CLC** measures the sharpness/reliability trade-off: `CLC = NMPIL·(1 + exp(−η·(PICP − γ)))` (lower is better once PICP ≥ γ).
- γ (coverage target): **0.950**  |  η (clc_eta): **10.00**  |  target_range: **893.2700**

| stratum | PICP raw | PICP cal | MPIW raw | MPIW cal | NMPIL raw | NMPIL cal | CLC raw | CLC cal |
|---|---|---|---|---|---|---|---|---|
| global | 0.031 | 0.949 | 3.4889 | 144.0637 | 0.0039 | 0.1613 | 38.1447 | 0.3241 |
| normal | 0.031 | 0.950 | 3.4258 | 141.0444 | 0.0038 | 0.1579 | 37.4603 | 0.3163 |
| rare_extreme | 0.031 | 0.943 | 4.0692 | 171.8628 | 0.0046 | 0.1924 | 44.4385 | 0.3985 |

