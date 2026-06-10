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
- Device: cuda  |  Generated (UTC): 2026-06-09T11:40:40+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 21.8151 | 48.1209 | 0.8569 | 0.0556 | 2.4755 | 0.027 | 0.948 |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 20.6748 | 45.3795 | 0.8421 | 0.0474 | 2.4641 | 0.027 | 0.948 |
| group:rare_or_extreme | 983379 | 32.3142 | 68.3783 | 0.9935 | 0.7148 | 2.5810 | 0.027 | 0.943 |
| label:unusually_low_solar_potential | 118147 | 101.1044 | 144.8177 | 1.6284 | 1.2977 | 3.0274 | 0.001 | 0.781 |
| label:unusually_high_solar_potential | 38401 | 58.9960 | 85.5334 | 1.8995 | 1.1102 | 4.3072 | 0.023 | 0.918 |
| label:extreme_temperature_condition | 436524 | 20.2931 | 44.9205 | 0.8419 | 0.5647 | 2.1871 | 0.036 | 0.970 |
| label:extreme_wind_condition | 458693 | 25.5538 | 54.8743 | 0.9205 | 0.0836 | 2.6386 | 0.025 | 0.959 |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

- MAE normal: 20.6748  |  MAE rare/extreme: 32.3142  |  ratio: **1.56×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty calibration

**Raw MC Dropout std is an uncalibrated diagnostic score, not a predictive std.** The raw band `mean ± 1.96·std_raw` is diagnostic only — its `coverage_95_raw` is typically far below the 0.95 target and must NOT be read as a 95% predictive interval. The **primary** intervals are the calibrated ones `mean ± k·std_raw`, where k is the `coverage_target` quantile of `|y_true − y_pred_mean| / std_raw` estimated on a separate calibration set (k already absorbs the quantile — no extra 1.96 factor). The test year is used only for evaluation.

- `coverage_95_raw` = diagnostic coverage of uncalibrated MC Dropout intervals (not a nominal 95% predictive interval).
- `coverage_95_calibrated` = **primary** coverage metric (global / normal / rare_extreme).
- Calibration years: 2018
- Coverage target: **0.950**
- Calibration factor: **73.2793**
- Calibration predictions: **10037664**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| global | 0.027 | 0.948 |
| normal | 0.027 | 0.948 |
| rare/extreme | 0.027 | 0.943 |

## Stratified uncertainty calibration

Global calibration uses a single factor for every test row; **group**/**label** strategies estimate separate factors on the calibration year's anomaly strata so rare/extreme bands are not under-covered. A stratum with fewer than `min_samples` calibration points falls back (label → rare/extreme group → global).

- Calibration strategy: **group**
- Calibration anomaly scores: outputs/pvgis_anomaly_2018_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Min samples per stratum: **1000**
- Global factor (k_global): **73.2793**
- Factor normal: **72.7644 (n=9207477)**
- Factor rare_or_extreme: **80.0845 (n=830187)**

Raw MC std on the calibration set (sanity check that large k is not driven by near-zero std):
- std_raw global: min **0.005312**, max **25.77**, % < eps **0.00%** (n=10037664)
- std_raw normal: mean **0.8764**, median **0.0511** (n=9207477)
- std_raw rare/extreme: mean **0.9542**, median **0.6825** (n=830187)

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| normal | 0.027 | 0.948 |
| rare/extreme | 0.027 | 0.943 |

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 20.6748  |  MAE rare/extreme: 32.3142  |  rare/normal MAE ratio: **1.56×**
- Mean uncertainty (std) normal: 0.8421  |  rare/extreme: 0.9935  |  rare/normal uncertainty ratio: **1.18×**
- Raw coverage@95 normal: 0.027  |  rare/extreme: 0.027
- Calibrated coverage@95 normal: 0.948  |  rare/extreme: 0.943

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.56×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.18×).

## Interval reliability & sharpness (PICP / NMPIL / CLC)

Predictive-interval reliability/sharpness, inspired by uncertainty-aware rainfall prediction. **Raw intervals are diagnostic and uncalibrated; the calibrated intervals are the main predictive intervals.**

- **PICP** measures empirical coverage (fraction of y_true inside the interval).
- **NMPIL** measures normalized interval width (MPIW / target_range).
- **CLC** measures the sharpness/reliability trade-off: `CLC = NMPIL·(1 + exp(−η·(PICP − γ)))` (lower is better once PICP ≥ γ).
- γ (coverage target): **0.950**  |  η (clc_eta): **10.00**  |  target_range: **893.2700**

| stratum | PICP raw | PICP cal | MPIW raw | MPIW cal | NMPIL raw | NMPIL cal | CLC raw | CLC cal |
|---|---|---|---|---|---|---|---|---|
| global | 0.027 | 0.948 | 3.3592 | 126.1337 | 0.0038 | 0.1412 | 38.4949 | 0.2853 |
| normal | 0.027 | 0.948 | 3.3010 | 122.5503 | 0.0037 | 0.1372 | 37.8503 | 0.2766 |
| rare_extreme | 0.027 | 0.943 | 3.8945 | 159.1263 | 0.0044 | 0.1781 | 44.3945 | 0.3683 |

