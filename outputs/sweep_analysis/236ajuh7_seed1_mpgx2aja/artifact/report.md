# PVGIS-only ST-GNN forecasting report

Reuses the existing STGNN architecture on a **PVGIS-only** input. No real plant production, no ENERGIA, no quality score, no kWp/UPN. Anomaly labels are used **only** for stratified evaluation.

## Experiment

- Mode: **pvgis_stgnn**
- Model type: **stgnn_enhanced_dropout**
- Feature set: **full**
- W&B enabled: **True**
- MC Dropout: **enabled** (experimental)

## Parameters

- Target variable: **pv_power_output**
- Selected features (11): temperature_2m, solar_irradiance_poa, wind_speed_10m, sin_elev, cos_elev, kt, kt_std_3h, dghi_dt, dni_norm, dhi_norm, pv_lag_pvgis
- MC samples: **20**
- seq_len: **24**  |  horizon: **1**
- Train years: 2016,2017,2018
- Test year: **2019**
- Nodes (locations): **1149**  |  epochs: **5**
- batch_size: 16  |  lr: 0.001
- Anomaly scores: outputs/pvgis_anomaly_2019_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Predictions: **10037664**
- Device: cuda  |  Generated (UTC): 2026-06-10T15:28:04+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 20.4104 | 46.2350 | 7.2214 | 2.4795 | 17.5130 | 0.759 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 19.3839 | 43.4713 | 7.0674 | 2.3337 | 17.4044 | 0.767 | — |
| group:rare_or_extreme | 983379 | 29.8615 | 66.4852 | 8.6390 | 8.9342 | 18.3919 | 0.684 | — |
| label:unusually_low_solar_potential | 118147 | 92.6164 | 144.1204 | 13.9547 | 13.9220 | 19.7848 | 0.182 | — |
| label:unusually_high_solar_potential | 38401 | 50.3756 | 73.6777 | 16.6398 | 16.0577 | 25.4243 | 0.426 | — |
| label:extreme_temperature_condition | 436524 | 18.2407 | 40.6447 | 8.0386 | 7.9465 | 17.6397 | 0.775 | — |
| label:extreme_wind_condition | 458693 | 23.6898 | 53.3127 | 7.4217 | 2.6696 | 17.7122 | 0.743 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 19.3839  |  MAE rare/extreme: 29.8615  |  ratio: **1.54×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 19.3839  |  MAE rare/extreme: 29.8615  |  rare/normal MAE ratio: **1.54×**
- Mean uncertainty (std) normal: 7.0674  |  rare/extreme: 8.6390  |  rare/normal uncertainty ratio: **1.22×**
- Gaussian coverage@95 (diagnostic) normal: 0.767  |  rare/extreme: 0.684
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.54×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.22×).

## Interval reliability & sharpness (PICP / NMPIL / CLC)

Paper-style evaluation (uncertainty-aware rainfall prediction). The **primary predictive intervals (`pi`) are built directly from the MC Dropout sample distribution** (empirical quantiles q(alpha/2), q(1-alpha/2)); **no post-hoc calibration is used in the main protocol**. The Gaussian band (`gaussian`, mean ± 1.96·std_raw) is a secondary diagnostic only.

- **PICP** measures empirical coverage (fraction of y_true inside the interval). It is **evaluated, not forced** to 0.95 — no factor is fit to hit the target in the main protocol.
- **NMPIL** measures normalized interval width (MPIW / target_range).
- **CLC** measures the sharpness/reliability trade-off: `CLC = NMPIL·(1 + exp(-eta·(PICP - gamma)))` (lower is better once PICP >= gamma).
- A very low PICP for `pi` means raw MC Dropout is sharp but **not reliable** in this PVGIS-only setting.
- gamma (coverage target): **0.950**  |  eta (clc_eta): **10.00**  |  target_range: **893.2700**

### PI (primary, MC quantiles)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.216 | 24.5362 | 0.0275 | 42.3880 |
| normal | 0.214 | 24.0068 | 0.0269 | 42.2519 |
| rare_extreme | 0.233 | 29.4104 | 0.0329 | 42.8137 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.759 | 28.3078 | 0.0317 | 0.2464 |
| normal | 0.767 | 27.7043 | 0.0310 | 0.2248 |
| rare_extreme | 0.684 | 33.8648 | 0.0379 | 0.5792 |

