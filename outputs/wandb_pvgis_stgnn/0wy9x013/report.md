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
- Train years: 2016,2017,2018
- Test year: **2019**
- Nodes (locations): **1149**  |  epochs: **5**
- batch_size: 16  |  lr: 0.001
- Anomaly scores: outputs/pvgis_anomaly_2019_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Predictions: **10037664**
- Device: cuda  |  Generated (UTC): 2026-06-10T09:28:45+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 20.7338 | 46.2352 | 0.8521 | 0.1623 | 2.3818 | 0.026 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 19.7830 | 43.7756 | 0.8369 | 0.1453 | 2.3641 | 0.026 | — |
| group:rare_or_extreme | 983379 | 29.4889 | 64.6223 | 0.9919 | 0.6705 | 2.5475 | 0.028 | — |
| label:unusually_low_solar_potential | 118147 | 89.2367 | 137.2233 | 1.6838 | 1.2133 | 3.4548 | 0.003 | — |
| label:unusually_high_solar_potential | 38401 | 52.9983 | 80.5378 | 1.8615 | 1.3561 | 3.6603 | 0.025 | — |
| label:extreme_temperature_condition | 436524 | 17.5976 | 39.3320 | 0.8138 | 0.4222 | 2.0339 | 0.038 | — |
| label:extreme_wind_condition | 458693 | 24.2230 | 52.8632 | 0.9332 | 0.2132 | 2.5992 | 0.026 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 19.7830  |  MAE rare/extreme: 29.4889  |  ratio: **1.49×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 19.7830  |  MAE rare/extreme: 29.4889  |  rare/normal MAE ratio: **1.49×**
- Mean uncertainty (std) normal: 0.8369  |  rare/extreme: 0.9919  |  rare/normal uncertainty ratio: **1.19×**
- Gaussian coverage@95 (diagnostic) normal: 0.026  |  rare/extreme: 0.028
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.49×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.19×).

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
| global | 0.023 | 2.9205 | 0.0033 | 34.6452 |
| normal | 0.023 | 2.8686 | 0.0032 | 34.0948 |
| rare_extreme | 0.025 | 3.3987 | 0.0038 | 39.6087 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.026 | 3.3404 | 0.0037 | 38.3537 |
| normal | 0.026 | 3.2808 | 0.0037 | 37.7514 |
| rare_extreme | 0.028 | 3.8884 | 0.0044 | 43.7707 |

