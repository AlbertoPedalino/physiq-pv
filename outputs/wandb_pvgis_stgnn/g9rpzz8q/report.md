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
- Device: cuda  |  Generated (UTC): 2026-06-10T09:05:06+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 19.8623 | 45.7355 | 0.9017 | 0.1065 | 2.5078 | 0.032 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 18.6339 | 42.5283 | 0.8846 | 0.0785 | 2.4907 | 0.032 | — |
| group:rare_or_extreme | 983379 | 31.1720 | 68.5431 | 1.0596 | 0.7577 | 2.6620 | 0.031 | — |
| label:unusually_low_solar_potential | 118147 | 110.9823 | 155.4652 | 1.9085 | 1.6204 | 3.4497 | 0.001 | — |
| label:unusually_high_solar_potential | 38401 | 36.2998 | 65.3544 | 1.8521 | 1.1751 | 3.9233 | 0.071 | — |
| label:extreme_temperature_condition | 436524 | 17.9711 | 40.1321 | 0.8825 | 0.6246 | 2.2279 | 0.038 | — |
| label:extreme_wind_condition | 458693 | 24.3043 | 53.3613 | 0.9750 | 0.1457 | 2.7033 | 0.029 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 18.6339  |  MAE rare/extreme: 31.1720  |  ratio: **1.67×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 18.6339  |  MAE rare/extreme: 31.1720  |  rare/normal MAE ratio: **1.67×**
- Mean uncertainty (std) normal: 0.8846  |  rare/extreme: 1.0596  |  rare/normal uncertainty ratio: **1.20×**
- Gaussian coverage@95 (diagnostic) normal: 0.032  |  rare/extreme: 0.031
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.67×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.20×).

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
| global | 0.028 | 3.0875 | 0.0035 | 34.8567 |
| normal | 0.028 | 3.0289 | 0.0034 | 34.1699 |
| rare_extreme | 0.027 | 3.6273 | 0.0041 | 41.2268 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.032 | 3.5348 | 0.0040 | 38.3289 |
| normal | 0.032 | 3.4676 | 0.0039 | 37.5718 |
| rare_extreme | 0.031 | 4.1536 | 0.0046 | 45.3524 |

