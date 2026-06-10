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
- Device: cuda  |  Generated (UTC): 2026-06-09T15:14:08+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 20.9775 | 46.1805 | 0.8936 | 0.2630 | 2.4218 | 0.027 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 20.0276 | 43.6986 | 0.8787 | 0.2404 | 2.4032 | 0.027 | — |
| group:rare_or_extreme | 983379 | 29.7239 | 64.7032 | 1.0304 | 0.6839 | 2.5921 | 0.030 | — |
| label:unusually_low_solar_potential | 118147 | 90.2401 | 138.0547 | 1.7043 | 1.2161 | 3.4974 | 0.002 | — |
| label:unusually_high_solar_potential | 38401 | 51.1089 | 78.1008 | 1.8990 | 1.4033 | 3.6798 | 0.029 | — |
| label:extreme_temperature_condition | 436524 | 17.8296 | 39.3560 | 0.8582 | 0.4705 | 2.0945 | 0.041 | — |
| label:extreme_wind_condition | 458693 | 24.4996 | 52.9015 | 0.9743 | 0.3155 | 2.6474 | 0.026 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 20.0276  |  MAE rare/extreme: 29.7239  |  ratio: **1.48×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 20.0276  |  MAE rare/extreme: 29.7239  |  rare/normal MAE ratio: **1.48×**
- Mean uncertainty (std) normal: 0.8787  |  rare/extreme: 1.0304  |  rare/normal uncertainty ratio: **1.17×**
- Gaussian coverage@95 (diagnostic) normal: 0.027  |  rare/extreme: 0.030
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.48×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.17×).

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
| global | 0.024 | 3.0625 | 0.0034 | 36.0711 |
| normal | 0.024 | 3.0116 | 0.0034 | 35.5604 |
| rare_extreme | 0.026 | 3.5305 | 0.0040 | 40.6458 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.027 | 3.5029 | 0.0039 | 39.8675 |
| normal | 0.027 | 3.4446 | 0.0039 | 39.3081 |
| rare_extreme | 0.030 | 4.0393 | 0.0045 | 44.8685 |

