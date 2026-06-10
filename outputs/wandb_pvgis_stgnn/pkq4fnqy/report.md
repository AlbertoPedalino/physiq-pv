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
- Device: cuda  |  Generated (UTC): 2026-06-09T16:21:06+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 19.4504 | 45.8219 | 0.8321 | 0.0249 | 2.6348 | 0.033 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 18.2725 | 42.6702 | 0.8213 | 0.0197 | 2.6270 | 0.033 | — |
| group:rare_or_extreme | 983379 | 30.2963 | 68.3196 | 0.9316 | 0.4437 | 2.7052 | 0.033 | — |
| label:unusually_low_solar_potential | 118147 | 100.7598 | 151.8092 | 1.6067 | 1.1037 | 3.5070 | 0.003 | — |
| label:unusually_high_solar_potential | 38401 | 63.4361 | 85.3180 | 1.2852 | 0.6295 | 3.2422 | 0.034 | — |
| label:extreme_temperature_condition | 436524 | 16.7141 | 39.7629 | 0.7829 | 0.2539 | 2.2377 | 0.044 | — |
| label:extreme_wind_condition | 458693 | 23.4551 | 52.9776 | 0.8935 | 0.0397 | 2.8167 | 0.029 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 18.2725  |  MAE rare/extreme: 30.2963  |  ratio: **1.66×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 18.2725  |  MAE rare/extreme: 30.2963  |  rare/normal MAE ratio: **1.66×**
- Mean uncertainty (std) normal: 0.8213  |  rare/extreme: 0.9316  |  rare/normal uncertainty ratio: **1.13×**
- Gaussian coverage@95 (diagnostic) normal: 0.033  |  rare/extreme: 0.033
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.66×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.13×).

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
| global | 0.029 | 2.8504 | 0.0032 | 32.0549 |
| normal | 0.028 | 2.8134 | 0.0031 | 31.6517 |
| rare_extreme | 0.029 | 3.1913 | 0.0036 | 35.7516 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.033 | 3.2617 | 0.0037 | 35.2004 |
| normal | 0.033 | 3.2193 | 0.0036 | 34.7552 |
| rare_extreme | 0.033 | 3.6518 | 0.0041 | 39.2847 |

