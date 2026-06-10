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
- Device: cuda  |  Generated (UTC): 2026-06-09T14:27:27+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 20.9273 | 47.2046 | 0.9011 | 0.0540 | 2.7708 | 0.030 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 19.7883 | 44.3305 | 0.8889 | 0.0407 | 2.7624 | 0.030 | — |
| group:rare_or_extreme | 983379 | 31.4141 | 68.1950 | 1.0136 | 0.5299 | 2.8504 | 0.032 | — |
| label:unusually_low_solar_potential | 118147 | 98.8469 | 147.5567 | 1.5508 | 1.0660 | 3.3775 | 0.007 | — |
| label:unusually_high_solar_potential | 38401 | 61.5215 | 86.5144 | 1.6331 | 0.9075 | 4.0310 | 0.042 | — |
| label:extreme_temperature_condition | 436524 | 18.9183 | 42.7249 | 0.9002 | 0.4598 | 2.4877 | 0.041 | — |
| label:extreme_wind_condition | 458693 | 24.8466 | 53.7327 | 0.9460 | 0.0691 | 2.9092 | 0.027 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 19.7883  |  MAE rare/extreme: 31.4141  |  ratio: **1.59×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 19.7883  |  MAE rare/extreme: 31.4141  |  rare/normal MAE ratio: **1.59×**
- Mean uncertainty (std) normal: 0.8889  |  rare/extreme: 1.0136  |  rare/normal uncertainty ratio: **1.14×**
- Gaussian coverage@95 (diagnostic) normal: 0.030  |  rare/extreme: 0.032
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.59×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.14×).

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
| global | 0.026 | 3.0867 | 0.0035 | 35.5609 |
| normal | 0.026 | 3.0449 | 0.0034 | 35.1365 |
| rare_extreme | 0.028 | 3.4719 | 0.0039 | 39.3983 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.030 | 3.5324 | 0.0040 | 39.1989 |
| normal | 0.030 | 3.4845 | 0.0039 | 38.7402 |
| rare_extreme | 0.032 | 3.9732 | 0.0044 | 43.3361 |

