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
- Device: cuda  |  Generated (UTC): 2026-06-09T14:04:11+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 19.9366 | 45.9243 | 0.9307 | 0.0755 | 2.6696 | 0.034 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 18.8458 | 43.1038 | 0.9129 | 0.0641 | 2.6544 | 0.034 | — |
| group:rare_or_extreme | 983379 | 29.9796 | 66.4909 | 1.0942 | 0.8031 | 2.8053 | 0.034 | — |
| label:unusually_low_solar_potential | 118147 | 98.1468 | 145.3967 | 1.8459 | 1.4039 | 3.7372 | 0.001 | — |
| label:unusually_high_solar_potential | 38401 | 48.7469 | 72.7353 | 2.1017 | 1.7952 | 3.6404 | 0.039 | — |
| label:extreme_temperature_condition | 436524 | 17.8337 | 40.1415 | 0.9199 | 0.6817 | 2.2941 | 0.043 | — |
| label:extreme_wind_condition | 458693 | 23.3854 | 53.0302 | 1.0057 | 0.0885 | 2.9004 | 0.033 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 18.8458  |  MAE rare/extreme: 29.9796  |  ratio: **1.59×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 18.8458  |  MAE rare/extreme: 29.9796  |  rare/normal MAE ratio: **1.59×**
- Mean uncertainty (std) normal: 0.9129  |  rare/extreme: 1.0942  |  rare/normal uncertainty ratio: **1.20×**
- Gaussian coverage@95 (diagnostic) normal: 0.034  |  rare/extreme: 0.034
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.59×).
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
| global | 0.030 | 3.1897 | 0.0036 | 35.4940 |
| normal | 0.030 | 3.1288 | 0.0035 | 34.8246 |
| rare_extreme | 0.030 | 3.7506 | 0.0042 | 41.6424 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.034 | 3.6483 | 0.0041 | 38.8922 |
| normal | 0.034 | 3.5787 | 0.0040 | 38.1528 |
| rare_extreme | 0.034 | 4.2894 | 0.0048 | 45.6955 |

