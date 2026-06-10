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
- Device: cuda  |  Generated (UTC): 2026-06-10T08:17:46+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 20.0907 | 46.1424 | 0.8997 | 0.0701 | 2.6073 | 0.032 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 18.9797 | 43.3281 | 0.8842 | 0.0599 | 2.5937 | 0.032 | — |
| group:rare_or_extreme | 983379 | 30.3202 | 66.6892 | 1.0426 | 0.7214 | 2.7343 | 0.031 | — |
| label:unusually_low_solar_potential | 118147 | 98.9084 | 145.1368 | 1.8088 | 1.3841 | 3.6173 | 0.002 | — |
| label:unusually_high_solar_potential | 38401 | 51.8064 | 74.9081 | 1.7076 | 1.2265 | 3.5189 | 0.023 | — |
| label:extreme_temperature_condition | 436524 | 18.0033 | 40.6744 | 0.8733 | 0.5468 | 2.2176 | 0.040 | — |
| label:extreme_wind_condition | 458693 | 23.5548 | 53.1530 | 0.9742 | 0.0835 | 2.8517 | 0.031 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 18.9797  |  MAE rare/extreme: 30.3202  |  ratio: **1.60×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 18.9797  |  MAE rare/extreme: 30.3202  |  rare/normal MAE ratio: **1.60×**
- Mean uncertainty (std) normal: 0.8842  |  rare/extreme: 1.0426  |  rare/normal uncertainty ratio: **1.18×**
- Gaussian coverage@95 (diagnostic) normal: 0.032  |  rare/extreme: 0.031
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.60×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.18×).

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
| global | 0.028 | 3.0833 | 0.0035 | 34.7819 |
| normal | 0.028 | 3.0301 | 0.0034 | 34.1516 |
| rare_extreme | 0.027 | 3.5730 | 0.0040 | 40.6369 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.032 | 3.5269 | 0.0039 | 38.2001 |
| normal | 0.032 | 3.4661 | 0.0039 | 37.5061 |
| rare_extreme | 0.031 | 4.0871 | 0.0046 | 44.6496 |

