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
- Device: cuda  |  Generated (UTC): 2026-06-10T08:41:28+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 19.3908 | 45.7061 | 0.8586 | 0.0287 | 2.6736 | 0.034 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 18.2222 | 42.5739 | 0.8474 | 0.0228 | 2.6657 | 0.034 | — |
| group:rare_or_extreme | 983379 | 30.1501 | 68.0807 | 0.9616 | 0.4771 | 2.7481 | 0.034 | — |
| label:unusually_low_solar_potential | 118147 | 102.3015 | 152.0803 | 1.6603 | 1.1729 | 3.5388 | 0.003 | — |
| label:unusually_high_solar_potential | 38401 | 58.3353 | 80.7132 | 1.3566 | 0.6710 | 3.4354 | 0.036 | — |
| label:extreme_temperature_condition | 436524 | 16.4443 | 39.4552 | 0.8099 | 0.2826 | 2.2778 | 0.045 | — |
| label:extreme_wind_condition | 458693 | 23.4148 | 52.9042 | 0.9181 | 0.0410 | 2.8575 | 0.030 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 18.2222  |  MAE rare/extreme: 30.1501  |  ratio: **1.65×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 18.2222  |  MAE rare/extreme: 30.1501  |  rare/normal MAE ratio: **1.65×**
- Mean uncertainty (std) normal: 0.8474  |  rare/extreme: 0.9616  |  rare/normal uncertainty ratio: **1.13×**
- Gaussian coverage@95 (diagnostic) normal: 0.034  |  rare/extreme: 0.034
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.65×).
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
| global | 0.029 | 2.9412 | 0.0033 | 32.7572 |
| normal | 0.030 | 2.9028 | 0.0032 | 32.3227 |
| rare_extreme | 0.029 | 3.2942 | 0.0037 | 36.7668 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.034 | 3.3657 | 0.0038 | 35.9061 |
| normal | 0.034 | 3.3218 | 0.0037 | 35.4276 |
| rare_extreme | 0.034 | 3.7695 | 0.0042 | 40.3249 |

