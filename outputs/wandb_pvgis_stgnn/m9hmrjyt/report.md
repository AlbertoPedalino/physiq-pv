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
- Device: cuda  |  Generated (UTC): 2026-06-09T15:51:43+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 20.0534 | 46.2216 | 0.9857 | 0.1101 | 2.8145 | 0.038 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 18.8173 | 43.0388 | 0.9707 | 0.0901 | 2.8054 | 0.038 | — |
| group:rare_or_extreme | 983379 | 31.4351 | 68.9362 | 1.1240 | 0.7193 | 2.8977 | 0.034 | — |
| label:unusually_low_solar_potential | 118147 | 106.6037 | 153.8062 | 2.0980 | 1.7984 | 3.7075 | 0.002 | — |
| label:unusually_high_solar_potential | 38401 | 59.5206 | 83.0944 | 1.4444 | 0.8778 | 3.2367 | 0.034 | — |
| label:extreme_temperature_condition | 436524 | 17.6933 | 40.3706 | 0.9295 | 0.5771 | 2.4576 | 0.043 | — |
| label:extreme_wind_condition | 458693 | 24.2197 | 53.5157 | 1.0537 | 0.1483 | 2.9692 | 0.034 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 18.8173  |  MAE rare/extreme: 31.4351  |  ratio: **1.67×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 18.8173  |  MAE rare/extreme: 31.4351  |  rare/normal MAE ratio: **1.67×**
- Mean uncertainty (std) normal: 0.9707  |  rare/extreme: 1.1240  |  rare/normal uncertainty ratio: **1.16×**
- Gaussian coverage@95 (diagnostic) normal: 0.038  |  rare/extreme: 0.034
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.67×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.16×).

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
| global | 0.033 | 3.3767 | 0.0038 | 36.2137 |
| normal | 0.034 | 3.3252 | 0.0037 | 35.5320 |
| rare_extreme | 0.030 | 3.8505 | 0.0043 | 42.7051 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.038 | 3.8640 | 0.0043 | 39.5353 |
| normal | 0.038 | 3.8051 | 0.0043 | 38.7689 |
| rare_extreme | 0.034 | 4.4060 | 0.0049 | 46.8685 |

