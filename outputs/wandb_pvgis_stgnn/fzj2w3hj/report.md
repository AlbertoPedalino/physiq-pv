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
- Device: cuda  |  Generated (UTC): 2026-06-09T15:57:23+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 19.8745 | 45.9398 | 0.9297 | 0.0757 | 2.6834 | 0.034 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 18.7977 | 43.1386 | 0.9126 | 0.0643 | 2.6691 | 0.034 | — |
| group:rare_or_extreme | 983379 | 29.7891 | 66.3923 | 1.0866 | 0.8002 | 2.8146 | 0.034 | — |
| label:unusually_low_solar_potential | 118147 | 97.1417 | 144.8540 | 1.8567 | 1.4168 | 3.7388 | 0.001 | — |
| label:unusually_high_solar_potential | 38401 | 49.1329 | 73.4039 | 2.0289 | 1.6283 | 3.7675 | 0.039 | — |
| label:extreme_temperature_condition | 436524 | 17.6176 | 40.1009 | 0.9053 | 0.6699 | 2.2691 | 0.043 | — |
| label:extreme_wind_condition | 458693 | 23.3405 | 53.0721 | 1.0039 | 0.0885 | 2.9189 | 0.033 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 18.7977  |  MAE rare/extreme: 29.7891  |  ratio: **1.58×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 18.7977  |  MAE rare/extreme: 29.7891  |  rare/normal MAE ratio: **1.58×**
- Mean uncertainty (std) normal: 0.9126  |  rare/extreme: 1.0866  |  rare/normal uncertainty ratio: **1.19×**
- Gaussian coverage@95 (diagnostic) normal: 0.034  |  rare/extreme: 0.034
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.58×).
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
| global | 0.030 | 3.1856 | 0.0036 | 35.4403 |
| normal | 0.030 | 3.1272 | 0.0035 | 34.7916 |
| rare_extreme | 0.030 | 3.7231 | 0.0042 | 41.4111 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.034 | 3.6443 | 0.0041 | 38.8426 |
| normal | 0.034 | 3.5775 | 0.0040 | 38.1300 |
| rare_extreme | 0.034 | 4.2596 | 0.0048 | 45.4036 |

