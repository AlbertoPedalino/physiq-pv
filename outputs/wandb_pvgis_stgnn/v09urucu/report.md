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
- Device: cuda  |  Generated (UTC): 2026-06-10T09:52:25+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 20.2005 | 46.3771 | 0.9371 | 0.0585 | 2.7452 | 0.036 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 18.9604 | 43.2322 | 0.9234 | 0.0476 | 2.7384 | 0.037 | — |
| group:rare_or_extreme | 983379 | 31.6191 | 68.8880 | 1.0630 | 0.6443 | 2.8072 | 0.031 | — |
| label:unusually_low_solar_potential | 118147 | 103.2072 | 151.6241 | 2.0139 | 1.6891 | 3.6234 | 0.002 | — |
| label:unusually_high_solar_potential | 38401 | 67.4519 | 90.4009 | 1.3045 | 0.7330 | 3.1014 | 0.030 | — |
| label:extreme_temperature_condition | 436524 | 18.1685 | 40.8365 | 0.8723 | 0.4934 | 2.3452 | 0.039 | — |
| label:extreme_wind_condition | 458693 | 24.3951 | 53.7029 | 1.0030 | 0.0964 | 2.8952 | 0.031 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 18.9604  |  MAE rare/extreme: 31.6191  |  ratio: **1.67×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 18.9604  |  MAE rare/extreme: 31.6191  |  rare/normal MAE ratio: **1.67×**
- Mean uncertainty (std) normal: 0.9234  |  rare/extreme: 1.0630  |  rare/normal uncertainty ratio: **1.15×**
- Gaussian coverage@95 (diagnostic) normal: 0.037  |  rare/extreme: 0.031
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.67×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.15×).

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
| global | 0.032 | 3.2103 | 0.0036 | 35.0281 |
| normal | 0.032 | 3.1635 | 0.0035 | 34.3660 |
| rare_extreme | 0.027 | 3.6412 | 0.0041 | 41.3709 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.036 | 3.6735 | 0.0041 | 38.3312 |
| normal | 0.037 | 3.6199 | 0.0041 | 37.5863 |
| rare_extreme | 0.031 | 4.1669 | 0.0047 | 45.4994 |

