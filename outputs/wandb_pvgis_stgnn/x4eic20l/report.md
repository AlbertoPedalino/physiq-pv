# PVGIS-only ST-GNN forecasting report

Reuses the existing STGNN architecture on a **PVGIS-only** input. No real plant production, no ENERGIA, no quality score, no kWp/UPN. Anomaly labels are used **only** for stratified evaluation.

## Experiment

- Mode: **pvgis_stgnn**
- Model type: **stgnn_enhanced_dropout**
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
- Device: cuda  |  Generated (UTC): 2026-06-10T16:24:00+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 20.3810 | 46.2012 | 7.2339 | 2.5106 | 17.5082 | 0.759 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 19.3444 | 43.3983 | 7.0803 | 2.3610 | 17.3982 | 0.767 | — |
| group:rare_or_extreme | 983379 | 29.9254 | 66.6851 | 8.6489 | 8.9354 | 18.3922 | 0.683 | — |
| label:unusually_low_solar_potential | 118147 | 93.2137 | 145.0763 | 13.9602 | 13.9090 | 19.8060 | 0.181 | — |
| label:unusually_high_solar_potential | 38401 | 49.9822 | 73.2308 | 16.6856 | 16.0666 | 25.5654 | 0.431 | — |
| label:extreme_temperature_condition | 436524 | 18.2833 | 40.6647 | 8.0423 | 7.9545 | 17.6358 | 0.772 | — |
| label:extreme_wind_condition | 458693 | 23.6644 | 53.2594 | 7.4368 | 2.7156 | 17.7111 | 0.743 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 19.3444  |  MAE rare/extreme: 29.9254  |  ratio: **1.55×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 19.3444  |  MAE rare/extreme: 29.9254  |  rare/normal MAE ratio: **1.55×**
- Mean uncertainty (std) normal: 7.0803  |  rare/extreme: 8.6489  |  rare/normal uncertainty ratio: **1.22×**
- Gaussian coverage@95 (diagnostic) normal: 0.767  |  rare/extreme: 0.683
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.55×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.22×).

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
| global | 0.217 | 24.5776 | 0.0275 | 42.1936 |
| normal | 0.215 | 24.0492 | 0.0269 | 42.0079 |
| rare_extreme | 0.232 | 29.4429 | 0.0330 | 43.0949 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.759 | 28.3571 | 0.0317 | 0.2460 |
| normal | 0.767 | 27.7546 | 0.0311 | 0.2241 |
| rare_extreme | 0.683 | 33.9038 | 0.0380 | 0.5855 |

## Daytime-only interval reliability

Eval-only split based on PVGIS `solar_irradiance_poa` at the target timestamp: daytime > **10.0 W/m²**, nighttime <= **10.0 W/m²**. The irradiance is diagnostic metadata and is not added to the model inputs or targets.

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | PICP PI | MPIW PI | NMPIL PI | CLC PI | PICP Gaussian | MPIW Gaussian | NMPIL Gaussian | CLC Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime | 4747545 | 41.0310 | 65.4384 | 13.6719 | 13.4570 | 19.9274 | 0.452 | 46.7505 | 0.0523 | 7.7014 | 0.507 | 53.5937 | 0.0600 | 5.0835 |
| nighttime | 5290119 | 1.8490 | 14.3940 | 1.4563 | 1.2871 | 2.2918 | 0.006 | 4.6788 | 0.0052 | 66.1476 | 0.985 | 5.7087 | 0.0064 | 0.0109 |
| normal_daytime | 4195053 | 40.2860 | 63.7317 | 13.6339 | 13.4279 | 19.8861 | 0.457 | 46.6218 | 0.0522 | 7.2606 | 0.512 | 53.4449 | 0.0598 | 4.8532 |
| rare_extreme_daytime | 552492 | 46.6873 | 77.1758 | 13.9602 | 13.6684 | 20.2479 | 0.409 | 47.7283 | 0.0534 | 12.0531 | 0.474 | 54.7240 | 0.0613 | 7.2313 |

| stratum | fraction y_true=0 | fraction lower PI <= 0 | fraction lower Gaussian <= 0 | PI coverage y=0 | Gaussian coverage y=0 | PI coverage y>0 | Gaussian coverage y>0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| daytime | 0.000 | 0.000 | 0.032 | — | — | 0.452 | 0.507 |
| nighttime | 0.988 | 0.000 | 0.984 | 0.000 | 0.987 | 0.452 | 0.793 |
| normal_daytime | 0.000 | 0.000 | 0.033 | — | — | 0.457 | 0.512 |
| rare_extreme_daytime | 0.000 | 0.000 | 0.028 | — | — | 0.409 | 0.474 |

**PICP PI daytime is materially higher than global** (0.452 vs 0.217, delta 0.235).
It nevertheless remains low relative to the 0.95 coverage target.

