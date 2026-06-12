# PVGIS-only ST-GNN forecasting report

Reuses the existing STGNN architecture on a **PVGIS-only** input. No real plant production, no ENERGIA, no quality score, no kWp/UPN. Anomaly labels are used **only** for stratified evaluation.

## Experiment

- Mode: **pvgis_stgnn**
- Model type: **stgnn_enhanced_dropout**
- Feature set: **full**
- W&B enabled: **False**
- MC Dropout: **enabled** (experimental)

## Parameters

- Target variable: **pv_power_output**
- PV normalized target upper clip: **none**
- PV normalized target lower clip: **0.0**
- Selected features (11): temperature_2m, solar_irradiance_poa, wind_speed_10m, sin_elev, cos_elev, kt, kt_std_3h, dghi_dt, dni_norm, dhi_norm, pv_lag_pvgis
- MC samples: **20**
- seq_len: **24**  |  horizon: **1**
- Train years: 2016,2017,2018
- Test year: **2019**
- Nodes (locations): **1149**  |  epochs: **5**
- batch_size: 16  |  lr: 0.001
- Anomaly scores: outputs/pvgis_anomaly_2019_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Predictions: **10037664**
- Device: cuda  |  Generated (UTC): 2026-06-11T09:36:55+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 19.9517 | 45.8703 | 8.6700 | 2.9755 | 21.2915 | 0.806 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 18.9248 | 43.0243 | 8.4757 | 2.7706 | 21.1413 | 0.812 | — |
| group:rare_or_extreme | 983379 | 29.4064 | 66.5838 | 10.4590 | 10.6162 | 22.5231 | 0.746 | — |
| label:unusually_low_solar_potential | 118147 | 93.4210 | 145.9354 | 16.8623 | 16.8701 | 24.0118 | 0.298 | — |
| label:unusually_high_solar_potential | 38401 | 50.4091 | 73.1864 | 21.3806 | 20.8141 | 32.8733 | 0.545 | — |
| label:extreme_temperature_condition | 436524 | 17.2563 | 39.7852 | 9.7534 | 9.2901 | 21.7109 | 0.840 | — |
| label:extreme_wind_condition | 458693 | 23.3288 | 53.0557 | 8.8716 | 3.1829 | 21.4573 | 0.787 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 18.9248  |  MAE rare/extreme: 29.4064  |  ratio: **1.55×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 18.9248  |  MAE rare/extreme: 29.4064  |  rare/normal MAE ratio: **1.55×**
- Mean uncertainty (std) normal: 8.4757  |  rare/extreme: 10.4590  |  rare/normal uncertainty ratio: **1.23×**
- Gaussian coverage@95 (diagnostic) normal: 0.812  |  rare/extreme: 0.746
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.55×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.23×).

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
| global | 0.259 | 29.4334 | 0.0330 | 33.0724 |
| normal | 0.256 | 28.7657 | 0.0322 | 33.2909 |
| rare_extreme | 0.286 | 35.5813 | 0.0398 | 30.4634 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.806 | 33.9865 | 0.0380 | 0.1990 |
| normal | 0.812 | 33.2249 | 0.0372 | 0.1846 |
| rare_extreme | 0.746 | 40.9994 | 0.0459 | 0.3983 |

## Daytime-only interval reliability

Eval-only split based on PVGIS `solar_irradiance_poa` at the target timestamp: daytime > **10.0 W/m²**, nighttime <= **10.0 W/m²**. The irradiance is diagnostic metadata and is not added to the model inputs or targets.

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | PICP PI | MPIW PI | NMPIL PI | CLC PI | PICP Gaussian | MPIW Gaussian | NMPIL Gaussian | CLC Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime | 4747545 | 40.1692 | 64.9080 | 16.5262 | 16.3118 | 24.2741 | 0.540 | 56.4826 | 0.0632 | 3.8781 | 0.594 | 64.7826 | 0.0725 | 2.6218 |
| nighttime | 5290119 | 1.8078 | 14.5402 | 1.6196 | 1.3698 | 2.6643 | 0.007 | 5.1585 | 0.0058 | 72.1408 | 0.996 | 6.3490 | 0.0071 | 0.0116 |
| normal_daytime | 4195053 | 39.4370 | 63.1865 | 16.4615 | 16.2630 | 24.1948 | 0.545 | 56.2629 | 0.0630 | 3.6775 | 0.597 | 64.5291 | 0.0722 | 2.5460 |
| rare_extreme_daytime | 552492 | 45.7286 | 76.7295 | 17.0172 | 16.6832 | 24.9032 | 0.502 | 58.1508 | 0.0651 | 5.8090 | 0.574 | 66.7073 | 0.0747 | 3.2761 |
| high_daytime | 1186852 | 42.5490 | 67.9698 | 15.5398 | 14.0024 | 23.7063 | 0.507 | 53.2582 | 0.0596 | 5.0510 | 0.561 | 60.9159 | 0.0682 | 3.3882 |
| peak_daytime | 474719 | 43.5214 | 67.8051 | 19.5003 | 18.4082 | 28.0705 | 0.572 | 66.7848 | 0.0748 | 3.3485 | 0.633 | 76.4412 | 0.0856 | 2.1203 |
| extreme_peak_daytime | 237361 | 46.2618 | 70.3523 | 22.7466 | 22.2214 | 31.0173 | 0.606 | 77.9024 | 0.0872 | 2.7996 | 0.667 | 89.1667 | 0.0998 | 1.7881 |

### Daytime production-tail diagnostics

| stratum | count | MAE | RMSE | mean residual | median residual | fraction underprediction | fraction above PI | PICP PI | PICP Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high_daytime | 1186852 | 42.5490 | 67.9698 | -31.7254 | -20.3344 | 0.794 | 0.422 | 0.507 | 0.561 |
| peak_daytime | 474719 | 43.5214 | 67.8051 | -39.1338 | -26.6057 | 0.883 | 0.411 | 0.572 | 0.633 |
| extreme_peak_daytime | 237361 | 46.2618 | 70.3523 | -43.4669 | -30.4503 | 0.909 | 0.389 | 0.606 | 0.667 |

| stratum | fraction y_true=0 | fraction lower PI <= 0 | fraction lower Gaussian <= 0 | PI coverage y=0 | Gaussian coverage y=0 | PI coverage y>0 | Gaussian coverage y>0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| daytime | 0.000 | 0.000 | 0.059 | — | — | 0.540 | 0.594 |
| nighttime | 0.988 | 0.000 | 0.995 | 0.000 | 0.997 | 0.539 | 0.915 |
| normal_daytime | 0.000 | 0.000 | 0.061 | — | — | 0.545 | 0.597 |
| rare_extreme_daytime | 0.000 | 0.000 | 0.049 | — | — | 0.502 | 0.574 |
| high_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.507 | 0.561 |
| peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.572 | 0.633 |
| extreme_peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.606 | 0.667 |

**PICP PI daytime is materially higher than global** (0.540 vs 0.259, delta 0.281).
It nevertheless remains low relative to the 0.95 coverage target.

## Residual bias diagnostics by stratum

`residual = y_pred_mean - y_true`: positive means overprediction, negative means underprediction.

| stratum | count | MAE | RMSE | mean_residual | median_residual | overprediction% | underprediction% | PICP PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 10037664 | 19.9517 | 45.8703 | -1.3242 | 0.9847 | 71.7% | 28.3% | 0.259 | 12.8% | 61.3% |
| daytime | 4747545 | 40.1692 | 64.9080 | -4.8123 | -6.0233 | 40.4% | 59.6% | 0.540 | 27.1% | 18.9% |
| nighttime | 5290119 | 1.8078 | 14.5402 | 1.8062 | 1.0818 | 99.8% | 0.2% | 0.007 | 0.0% | 99.3% |
| normal | 9054285 | 18.9248 | 43.0243 | -2.2981 | 0.9780 | 71.9% | 28.1% | 0.256 | 12.9% | 61.5% |
| rare_extreme | 983379 | 29.4064 | 66.5838 | 7.6436 | 1.0651 | 70.6% | 29.4% | 0.286 | 12.5% | 58.9% |
| normal_daytime | 4195053 | 39.4370 | 63.1865 | -6.3672 | -6.5636 | 39.5% | 60.5% | 0.545 | 27.8% | 17.7% |
| rare_extreme_daytime | 552492 | 45.7286 | 76.7295 | 6.9944 | -1.5639 | 47.9% | 52.1% | 0.502 | 22.2% | 27.6% |
| normal_nighttime | 4859232 | 1.2164 | 1.5350 | 1.2148 | 1.0793 | 99.8% | 0.2% | 0.006 | 0.0% | 99.3% |
| rare_extreme_nighttime | 430887 | 8.4777 | 50.6857 | 8.4760 | 1.1121 | 99.8% | 0.2% | 0.009 | 0.0% | 99.1% |
| label:unusually_low_solar_potential | 118147 | 93.4210 | 145.9354 | 92.8809 | 47.4087 | 98.0% | 2.0% | 0.131 | 0.2% | 86.7% |
| label:unusually_high_solar_potential | 38401 | 50.4091 | 73.1864 | -49.3749 | -36.7036 | 4.2% | 95.8% | 0.478 | 51.9% | 0.3% |
| label:extreme_temperature_condition | 436524 | 17.2563 | 39.7852 | -2.9971 | 0.9297 | 65.8% | 34.2% | 0.361 | 12.2% | 51.7% |
| label:extreme_wind_condition | 458693 | 23.3288 | 53.0557 | 0.8156 | 1.0359 | 73.9% | 26.1% | 0.250 | 12.4% | 62.6% |

## Daytime production-bin diagnostics

Bins use physical `y_true` in watts and only samples with target-time `solar_irradiance_poa > 10 W/m²`. Intervals are `[lower, upper)`, with the final bin `y_true >= 100 W`.

| bin | count | MAE | RMSE | mean_residual | median_residual | underprediction% | overprediction% | PICP PI | MPIW PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime_0_20 | 396207 | 15.5134 | 27.9487 | 10.7331 | 1.0513 | 46.4% | 53.6% | 0.545 | 27.3813 | 17.3% | 28.2% |
| daytime_20_40 | 376626 | 26.9661 | 46.2158 | 20.6066 | 10.3488 | 31.1% | 68.9% | 0.589 | 48.9868 | 7.7% | 33.4% |
| daytime_40_60 | 352255 | 34.6562 | 66.0859 | 24.8487 | 6.6257 | 39.8% | 60.2% | 0.645 | 57.6471 | 6.1% | 29.4% |
| daytime_60_80 | 266151 | 38.7477 | 68.1707 | 23.4389 | 5.4381 | 43.6% | 56.4% | 0.630 | 63.9882 | 9.2% | 27.9% |
| daytime_80_100 | 206451 | 42.1793 | 70.3362 | 18.0308 | 0.6556 | 49.4% | 50.6% | 0.600 | 66.6035 | 15.3% | 24.7% |
| daytime_gt_100 | 3149855 | 45.4541 | 69.2173 | -17.0083 | -15.9107 | 68.8% | 31.2% | 0.510 | 59.6116 | 35.4% | 13.6% |

## Automatic interpretation of residual asymmetry

- **Global:** underprediction; mean residual -1.3242 W, over 71.7%, under 28.3%.
- **Daytime:** underprediction; mean residual -4.8123 W, over 40.4%, under 59.6%.
- **Nighttime:** overprediction; mean residual 1.8062 W, over 99.8%, under 0.2%.
- **Unusually low solar potential:** overprediction; mean residual 92.8809 W, over 98.0%, under 2.0%.
- **Unusually high solar potential:** underprediction; mean residual -49.3749 W, over 4.2%, under 95.8%.
- **Rare/extreme daytime:** overprediction; mean residual 6.9944 W, over 47.9%, under 52.1%.
- **Production >= 100 W:** underprediction; mean residual -17.0083 W, over 31.2%, under 68.8%.
- **Nighttime softplus signature:** misses are predominantly below the PI while most targets are zero and most empirical lower bounds remain positive. This is consistent with `softplus` plus `y_true=0`.
- The model tends to **overpredict unusually low solar potential**.
- The model tends to **underpredict unusually high solar potential**.
- The model **underpredicts the >=100 W production bin**. Targets exceed `upper_pi` in 35.4% of these samples.
- The model overpredicts the low daytime production bin.
- **Global PI miss direction:** below 61.3%, above 12.8%; intervals/centres are predominantly too high.
- **Likely cause of low PICP:** softplus/zero-target nighttime misses, high-production targets above the PI. Centre bias and interval width should be interpreted together.

