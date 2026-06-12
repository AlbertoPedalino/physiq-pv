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
- Device: cuda  |  Generated (UTC): 2026-06-11T13:57:19+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 21.2822 | 46.8782 | 8.1326 | 2.3843 | 20.3071 | 0.770 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 20.3720 | 44.5026 | 7.9268 | 2.2505 | 20.0992 | 0.775 | — |
| group:rare_or_extreme | 983379 | 29.6627 | 64.7788 | 10.0274 | 9.9090 | 22.0185 | 0.722 | — |
| label:unusually_low_solar_potential | 118147 | 84.2410 | 135.2138 | 15.8832 | 16.1173 | 22.8993 | 0.320 | — |
| label:unusually_high_solar_potential | 38401 | 59.7073 | 85.0124 | 22.6032 | 22.2630 | 35.4483 | 0.519 | — |
| label:extreme_temperature_condition | 436524 | 18.5240 | 40.4730 | 9.4351 | 7.9629 | 21.5688 | 0.814 | — |
| label:extreme_wind_condition | 458693 | 24.5433 | 53.3197 | 8.3712 | 2.5285 | 20.5726 | 0.749 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 20.3720  |  MAE rare/extreme: 29.6627  |  ratio: **1.46×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 20.3720  |  MAE rare/extreme: 29.6627  |  rare/normal MAE ratio: **1.46×**
- Mean uncertainty (std) normal: 7.9268  |  rare/extreme: 10.0274  |  rare/normal uncertainty ratio: **1.26×**
- Gaussian coverage@95 (diagnostic) normal: 0.775  |  rare/extreme: 0.722
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.46×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.26×).

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
| global | 0.232 | 27.6885 | 0.0310 | 40.7087 |
| normal | 0.228 | 26.9828 | 0.0302 | 41.2555 |
| rare_extreme | 0.268 | 34.1855 | 0.0383 | 35.0488 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.770 | 31.8799 | 0.0357 | 0.2517 |
| normal | 0.775 | 31.0732 | 0.0348 | 0.2347 |
| rare_extreme | 0.722 | 39.3074 | 0.0440 | 0.4738 |

## Daytime-only interval reliability

Eval-only split based on PVGIS `solar_irradiance_poa` at the target timestamp: daytime > **10.0 W/m²**, nighttime <= **10.0 W/m²**. The irradiance is diagnostic metadata and is not added to the model inputs or targets.

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | PICP PI | MPIW PI | NMPIL PI | CLC PI | PICP Gaussian | MPIW Gaussian | NMPIL Gaussian | CLC Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime | 4747545 | 42.7191 | 66.7087 | 15.5606 | 15.4209 | 23.3899 | 0.483 | 53.1854 | 0.0595 | 6.4255 | 0.534 | 60.9974 | 0.0683 | 4.4614 |
| nighttime | 5290119 | 2.0439 | 13.2703 | 1.4665 | 1.2412 | 2.3118 | 0.007 | 4.8066 | 0.0054 | 67.0229 | 0.982 | 5.7488 | 0.0064 | 0.0111 |
| normal_daytime | 4195053 | 42.1741 | 65.3536 | 15.4515 | 15.3442 | 23.2348 | 0.485 | 52.8124 | 0.0591 | 6.2646 | 0.533 | 60.5699 | 0.0678 | 4.4681 |
| rare_extreme_daytime | 552492 | 46.8578 | 76.2162 | 16.3887 | 16.0188 | 24.6435 | 0.469 | 56.0183 | 0.0627 | 7.7789 | 0.540 | 64.2436 | 0.0719 | 4.4040 |
| high_daytime | 1186852 | 45.0343 | 72.0501 | 16.8885 | 15.0286 | 26.3229 | 0.527 | 57.8808 | 0.0648 | 4.5043 | 0.580 | 66.2030 | 0.0741 | 3.0821 |
| peak_daytime | 474719 | 45.3240 | 71.4751 | 21.6447 | 20.6036 | 30.9869 | 0.607 | 74.1627 | 0.0830 | 2.6437 | 0.664 | 84.8474 | 0.0950 | 1.7522 |
| extreme_peak_daytime | 237361 | 47.8882 | 73.9218 | 25.2245 | 24.9557 | 33.9147 | 0.647 | 86.4483 | 0.0968 | 2.1047 | 0.700 | 98.8799 | 0.1107 | 1.4582 |

### Daytime production-tail diagnostics

| stratum | count | MAE | RMSE | mean residual | median residual | fraction underprediction | fraction above PI | PICP PI | PICP Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high_daytime | 1186852 | 45.0343 | 72.0501 | -36.2264 | -22.2830 | 0.842 | 0.419 | 0.527 | 0.580 |
| peak_daytime | 474719 | 45.3240 | 71.4751 | -41.9615 | -27.5567 | 0.905 | 0.382 | 0.607 | 0.664 |
| extreme_peak_daytime | 237361 | 47.8882 | 73.9218 | -45.8950 | -31.1082 | 0.928 | 0.351 | 0.647 | 0.700 |

| stratum | fraction y_true=0 | fraction lower PI <= 0 | fraction lower Gaussian <= 0 | PI coverage y=0 | Gaussian coverage y=0 | PI coverage y>0 | Gaussian coverage y>0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| daytime | 0.000 | 0.000 | 0.091 | — | — | 0.483 | 0.534 |
| nighttime | 0.988 | 0.000 | 0.981 | 0.000 | 0.983 | 0.563 | 0.929 |
| normal_daytime | 0.000 | 0.000 | 0.093 | — | — | 0.485 | 0.533 |
| rare_extreme_daytime | 0.000 | 0.000 | 0.077 | — | — | 0.469 | 0.540 |
| high_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.527 | 0.580 |
| peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.607 | 0.664 |
| extreme_peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.647 | 0.700 |

**PICP PI daytime is materially higher than global** (0.483 vs 0.232, delta 0.251).
It nevertheless remains low relative to the 0.95 coverage target.

## Residual bias diagnostics by stratum

`residual = y_pred_mean - y_true`: positive means overprediction, negative means underprediction.

| stratum | count | MAE | RMSE | mean_residual | median_residual | overprediction% | underprediction% | PICP PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 10037664 | 21.2822 | 46.8782 | -4.8527 | 1.3085 | 67.9% | 32.1% | 0.232 | 16.6% | 60.2% |
| daytime | 4747545 | 42.7191 | 66.7087 | -12.5362 | -12.8346 | 32.3% | 67.7% | 0.483 | 35.2% | 16.6% |
| nighttime | 5290119 | 2.0439 | 13.2703 | 2.0428 | 1.4553 | 99.8% | 0.2% | 0.007 | 0.0% | 99.3% |
| normal | 9054285 | 20.3720 | 44.5026 | -5.7589 | 1.3035 | 68.0% | 32.0% | 0.228 | 16.8% | 60.4% |
| rare_extreme | 983379 | 29.6627 | 64.7788 | 3.4918 | 1.3660 | 67.3% | 32.7% | 0.268 | 15.2% | 58.0% |
| normal_daytime | 4195053 | 42.1741 | 65.3536 | -14.2238 | -13.6885 | 31.1% | 68.9% | 0.485 | 36.2% | 15.3% |
| rare_extreme_daytime | 552492 | 46.8578 | 76.2162 | 0.2775 | -5.9642 | 41.9% | 58.1% | 0.469 | 27.1% | 26.1% |
| normal_nighttime | 4859232 | 1.5500 | 1.7226 | 1.5489 | 1.4535 | 99.9% | 0.1% | 0.007 | 0.0% | 99.3% |
| rare_extreme_nighttime | 430887 | 7.6147 | 46.1363 | 7.6133 | 1.4758 | 99.8% | 0.2% | 0.011 | 0.0% | 98.9% |
| label:unusually_low_solar_potential | 118147 | 84.2410 | 135.2138 | 84.0081 | 43.9338 | 98.1% | 1.9% | 0.134 | 0.2% | 86.4% |
| label:unusually_high_solar_potential | 38401 | 59.7073 | 85.0124 | -58.9863 | -42.8379 | 3.2% | 96.8% | 0.452 | 54.6% | 0.2% |
| label:extreme_temperature_condition | 436524 | 18.5240 | 40.4730 | -5.1447 | 1.2452 | 62.5% | 37.5% | 0.344 | 14.1% | 51.5% |
| label:extreme_wind_condition | 458693 | 24.5433 | 53.3197 | -2.9219 | 1.3499 | 69.7% | 30.3% | 0.221 | 16.4% | 61.5% |

## Daytime production-bin diagnostics

Bins use physical `y_true` in watts and only samples with target-time `solar_irradiance_poa > 10 W/m²`. Intervals are `[lower, upper)`, with the final bin `y_true >= 100 W`.

| bin | count | MAE | RMSE | mean_residual | median_residual | underprediction% | overprediction% | PICP PI | MPIW PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime_0_20 | 396207 | 13.1311 | 23.1715 | 6.4795 | -0.8128 | 53.5% | 46.5% | 0.493 | 22.6639 | 27.5% | 23.2% |
| daytime_20_40 | 376626 | 26.1321 | 42.2074 | 11.5564 | 3.0958 | 45.0% | 55.0% | 0.497 | 40.3853 | 22.9% | 27.4% |
| daytime_40_60 | 352255 | 36.6333 | 62.5514 | 12.1855 | -2.2545 | 52.4% | 47.6% | 0.513 | 47.9840 | 23.7% | 25.0% |
| daytime_60_80 | 266151 | 42.4719 | 67.2247 | 10.7860 | -4.6170 | 54.2% | 45.8% | 0.504 | 55.3353 | 25.0% | 24.6% |
| daytime_80_100 | 206451 | 47.2346 | 71.4862 | 6.1095 | -10.5226 | 58.8% | 41.2% | 0.482 | 59.1237 | 28.9% | 22.8% |
| daytime_gt_100 | 3149855 | 48.8297 | 72.6108 | -23.7662 | -21.1189 | 75.6% | 24.4% | 0.475 | 58.5660 | 40.1% | 12.4% |

## Automatic interpretation of residual asymmetry

- **Global:** underprediction; mean residual -4.8527 W, over 67.9%, under 32.1%.
- **Daytime:** underprediction; mean residual -12.5362 W, over 32.3%, under 67.7%.
- **Nighttime:** overprediction; mean residual 2.0428 W, over 99.8%, under 0.2%.
- **Unusually low solar potential:** overprediction; mean residual 84.0081 W, over 98.1%, under 1.9%.
- **Unusually high solar potential:** underprediction; mean residual -58.9863 W, over 3.2%, under 96.8%.
- **Rare/extreme daytime:** overprediction; mean residual 0.2775 W, over 41.9%, under 58.1%.
- **Production >= 100 W:** underprediction; mean residual -23.7662 W, over 24.4%, under 75.6%.
- **Nighttime softplus signature:** misses are predominantly below the PI while most targets are zero and most empirical lower bounds remain positive. This is consistent with `softplus` plus `y_true=0`.
- The model tends to **overpredict unusually low solar potential**.
- The model tends to **underpredict unusually high solar potential**.
- The model **underpredicts the >=100 W production bin**. Targets exceed `upper_pi` in 40.1% of these samples.
- The model overpredicts the low daytime production bin.
- **Global PI miss direction:** below 60.2%, above 16.6%; intervals/centres are predominantly too high.
- **Likely cause of low PICP:** softplus/zero-target nighttime misses, high-production targets above the PI. Centre bias and interval width should be interpreted together.

