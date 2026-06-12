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
- Device: cuda  |  Generated (UTC): 2026-06-11T13:09:17+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 19.8183 | 45.6349 | 9.2706 | 3.9723 | 21.8050 | 0.823 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 18.6291 | 42.4404 | 9.0406 | 3.6228 | 21.5825 | 0.831 | — |
| group:rare_or_extreme | 983379 | 30.7677 | 68.3602 | 11.3887 | 11.7383 | 23.7204 | 0.752 | — |
| label:unusually_low_solar_potential | 118147 | 107.8936 | 154.5116 | 18.0457 | 17.9197 | 24.4602 | 0.163 | — |
| label:unusually_high_solar_potential | 38401 | 40.9232 | 68.6322 | 24.9586 | 24.9213 | 38.4243 | 0.725 | — |
| label:extreme_temperature_condition | 436524 | 17.4069 | 39.9358 | 10.7296 | 10.4450 | 23.3759 | 0.859 | — |
| label:extreme_wind_condition | 458693 | 24.2270 | 53.2716 | 9.5413 | 4.6841 | 21.9947 | 0.789 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 18.6291  |  MAE rare/extreme: 30.7677  |  ratio: **1.65×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 18.6291  |  MAE rare/extreme: 30.7677  |  rare/normal MAE ratio: **1.65×**
- Mean uncertainty (std) normal: 9.0406  |  rare/extreme: 11.3887  |  rare/normal uncertainty ratio: **1.26×**
- Gaussian coverage@95 (diagnostic) normal: 0.831  |  rare/extreme: 0.752
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.65×).
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
| global | 0.272 | 31.4436 | 0.0352 | 31.0721 |
| normal | 0.270 | 30.6519 | 0.0343 | 30.9610 |
| rare_extreme | 0.292 | 38.7336 | 0.0434 | 31.2819 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.823 | 36.3407 | 0.0407 | 0.1851 |
| normal | 0.831 | 35.4390 | 0.0397 | 0.1699 |
| rare_extreme | 0.752 | 44.6437 | 0.0500 | 0.4134 |

## Daytime-only interval reliability

Eval-only split based on PVGIS `solar_irradiance_poa` at the target timestamp: daytime > **10.0 W/m²**, nighttime <= **10.0 W/m²**. The irradiance is diagnostic metadata and is not added to the model inputs or targets.

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | PICP PI | MPIW PI | NMPIL PI | CLC PI | PICP Gaussian | MPIW Gaussian | NMPIL Gaussian | CLC Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime | 4747545 | 39.5215 | 64.4269 | 17.4326 | 16.8998 | 25.1007 | 0.570 | 59.5720 | 0.0667 | 3.0527 | 0.634 | 68.3357 | 0.0765 | 1.8802 |
| nighttime | 5290119 | 2.1360 | 15.0465 | 1.9457 | 1.5810 | 3.3608 | 0.004 | 6.2002 | 0.0069 | 88.7854 | 0.993 | 7.6273 | 0.0085 | 0.0141 |
| normal_daytime | 4195053 | 38.4666 | 62.3046 | 17.3112 | 16.8081 | 24.9031 | 0.577 | 59.1559 | 0.0662 | 2.8304 | 0.640 | 67.8599 | 0.0760 | 1.7677 |
| rare_extreme_daytime | 552492 | 47.5311 | 78.6960 | 18.3543 | 17.6023 | 26.7506 | 0.517 | 62.7315 | 0.0702 | 5.4268 | 0.591 | 71.9487 | 0.0805 | 3.0106 |
| high_daytime | 1186852 | 39.7405 | 67.3025 | 18.6291 | 16.5066 | 30.0016 | 0.624 | 63.8238 | 0.0714 | 1.9375 | 0.659 | 73.0259 | 0.0818 | 1.5828 |
| peak_daytime | 474719 | 35.4270 | 64.1453 | 24.8807 | 24.3228 | 34.7405 | 0.765 | 85.1955 | 0.0954 | 0.7012 | 0.793 | 97.5325 | 0.1092 | 0.6319 |
| extreme_peak_daytime | 237361 | 35.2939 | 64.8768 | 28.5803 | 28.6177 | 37.5859 | 0.802 | 97.8481 | 0.1095 | 0.5920 | 0.826 | 112.0349 | 0.1254 | 0.5602 |

### Daytime production-tail diagnostics

| stratum | count | MAE | RMSE | mean residual | median residual | fraction underprediction | fraction above PI | PICP PI | PICP Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high_daytime | 1186852 | 39.7405 | 67.3025 | -21.7164 | -7.0099 | 0.608 | 0.293 | 0.624 | 0.659 |
| peak_daytime | 474719 | 35.4270 | 64.1453 | -20.9130 | -5.2167 | 0.582 | 0.207 | 0.765 | 0.793 |
| extreme_peak_daytime | 237361 | 35.2939 | 64.8768 | -24.3275 | -8.6241 | 0.638 | 0.189 | 0.802 | 0.826 |

| stratum | fraction y_true=0 | fraction lower PI <= 0 | fraction lower Gaussian <= 0 | PI coverage y=0 | Gaussian coverage y=0 | PI coverage y>0 | Gaussian coverage y>0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| daytime | 0.000 | 0.000 | 0.041 | — | — | 0.570 | 0.634 |
| nighttime | 0.988 | 0.000 | 0.993 | 0.000 | 0.996 | 0.349 | 0.814 |
| normal_daytime | 0.000 | 0.000 | 0.042 | — | — | 0.577 | 0.640 |
| rare_extreme_daytime | 0.000 | 0.000 | 0.031 | — | — | 0.517 | 0.591 |
| high_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.624 | 0.659 |
| peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.765 | 0.793 |
| extreme_peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.802 | 0.826 |

**PICP PI daytime is materially higher than global** (0.570 vs 0.272, delta 0.298).
It nevertheless remains low relative to the 0.95 coverage target.

## Residual bias diagnostics by stratum

`residual = y_pred_mean - y_true`: positive means overprediction, negative means underprediction.

| stratum | count | MAE | RMSE | mean_residual | median_residual | overprediction% | underprediction% | PICP PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 10037664 | 19.8183 | 45.6349 | 1.7342 | 1.1297 | 76.0% | 24.0% | 0.272 | 9.8% | 63.0% |
| daytime | 4747545 | 39.5215 | 64.4269 | 1.2871 | -0.5003 | 49.2% | 50.8% | 0.570 | 20.8% | 22.3% |
| nighttime | 5290119 | 2.1360 | 15.0465 | 2.1354 | 1.1592 | 99.9% | 0.1% | 0.004 | 0.0% | 99.6% |
| normal | 9054285 | 18.6291 | 42.4404 | 0.4506 | 1.1118 | 75.8% | 24.2% | 0.270 | 9.9% | 63.1% |
| rare_extreme | 983379 | 30.7677 | 68.3602 | 13.5528 | 1.3800 | 77.2% | 22.8% | 0.292 | 8.7% | 62.1% |
| normal_daytime | 4195053 | 38.4666 | 62.3046 | -0.7678 | -1.3520 | 47.9% | 52.1% | 0.577 | 21.5% | 20.9% |
| rare_extreme_daytime | 552492 | 47.5311 | 78.6960 | 16.8904 | 7.6265 | 59.5% | 40.5% | 0.517 | 15.4% | 32.9% |
| normal_nighttime | 4859232 | 1.5031 | 2.2153 | 1.5025 | 1.1551 | 99.9% | 0.1% | 0.004 | 0.0% | 99.6% |
| rare_extreme_nighttime | 430887 | 9.2735 | 52.1939 | 9.2731 | 1.2101 | 100.0% | 0.0% | 0.004 | 0.0% | 99.6% |
| label:unusually_low_solar_potential | 118147 | 107.8936 | 154.5116 | 107.8571 | 62.3474 | 99.8% | 0.2% | 0.041 | 0.0% | 95.9% |
| label:unusually_high_solar_potential | 38401 | 40.9232 | 68.6322 | -38.0434 | -23.2989 | 13.9% | 86.1% | 0.693 | 30.4% | 0.3% |
| label:extreme_temperature_condition | 436524 | 17.4069 | 39.9358 | 3.4594 | 1.2845 | 76.5% | 23.5% | 0.373 | 7.5% | 55.2% |
| label:extreme_wind_condition | 458693 | 24.2270 | 53.2716 | 4.3830 | 1.2114 | 78.1% | 21.9% | 0.244 | 10.1% | 65.5% |

## Daytime production-bin diagnostics

Bins use physical `y_true` in watts and only samples with target-time `solar_irradiance_poa > 10 W/m²`. Intervals are `[lower, upper)`, with the final bin `y_true >= 100 W`.

| bin | count | MAE | RMSE | mean_residual | median_residual | underprediction% | overprediction% | PICP PI | MPIW PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime_0_20 | 396207 | 20.1741 | 31.9481 | 18.6338 | 10.8833 | 21.6% | 78.4% | 0.527 | 37.0286 | 3.7% | 43.6% |
| daytime_20_40 | 376626 | 30.7828 | 50.0611 | 29.1348 | 17.5362 | 14.0% | 86.0% | 0.582 | 55.8122 | 0.7% | 41.1% |
| daytime_40_60 | 352255 | 37.6865 | 69.4966 | 32.6750 | 13.9132 | 26.5% | 73.5% | 0.643 | 61.7987 | 1.5% | 34.1% |
| daytime_60_80 | 266151 | 41.4597 | 70.7177 | 30.5571 | 12.1124 | 35.9% | 64.1% | 0.616 | 66.1003 | 4.6% | 33.8% |
| daytime_80_100 | 206451 | 43.5915 | 72.1258 | 26.0580 | 7.4101 | 41.5% | 58.5% | 0.602 | 68.1582 | 9.1% | 30.7% |
| daytime_gt_100 | 3149855 | 42.7747 | 67.2647 | -11.8315 | -9.9500 | 63.4% | 36.6% | 0.559 | 61.4938 | 29.6% | 14.5% |

## Automatic interpretation of residual asymmetry

- **Global:** overprediction; mean residual 1.7342 W, over 76.0%, under 24.0%.
- **Daytime:** overprediction; mean residual 1.2871 W, over 49.2%, under 50.8%.
- **Nighttime:** overprediction; mean residual 2.1354 W, over 99.9%, under 0.1%.
- **Unusually low solar potential:** overprediction; mean residual 107.8571 W, over 99.8%, under 0.2%.
- **Unusually high solar potential:** underprediction; mean residual -38.0434 W, over 13.9%, under 86.1%.
- **Rare/extreme daytime:** overprediction; mean residual 16.8904 W, over 59.5%, under 40.5%.
- **Production >= 100 W:** underprediction; mean residual -11.8315 W, over 36.6%, under 63.4%.
- **Nighttime softplus signature:** misses are predominantly below the PI while most targets are zero and most empirical lower bounds remain positive. This is consistent with `softplus` plus `y_true=0`.
- The model tends to **overpredict unusually low solar potential**.
- The model tends to **underpredict unusually high solar potential**.
- The model **underpredicts the >=100 W production bin**. Targets exceed `upper_pi` in 29.6% of these samples.
- The model overpredicts the low daytime production bin.
- **Global PI miss direction:** below 63.0%, above 9.8%; intervals/centres are predominantly too high.
- **Likely cause of low PICP:** softplus/zero-target nighttime misses, high-production targets above the PI. Centre bias and interval width should be interpreted together.

