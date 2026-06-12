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
- Device: cuda  |  Generated (UTC): 2026-06-11T14:30:21+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 21.7763 | 47.6383 | 8.8073 | 3.2501 | 21.1446 | 0.786 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 20.4692 | 44.4150 | 8.6248 | 2.8045 | 21.0641 | 0.796 | — |
| group:rare_or_extreme | 983379 | 33.8111 | 70.7202 | 10.4881 | 11.8931 | 21.7849 | 0.692 | — |
| label:unusually_low_solar_potential | 118147 | 102.5071 | 151.5909 | 17.9408 | 17.8564 | 24.5045 | 0.192 | — |
| label:unusually_high_solar_potential | 38401 | 85.1651 | 107.0737 | 15.7528 | 15.7498 | 20.3564 | 0.256 | — |
| label:extreme_temperature_condition | 436524 | 20.5379 | 43.3500 | 9.8460 | 10.6956 | 21.1499 | 0.788 | — |
| label:extreme_wind_condition | 458693 | 25.4724 | 54.4282 | 9.0668 | 3.9047 | 21.4365 | 0.758 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 20.4692  |  MAE rare/extreme: 33.8111  |  ratio: **1.65×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 20.4692  |  MAE rare/extreme: 33.8111  |  rare/normal MAE ratio: **1.65×**
- Mean uncertainty (std) normal: 8.6248  |  rare/extreme: 10.4881  |  rare/normal uncertainty ratio: **1.22×**
- Gaussian coverage@95 (diagnostic) normal: 0.796  |  rare/extreme: 0.692
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.65×).
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
| global | 0.238 | 30.0042 | 0.0336 | 41.6948 |
| normal | 0.238 | 29.3787 | 0.0329 | 40.8460 |
| rare_extreme | 0.238 | 35.7635 | 0.0400 | 49.4694 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.786 | 34.5247 | 0.0386 | 0.2385 |
| normal | 0.796 | 33.8091 | 0.0378 | 0.2146 |
| rare_extreme | 0.692 | 41.1133 | 0.0460 | 0.6550 |

## Daytime-only interval reliability

Eval-only split based on PVGIS `solar_irradiance_poa` at the target timestamp: daytime > **10.0 W/m²**, nighttime <= **10.0 W/m²**. The irradiance is diagnostic metadata and is not added to the model inputs or targets.

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | PICP PI | MPIW PI | NMPIL PI | CLC PI | PICP Gaussian | MPIW Gaussian | NMPIL Gaussian | CLC Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime | 4747545 | 43.5287 | 67.2341 | 17.0197 | 16.7451 | 23.8447 | 0.500 | 58.1793 | 0.0651 | 5.9302 | 0.572 | 66.7173 | 0.0747 | 3.3563 |
| nighttime | 5290119 | 2.2549 | 15.7880 | 1.4372 | 1.0421 | 2.4679 | 0.002 | 4.7189 | 0.0053 | 68.9669 | 0.978 | 5.6339 | 0.0063 | 0.0111 |
| normal_daytime | 4195053 | 42.3323 | 65.1824 | 17.0076 | 16.7355 | 23.8556 | 0.510 | 58.1382 | 0.0651 | 5.3487 | 0.581 | 66.6696 | 0.0746 | 3.0485 |
| rare_extreme_daytime | 552492 | 52.6136 | 81.1375 | 17.1120 | 16.8120 | 23.7608 | 0.421 | 58.4908 | 0.0655 | 13.0240 | 0.498 | 67.0790 | 0.0751 | 7.0057 |
| high_daytime | 1186852 | 52.1997 | 76.7237 | 14.6509 | 14.3217 | 18.7472 | 0.379 | 50.1846 | 0.0562 | 16.9830 | 0.427 | 57.4317 | 0.0643 | 12.1008 |
| peak_daytime | 474719 | 67.0463 | 86.2707 | 15.5986 | 15.4512 | 19.1230 | 0.200 | 53.3343 | 0.0597 | 107.5579 | 0.241 | 61.1467 | 0.0685 | 82.0986 |
| extreme_peak_daytime | 237361 | 89.9588 | 101.7163 | 16.0018 | 15.8699 | 19.5189 | 0.024 | 54.6979 | 0.0612 | 646.1687 | 0.031 | 62.7269 | 0.0702 | 689.3135 |

### Daytime production-tail diagnostics

| stratum | count | MAE | RMSE | mean residual | median residual | fraction underprediction | fraction above PI | PICP PI | PICP Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high_daytime | 1186852 | 52.1997 | 76.7237 | -36.3346 | -22.4500 | 0.680 | 0.488 | 0.379 | 0.427 |
| peak_daytime | 474719 | 67.0463 | 86.2707 | -66.0626 | -57.3544 | 0.964 | 0.794 | 0.200 | 0.241 |
| extreme_peak_daytime | 237361 | 89.9588 | 101.7163 | -89.9355 | -80.9165 | 0.998 | 0.976 | 0.024 | 0.031 |

| stratum | fraction y_true=0 | fraction lower PI <= 0 | fraction lower Gaussian <= 0 | PI coverage y=0 | Gaussian coverage y=0 | PI coverage y>0 | Gaussian coverage y>0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| daytime | 0.000 | 0.000 | 0.036 | — | — | 0.500 | 0.572 |
| nighttime | 0.988 | 0.000 | 0.977 | 0.000 | 0.981 | 0.186 | 0.721 |
| normal_daytime | 0.000 | 0.000 | 0.037 | — | — | 0.510 | 0.581 |
| rare_extreme_daytime | 0.000 | 0.000 | 0.032 | — | — | 0.421 | 0.498 |
| high_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.379 | 0.427 |
| peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.200 | 0.241 |
| extreme_peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.024 | 0.031 |

**PICP PI daytime is materially higher than global** (0.500 vs 0.238, delta 0.262).
It nevertheless remains low relative to the 0.95 coverage target.

## Residual bias diagnostics by stratum

`residual = y_pred_mean - y_true`: positive means overprediction, negative means underprediction.

| stratum | count | MAE | RMSE | mean_residual | median_residual | overprediction% | underprediction% | PICP PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 10037664 | 21.7763 | 47.6383 | 0.7412 | 1.0938 | 76.6% | 23.4% | 0.238 | 11.6% | 64.6% |
| daytime | 4747545 | 43.5287 | 67.2341 | -0.9454 | 0.4187 | 50.5% | 49.5% | 0.500 | 24.5% | 25.5% |
| nighttime | 5290119 | 2.2549 | 15.7880 | 2.2548 | 1.1015 | 100.0% | 0.0% | 0.002 | 0.0% | 99.8% |
| normal | 9054285 | 20.4692 | 44.4150 | -0.1480 | 1.0844 | 76.7% | 23.3% | 0.238 | 11.4% | 64.8% |
| rare_extreme | 983379 | 33.8111 | 70.7202 | 8.9281 | 1.2216 | 75.3% | 24.7% | 0.238 | 13.2% | 63.0% |
| normal_daytime | 4195053 | 42.3323 | 65.1824 | -2.1663 | -0.1514 | 49.8% | 50.2% | 0.510 | 24.7% | 24.3% |
| rare_extreme_daytime | 552492 | 52.6136 | 81.1375 | 8.3245 | 5.9013 | 56.0% | 44.0% | 0.421 | 23.5% | 34.4% |
| normal_nighttime | 4859232 | 1.5945 | 2.7826 | 1.5944 | 1.0980 | 100.0% | 0.0% | 0.002 | 0.0% | 99.8% |
| rare_extreme_nighttime | 430887 | 9.7022 | 54.5247 | 9.7021 | 1.1463 | 100.0% | 0.0% | 0.003 | 0.0% | 99.7% |
| label:unusually_low_solar_potential | 118147 | 102.5071 | 151.5909 | 102.3462 | 58.9612 | 99.2% | 0.8% | 0.070 | 0.0% | 93.0% |
| label:unusually_high_solar_potential | 38401 | 85.1651 | 107.0737 | -82.1029 | -82.6549 | 11.2% | 88.8% | 0.237 | 74.4% | 2.0% |
| label:extreme_temperature_condition | 436524 | 20.5379 | 43.3500 | -0.3878 | 1.1151 | 72.7% | 27.3% | 0.309 | 12.3% | 56.9% |
| label:extreme_wind_condition | 458693 | 25.4724 | 54.4282 | 1.7545 | 1.1272 | 77.1% | 22.9% | 0.220 | 12.3% | 65.7% |

## Daytime production-bin diagnostics

Bins use physical `y_true` in watts and only samples with target-time `solar_irradiance_poa > 10 W/m²`. Intervals are `[lower, upper)`, with the final bin `y_true >= 100 W`.

| bin | count | MAE | RMSE | mean_residual | median_residual | underprediction% | overprediction% | PICP PI | MPIW PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime_0_20 | 396207 | 21.4315 | 33.5137 | 20.0564 | 13.2482 | 17.8% | 82.2% | 0.466 | 36.4934 | 3.2% | 50.2% |
| daytime_20_40 | 376626 | 33.2412 | 51.6104 | 30.2276 | 20.3415 | 16.7% | 83.3% | 0.504 | 53.9516 | 3.0% | 46.7% |
| daytime_40_60 | 352255 | 39.3014 | 68.0837 | 33.2821 | 16.5043 | 22.9% | 77.1% | 0.577 | 60.5530 | 4.4% | 37.9% |
| daytime_60_80 | 266151 | 43.7634 | 72.0342 | 32.0785 | 13.6455 | 33.9% | 66.1% | 0.565 | 65.2404 | 7.3% | 36.2% |
| daytime_80_100 | 206451 | 45.9074 | 73.4077 | 26.7840 | 8.5520 | 41.7% | 58.3% | 0.563 | 67.4728 | 10.9% | 32.8% |
| daytime_gt_100 | 3149855 | 47.8354 | 71.0177 | -15.7501 | -12.4138 | 62.2% | 37.8% | 0.486 | 59.9413 | 34.4% | 17.1% |

## Automatic interpretation of residual asymmetry

- **Global:** overprediction; mean residual 0.7412 W, over 76.6%, under 23.4%.
- **Daytime:** underprediction; mean residual -0.9454 W, over 50.5%, under 49.5%.
- **Nighttime:** overprediction; mean residual 2.2548 W, over 100.0%, under 0.0%.
- **Unusually low solar potential:** overprediction; mean residual 102.3462 W, over 99.2%, under 0.8%.
- **Unusually high solar potential:** underprediction; mean residual -82.1029 W, over 11.2%, under 88.8%.
- **Rare/extreme daytime:** overprediction; mean residual 8.3245 W, over 56.0%, under 44.0%.
- **Production >= 100 W:** underprediction; mean residual -15.7501 W, over 37.8%, under 62.2%.
- **Nighttime softplus signature:** misses are predominantly below the PI while most targets are zero and most empirical lower bounds remain positive. This is consistent with `softplus` plus `y_true=0`.
- The model tends to **overpredict unusually low solar potential**.
- The model tends to **underpredict unusually high solar potential**.
- The model **underpredicts the >=100 W production bin**. Targets exceed `upper_pi` in 34.4% of these samples.
- The model overpredicts the low daytime production bin.
- **Global PI miss direction:** below 64.6%, above 11.6%; intervals/centres are predominantly too high.
- **Likely cause of low PICP:** softplus/zero-target nighttime misses, high-production targets above the PI. Centre bias and interval width should be interpreted together.

