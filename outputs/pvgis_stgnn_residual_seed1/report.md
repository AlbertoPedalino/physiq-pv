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
- PV normalized target upper clip: **1.5**
- Selected features (11): temperature_2m, solar_irradiance_poa, wind_speed_10m, sin_elev, cos_elev, kt, kt_std_3h, dghi_dt, dni_norm, dhi_norm, pv_lag_pvgis
- MC samples: **20**
- seq_len: **24**  |  horizon: **1**
- Train years: 2016,2017,2018
- Test year: **2019**
- Nodes (locations): **1149**  |  epochs: **5**
- batch_size: 16  |  lr: 0.001
- Anomaly scores: outputs/pvgis_anomaly_2019_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Predictions: **10037664**
- Device: cuda  |  Generated (UTC): 2026-06-10T17:45:11+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 20.3935 | 46.2292 | 7.2354 | 2.4964 | 17.5283 | 0.760 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 19.3614 | 43.4516 | 7.0820 | 2.3501 | 17.4171 | 0.768 | — |
| group:rare_or_extreme | 983379 | 29.8960 | 66.5635 | 8.6475 | 8.9271 | 18.4096 | 0.684 | — |
| label:unusually_low_solar_potential | 118147 | 92.6853 | 144.4260 | 13.9401 | 13.8953 | 19.8138 | 0.187 | — |
| label:unusually_high_solar_potential | 38401 | 50.0998 | 73.3766 | 16.6975 | 16.0847 | 25.5549 | 0.431 | — |
| label:extreme_temperature_condition | 436524 | 18.3364 | 40.7077 | 8.0461 | 7.9373 | 17.6581 | 0.771 | — |
| label:extreme_wind_condition | 458693 | 23.6550 | 53.2847 | 7.4325 | 2.6897 | 17.7247 | 0.744 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 19.3614  |  MAE rare/extreme: 29.8960  |  ratio: **1.54×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 19.3614  |  MAE rare/extreme: 29.8960  |  rare/normal MAE ratio: **1.54×**
- Mean uncertainty (std) normal: 7.0820  |  rare/extreme: 8.6475  |  rare/normal uncertainty ratio: **1.22×**
- Gaussian coverage@95 (diagnostic) normal: 0.768  |  rare/extreme: 0.684
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.54×).
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
| global | 0.217 | 24.5801 | 0.0275 | 42.0381 |
| normal | 0.215 | 24.0527 | 0.0269 | 41.8590 |
| rare_extreme | 0.233 | 29.4359 | 0.0330 | 42.8834 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.760 | 28.3626 | 0.0318 | 0.2451 |
| normal | 0.768 | 27.7614 | 0.0311 | 0.2233 |
| rare_extreme | 0.684 | 33.8981 | 0.0379 | 0.5824 |

## Daytime-only interval reliability

Eval-only split based on PVGIS `solar_irradiance_poa` at the target timestamp: daytime > **10.0 W/m²**, nighttime <= **10.0 W/m²**. The irradiance is diagnostic metadata and is not added to the model inputs or targets.

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | PICP PI | MPIW PI | NMPIL PI | CLC PI | PICP Gaussian | MPIW Gaussian | NMPIL Gaussian | CLC Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime | 4747545 | 41.0931 | 65.5100 | 13.6791 | 13.4766 | 19.9462 | 0.452 | 46.7753 | 0.0524 | 7.6666 | 0.507 | 53.6220 | 0.0600 | 5.1179 |
| nighttime | 5290119 | 1.8169 | 14.2719 | 1.4526 | 1.2869 | 2.2851 | 0.006 | 4.6614 | 0.0052 | 65.7289 | 0.986 | 5.6940 | 0.0064 | 0.0108 |
| normal_daytime | 4195053 | 40.3532 | 63.8118 | 13.6419 | 13.4494 | 19.9051 | 0.458 | 46.6492 | 0.0522 | 7.2282 | 0.511 | 53.4762 | 0.0599 | 4.8879 |
| rare_extreme_daytime | 552492 | 46.7113 | 77.1951 | 13.9614 | 13.6751 | 20.2692 | 0.409 | 47.7325 | 0.0534 | 11.9924 | 0.473 | 54.7287 | 0.0613 | 7.2595 |
| high_daytime | 1186852 | 42.9724 | 68.6347 | 12.5077 | 11.4533 | 18.3174 | 0.412 | 42.8557 | 0.0480 | 10.4505 | 0.469 | 49.0301 | 0.0549 | 6.7714 |
| peak_daytime | 474719 | 42.9521 | 67.5461 | 14.7998 | 13.8016 | 21.4532 | 0.444 | 50.6306 | 0.0567 | 8.9546 | 0.516 | 58.0152 | 0.0649 | 5.0667 |
| extreme_peak_daytime | 237361 | 45.2435 | 69.6577 | 17.2605 | 16.7068 | 23.8260 | 0.486 | 59.0260 | 0.0661 | 6.9178 | 0.562 | 67.6610 | 0.0757 | 3.7613 |

### Daytime production-tail diagnostics

| stratum | count | MAE | RMSE | mean residual | median residual | fraction underprediction | fraction above PI | PICP PI | PICP Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high_daytime | 1186852 | 42.9724 | 68.6347 | -31.5442 | -20.2670 | 0.784 | 0.499 | 0.412 | 0.469 |
| peak_daytime | 474719 | 42.9521 | 67.5461 | -38.1780 | -25.7950 | 0.874 | 0.528 | 0.444 | 0.516 |
| extreme_peak_daytime | 237361 | 45.2435 | 69.6577 | -42.1238 | -28.9521 | 0.901 | 0.503 | 0.486 | 0.562 |

| stratum | fraction y_true=0 | fraction lower PI <= 0 | fraction lower Gaussian <= 0 | PI coverage y=0 | Gaussian coverage y=0 | PI coverage y>0 | Gaussian coverage y>0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| daytime | 0.000 | 0.000 | 0.034 | — | — | 0.452 | 0.507 |
| nighttime | 0.988 | 0.000 | 0.986 | 0.000 | 0.989 | 0.473 | 0.810 |
| normal_daytime | 0.000 | 0.000 | 0.034 | — | — | 0.458 | 0.511 |
| rare_extreme_daytime | 0.000 | 0.000 | 0.029 | — | — | 0.409 | 0.473 |
| high_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.412 | 0.469 |
| peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.444 | 0.516 |
| extreme_peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.486 | 0.562 |

**PICP PI daytime is materially higher than global** (0.452 vs 0.217, delta 0.235).
It nevertheless remains low relative to the 0.95 coverage target.

## Residual bias diagnostics by stratum

`residual = y_pred_mean - y_true`: positive means overprediction, negative means underprediction.

| stratum | count | MAE | RMSE | mean_residual | median_residual | overprediction% | underprediction% | PICP PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 10037664 | 20.3935 | 46.2292 | -2.6100 | 0.9931 | 70.4% | 29.6% | 0.217 | 16.1% | 62.2% |
| daytime | 4747545 | 41.0931 | 65.5100 | -7.5415 | -8.5449 | 37.6% | 62.4% | 0.452 | 34.1% | 20.7% |
| nighttime | 5290119 | 1.8169 | 14.2719 | 1.8157 | 1.1065 | 99.9% | 0.1% | 0.006 | 0.0% | 99.4% |
| normal | 9054285 | 19.3614 | 43.4516 | -3.5829 | 0.9865 | 70.5% | 29.5% | 0.215 | 16.1% | 62.4% |
| rare_extreme | 983379 | 29.8960 | 66.5635 | 6.3475 | 1.0711 | 69.4% | 30.6% | 0.233 | 16.3% | 60.4% |
| normal_daytime | 4195053 | 40.3532 | 63.8118 | -9.1667 | -9.1525 | 36.5% | 63.5% | 0.458 | 34.7% | 19.5% |
| rare_extreme_daytime | 552492 | 46.7113 | 77.1951 | 4.7984 | -3.3319 | 45.6% | 54.4% | 0.409 | 29.0% | 30.1% |
| normal_nighttime | 4859232 | 1.2389 | 1.6236 | 1.2377 | 1.1035 | 99.9% | 0.1% | 0.006 | 0.0% | 99.4% |
| rare_extreme_nighttime | 430887 | 8.3351 | 49.7091 | 8.3338 | 1.1427 | 99.8% | 0.2% | 0.007 | 0.0% | 99.3% |
| label:unusually_low_solar_potential | 118147 | 92.6853 | 144.4260 | 92.2333 | 47.4185 | 98.4% | 1.6% | 0.081 | 0.3% | 91.6% |
| label:unusually_high_solar_potential | 38401 | 50.0998 | 73.3766 | -49.0089 | -36.3984 | 4.8% | 95.2% | 0.358 | 63.6% | 0.5% |
| label:extreme_temperature_condition | 436524 | 18.3364 | 40.7077 | -4.6804 | 0.9350 | 63.9% | 36.1% | 0.294 | 17.6% | 53.0% |
| label:extreme_wind_condition | 458693 | 23.6550 | 53.2847 | -0.2584 | 1.0495 | 72.7% | 27.3% | 0.211 | 15.0% | 63.9% |

## Daytime production-bin diagnostics

Bins use physical `y_true` in watts and only samples with target-time `solar_irradiance_poa > 10 W/m²`. Intervals are `[lower, upper)`, with the final bin `y_true >= 100 W`.

| bin | count | MAE | RMSE | mean_residual | median_residual | underprediction% | overprediction% | PICP PI | MPIW PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime_0_20 | 396207 | 15.9818 | 27.4097 | 11.6808 | 4.1085 | 39.9% | 60.1% | 0.429 | 23.1514 | 19.3% | 37.9% |
| daytime_20_40 | 376626 | 25.9642 | 45.4933 | 20.1706 | 9.0089 | 30.9% | 69.1% | 0.535 | 39.5008 | 8.4% | 38.1% |
| daytime_40_60 | 352255 | 34.1646 | 65.2609 | 22.9288 | 4.6079 | 43.3% | 56.7% | 0.568 | 46.1461 | 10.3% | 32.8% |
| daytime_60_80 | 266151 | 38.1997 | 67.0700 | 20.3106 | 2.4973 | 47.0% | 53.0% | 0.534 | 51.4357 | 16.6% | 30.0% |
| daytime_80_100 | 206451 | 41.6751 | 69.3297 | 14.1986 | -3.3375 | 53.4% | 46.6% | 0.507 | 53.8264 | 23.6% | 25.7% |
| daytime_gt_100 | 3149855 | 47.0419 | 70.3915 | -20.4589 | -19.0907 | 73.0% | 27.0% | 0.421 | 49.8311 | 43.8% | 14.0% |

## Automatic interpretation of residual asymmetry

- **Global:** underprediction; mean residual -2.6100 W, over 70.4%, under 29.6%.
- **Daytime:** underprediction; mean residual -7.5415 W, over 37.6%, under 62.4%.
- **Nighttime:** overprediction; mean residual 1.8157 W, over 99.9%, under 0.1%.
- **Unusually low solar potential:** overprediction; mean residual 92.2333 W, over 98.4%, under 1.6%.
- **Unusually high solar potential:** underprediction; mean residual -49.0089 W, over 4.8%, under 95.2%.
- **Rare/extreme daytime:** overprediction; mean residual 4.7984 W, over 45.6%, under 54.4%.
- **Production >= 100 W:** underprediction; mean residual -20.4589 W, over 27.0%, under 73.0%.
- **Nighttime softplus signature:** misses are predominantly below the PI while most targets are zero and most empirical lower bounds remain positive. This is consistent with `softplus` plus `y_true=0`.
- The model tends to **overpredict unusually low solar potential**.
- The model tends to **underpredict unusually high solar potential**.
- The model **underpredicts the >=100 W production bin**. Targets exceed `upper_pi` in 43.8% of these samples.
- The active normalized-target upper clip is consistent with a peak-smoothing hypothesis, but this diagnostic is observational and does not establish causality.
- The model overpredicts the low daytime production bin.
- **Global PI miss direction:** below 62.2%, above 16.1%; intervals/centres are predominantly too high.
- **Likely cause of low PICP:** softplus/zero-target nighttime misses, high-production targets above the PI. Centre bias and interval width should be interpreted together.

