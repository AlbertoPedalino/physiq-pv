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
- Device: cuda  |  Generated (UTC): 2026-06-11T11:34:36+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 19.8950 | 45.8203 | 8.6771 | 2.9822 | 21.3193 | 0.807 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 18.8945 | 43.0492 | 8.4825 | 2.7755 | 21.1662 | 0.813 | — |
| group:rare_or_extreme | 983379 | 29.1072 | 66.0828 | 10.4689 | 10.5858 | 22.5730 | 0.748 | — |
| label:unusually_low_solar_potential | 118147 | 91.4734 | 143.9204 | 16.8091 | 16.8251 | 23.9790 | 0.307 | — |
| label:unusually_high_solar_potential | 38401 | 49.3832 | 72.5381 | 21.5249 | 21.0203 | 33.0325 | 0.561 | — |
| label:extreme_temperature_condition | 436524 | 17.1465 | 39.6810 | 9.7667 | 9.2744 | 21.7572 | 0.841 | — |
| label:extreme_wind_condition | 458693 | 23.2982 | 53.1047 | 8.8779 | 3.2002 | 21.4841 | 0.787 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 18.8945  |  MAE rare/extreme: 29.1072  |  ratio: **1.54×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 18.8945  |  MAE rare/extreme: 29.1072  |  rare/normal MAE ratio: **1.54×**
- Mean uncertainty (std) normal: 8.4825  |  rare/extreme: 10.4689  |  rare/normal uncertainty ratio: **1.23×**
- Gaussian coverage@95 (diagnostic) normal: 0.813  |  rare/extreme: 0.748
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.54×).
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
| global | 0.260 | 29.4579 | 0.0330 | 32.6329 |
| normal | 0.257 | 28.7892 | 0.0322 | 32.9120 |
| rare_extreme | 0.289 | 35.6152 | 0.0399 | 29.5282 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.807 | 34.0142 | 0.0381 | 0.1978 |
| normal | 0.813 | 33.2513 | 0.0372 | 0.1838 |
| rare_extreme | 0.748 | 41.0382 | 0.0459 | 0.3913 |

## Daytime-only interval reliability

Eval-only split based on PVGIS `solar_irradiance_poa` at the target timestamp: daytime > **10.0 W/m²**, nighttime <= **10.0 W/m²**. The irradiance is diagnostic metadata and is not added to the model inputs or targets.

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | PICP PI | MPIW PI | NMPIL PI | CLC PI | PICP Gaussian | MPIW Gaussian | NMPIL Gaussian | CLC Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime | 4747545 | 40.0523 | 64.8877 | 16.5374 | 16.3222 | 24.3098 | 0.543 | 56.5200 | 0.0633 | 3.7645 | 0.596 | 64.8267 | 0.0726 | 2.5783 |
| nighttime | 5290119 | 1.8052 | 14.3208 | 1.6229 | 1.3716 | 2.6702 | 0.007 | 5.1714 | 0.0058 | 72.3754 | 0.996 | 6.3619 | 0.0071 | 0.0116 |
| normal_daytime | 4195053 | 39.3615 | 63.2229 | 16.4732 | 16.2747 | 24.2262 | 0.548 | 56.3016 | 0.0630 | 3.5834 | 0.598 | 64.5748 | 0.0723 | 2.5103 |
| rare_extreme_daytime | 552492 | 45.2975 | 76.3532 | 17.0255 | 16.6800 | 24.9702 | 0.508 | 58.1786 | 0.0651 | 5.4777 | 0.578 | 66.7398 | 0.0747 | 3.1592 |
| high_daytime | 1186852 | 41.6973 | 67.7560 | 15.6967 | 14.1328 | 24.0453 | 0.530 | 53.7945 | 0.0602 | 4.0649 | 0.581 | 61.5309 | 0.0689 | 2.8302 |
| peak_daytime | 474719 | 41.5567 | 66.7233 | 19.7716 | 18.7082 | 28.4001 | 0.612 | 67.7116 | 0.0758 | 2.3117 | 0.666 | 77.5048 | 0.0868 | 1.5720 |
| extreme_peak_daytime | 237361 | 44.1195 | 69.0837 | 23.0482 | 22.5755 | 31.3032 | 0.644 | 78.9322 | 0.0884 | 1.9754 | 0.699 | 90.3490 | 0.1011 | 1.3511 |

### Daytime production-tail diagnostics

| stratum | count | MAE | RMSE | mean residual | median residual | fraction underprediction | fraction above PI | PICP PI | PICP Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high_daytime | 1186852 | 41.6973 | 67.7560 | -30.0435 | -18.2792 | 0.774 | 0.394 | 0.530 | 0.581 |
| peak_daytime | 474719 | 41.5567 | 66.7233 | -36.4472 | -23.7462 | 0.861 | 0.369 | 0.612 | 0.666 |
| extreme_peak_daytime | 237361 | 44.1195 | 69.0837 | -40.7855 | -27.5990 | 0.891 | 0.350 | 0.644 | 0.699 |

| stratum | fraction y_true=0 | fraction lower PI <= 0 | fraction lower Gaussian <= 0 | PI coverage y=0 | Gaussian coverage y=0 | PI coverage y>0 | Gaussian coverage y>0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| daytime | 0.000 | 0.000 | 0.060 | — | — | 0.543 | 0.596 |
| nighttime | 0.988 | 0.000 | 0.995 | 0.000 | 0.997 | 0.533 | 0.915 |
| normal_daytime | 0.000 | 0.000 | 0.061 | — | — | 0.548 | 0.598 |
| rare_extreme_daytime | 0.000 | 0.000 | 0.050 | — | — | 0.508 | 0.578 |
| high_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.530 | 0.581 |
| peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.612 | 0.666 |
| extreme_peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.644 | 0.699 |

**PICP PI daytime is materially higher than global** (0.543 vs 0.260, delta 0.283).
It nevertheless remains low relative to the 0.95 coverage target.

## Residual bias diagnostics by stratum

`residual = y_pred_mean - y_true`: positive means overprediction, negative means underprediction.

| stratum | count | MAE | RMSE | mean_residual | median_residual | overprediction% | underprediction% | PICP PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 10037664 | 19.8950 | 45.8203 | -1.4430 | 0.9893 | 71.7% | 28.3% | 0.260 | 12.8% | 61.2% |
| daytime | 4747545 | 40.0523 | 64.8877 | -5.0606 | -6.0623 | 40.4% | 59.6% | 0.543 | 27.0% | 18.7% |
| nighttime | 5290119 | 1.8052 | 14.3208 | 1.8035 | 1.0876 | 99.8% | 0.2% | 0.007 | 0.0% | 99.3% |
| normal | 9054285 | 18.8945 | 43.0492 | -2.3977 | 0.9822 | 71.8% | 28.2% | 0.257 | 12.8% | 61.4% |
| rare_extreme | 983379 | 29.1072 | 66.0828 | 7.3465 | 1.0740 | 70.7% | 29.3% | 0.289 | 12.4% | 58.7% |
| normal_daytime | 4195053 | 39.3615 | 63.2229 | -6.5920 | -6.6103 | 39.4% | 60.6% | 0.548 | 27.7% | 17.5% |
| rare_extreme_daytime | 552492 | 45.2975 | 76.3532 | 6.5671 | -1.4965 | 48.0% | 52.0% | 0.508 | 22.0% | 27.2% |
| normal_nighttime | 4859232 | 1.2250 | 1.5381 | 1.2234 | 1.0848 | 99.8% | 0.2% | 0.006 | 0.0% | 99.4% |
| rare_extreme_nighttime | 430887 | 8.3476 | 49.9120 | 8.3458 | 1.1213 | 99.8% | 0.2% | 0.009 | 0.0% | 99.1% |
| label:unusually_low_solar_potential | 118147 | 91.4734 | 143.9204 | 90.8097 | 46.4417 | 97.7% | 2.3% | 0.139 | 0.3% | 85.8% |
| label:unusually_high_solar_potential | 38401 | 49.3832 | 72.5381 | -48.2142 | -35.7738 | 4.8% | 95.2% | 0.498 | 49.9% | 0.3% |
| label:extreme_temperature_condition | 436524 | 17.1465 | 39.6810 | -3.1215 | 0.9408 | 65.9% | 34.1% | 0.364 | 12.0% | 51.6% |
| label:extreme_wind_condition | 458693 | 23.2982 | 53.1047 | 0.5815 | 1.0413 | 73.7% | 26.3% | 0.251 | 12.5% | 62.4% |

## Daytime production-bin diagnostics

Bins use physical `y_true` in watts and only samples with target-time `solar_irradiance_poa > 10 W/m²`. Intervals are `[lower, upper)`, with the final bin `y_true >= 100 W`.

| bin | count | MAE | RMSE | mean_residual | median_residual | underprediction% | overprediction% | PICP PI | MPIW PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime_0_20 | 396207 | 15.2712 | 27.4204 | 10.4720 | 1.1853 | 46.0% | 54.0% | 0.545 | 27.2202 | 17.6% | 28.0% |
| daytime_20_40 | 376626 | 26.5082 | 45.4657 | 19.8721 | 9.4783 | 32.4% | 67.6% | 0.592 | 48.5605 | 8.1% | 32.8% |
| daytime_40_60 | 352255 | 34.4116 | 65.2948 | 23.8757 | 6.0706 | 41.2% | 58.8% | 0.643 | 57.2735 | 6.6% | 29.1% |
| daytime_60_80 | 266151 | 38.4540 | 67.6253 | 22.4554 | 4.7673 | 44.3% | 55.7% | 0.628 | 63.6765 | 10.0% | 27.3% |
| daytime_80_100 | 206451 | 41.9598 | 69.9539 | 16.8764 | -0.3233 | 50.3% | 49.7% | 0.599 | 66.3311 | 16.2% | 23.9% |
| daytime_gt_100 | 3149855 | 45.4297 | 69.4288 | -16.9944 | -15.4023 | 68.6% | 31.4% | 0.515 | 59.8252 | 34.9% | 13.6% |

## Automatic interpretation of residual asymmetry

- **Global:** underprediction; mean residual -1.4430 W, over 71.7%, under 28.3%.
- **Daytime:** underprediction; mean residual -5.0606 W, over 40.4%, under 59.6%.
- **Nighttime:** overprediction; mean residual 1.8035 W, over 99.8%, under 0.2%.
- **Unusually low solar potential:** overprediction; mean residual 90.8097 W, over 97.7%, under 2.3%.
- **Unusually high solar potential:** underprediction; mean residual -48.2142 W, over 4.8%, under 95.2%.
- **Rare/extreme daytime:** overprediction; mean residual 6.5671 W, over 48.0%, under 52.0%.
- **Production >= 100 W:** underprediction; mean residual -16.9944 W, over 31.4%, under 68.6%.
- **Nighttime softplus signature:** misses are predominantly below the PI while most targets are zero and most empirical lower bounds remain positive. This is consistent with `softplus` plus `y_true=0`.
- The model tends to **overpredict unusually low solar potential**.
- The model tends to **underpredict unusually high solar potential**.
- The model **underpredicts the >=100 W production bin**. Targets exceed `upper_pi` in 34.9% of these samples.
- The model overpredicts the low daytime production bin.
- **Global PI miss direction:** below 61.2%, above 12.8%; intervals/centres are predominantly too high.
- **Likely cause of low PICP:** softplus/zero-target nighttime misses, high-production targets above the PI. Centre bias and interval width should be interpreted together.

