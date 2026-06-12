# PVGIS-only ST-GNN forecasting report

Reuses the existing STGNN architecture on a **PVGIS-only** input. No real plant production, no ENERGIA, no quality score, no kWp/UPN. Anomaly labels are used **only** for stratified evaluation.

## Experiment

- Mode: **pvgis_stgnn**
- Model type: **stgnn_enhanced_dropout**
- Feature set: **no_pv_lag**
- W&B enabled: **False**
- MC Dropout: **enabled** (experimental)

## Parameters

- Target variable: **pv_power_output**
- PV normalized target upper clip: **none**
- PV normalized target lower clip: **0.0**
- Selected features (10): temperature_2m, solar_irradiance_poa, wind_speed_10m, sin_elev, cos_elev, kt, kt_std_3h, dghi_dt, dni_norm, dhi_norm
- MC samples: **20**
- seq_len: **24**  |  horizon: **1**
- Train years: 2016,2017,2018
- Test year: **2019**
- Nodes (locations): **1149**  |  epochs: **5**
- batch_size: 16  |  lr: 0.001
- Anomaly scores: outputs/pvgis_anomaly_2019_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Predictions: **10037664**
- Device: cuda  |  Generated (UTC): 2026-06-11T10:48:41+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 22.2783 | 47.3943 | 8.3802 | 2.5886 | 20.8114 | 0.757 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 21.2304 | 44.6586 | 8.1764 | 2.3999 | 20.6227 | 0.763 | — |
| group:rare_or_extreme | 983379 | 31.9264 | 67.5640 | 10.2571 | 10.4970 | 22.3267 | 0.700 | — |
| label:unusually_low_solar_potential | 118147 | 91.7179 | 144.6682 | 16.4451 | 16.4937 | 23.5819 | 0.363 | — |
| label:unusually_high_solar_potential | 38401 | 72.5989 | 90.8841 | 21.4609 | 21.6291 | 31.2911 | 0.234 | — |
| label:extreme_temperature_condition | 436524 | 19.9253 | 41.2427 | 9.6608 | 9.0821 | 21.7864 | 0.780 | — |
| label:extreme_wind_condition | 458693 | 25.2554 | 53.8380 | 8.6283 | 2.8661 | 21.0478 | 0.748 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 21.2304  |  MAE rare/extreme: 31.9264  |  ratio: **1.50×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 21.2304  |  MAE rare/extreme: 31.9264  |  rare/normal MAE ratio: **1.50×**
- Mean uncertainty (std) normal: 8.1764  |  rare/extreme: 10.2571  |  rare/normal uncertainty ratio: **1.25×**
- Gaussian coverage@95 (diagnostic) normal: 0.763  |  rare/extreme: 0.700
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.50×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.25×).

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
| global | 0.213 | 28.5027 | 0.0319 | 50.4397 |
| normal | 0.211 | 27.8040 | 0.0311 | 50.5279 |
| rare_extreme | 0.238 | 34.9355 | 0.0391 | 48.4102 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.757 | 32.8504 | 0.0368 | 0.2912 |
| normal | 0.763 | 32.0513 | 0.0359 | 0.2693 |
| rare_extreme | 0.700 | 40.2079 | 0.0450 | 0.5937 |

## Daytime-only interval reliability

Eval-only split based on PVGIS `solar_irradiance_poa` at the target timestamp: daytime > **10.0 W/m²**, nighttime <= **10.0 W/m²**. The irradiance is diagnostic metadata and is not added to the model inputs or targets.

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | PICP PI | MPIW PI | NMPIL PI | CLC PI | PICP Gaussian | MPIW Gaussian | NMPIL Gaussian | CLC Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime | 4747545 | 45.0628 | 67.2769 | 16.1761 | 15.9990 | 23.8048 | 0.443 | 55.2632 | 0.0619 | 9.8634 | 0.493 | 63.4102 | 0.0710 | 6.9175 |
| nighttime | 5290119 | 1.8306 | 14.1456 | 1.3839 | 1.1274 | 2.2917 | 0.007 | 4.4868 | 0.0050 | 62.5034 | 0.993 | 5.4249 | 0.0061 | 0.0100 |
| normal_daytime | 4195053 | 44.3622 | 65.5903 | 16.0858 | 15.9223 | 23.6870 | 0.447 | 54.9553 | 0.0615 | 9.4744 | 0.493 | 63.0564 | 0.0706 | 6.8750 |
| rare_extreme_daytime | 552492 | 50.3829 | 78.9160 | 16.8614 | 16.5939 | 24.6903 | 0.417 | 57.6009 | 0.0645 | 13.3780 | 0.493 | 66.0968 | 0.0740 | 7.2416 |
| high_daytime | 1186852 | 56.4718 | 77.5119 | 17.2843 | 16.1472 | 25.2456 | 0.315 | 59.2299 | 0.0663 | 37.8517 | 0.381 | 67.7545 | 0.0758 | 22.5744 |
| peak_daytime | 474719 | 58.3625 | 78.9404 | 21.2765 | 20.7699 | 28.4859 | 0.384 | 72.8637 | 0.0816 | 23.4914 | 0.472 | 83.4040 | 0.0934 | 11.1644 |
| extreme_peak_daytime | 237361 | 64.1235 | 83.3753 | 23.7937 | 23.6343 | 30.5892 | 0.357 | 81.4857 | 0.0912 | 34.5479 | 0.456 | 93.2715 | 0.1044 | 14.7298 |

### Daytime production-tail diagnostics

| stratum | count | MAE | RMSE | mean residual | median residual | fraction underprediction | fraction above PI | PICP PI | PICP Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high_daytime | 1186852 | 56.4718 | 77.5119 | -50.4536 | -39.2607 | 0.914 | 0.648 | 0.315 | 0.381 |
| peak_daytime | 474719 | 58.3625 | 78.9404 | -56.8580 | -43.8338 | 0.961 | 0.611 | 0.384 | 0.472 |
| extreme_peak_daytime | 237361 | 64.1235 | 83.3753 | -63.5974 | -50.2823 | 0.981 | 0.643 | 0.357 | 0.456 |

| stratum | fraction y_true=0 | fraction lower PI <= 0 | fraction lower Gaussian <= 0 | PI coverage y=0 | Gaussian coverage y=0 | PI coverage y>0 | Gaussian coverage y>0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| daytime | 0.000 | 0.000 | 0.081 | — | — | 0.443 | 0.493 |
| nighttime | 0.988 | 0.000 | 0.993 | 0.000 | 0.994 | 0.571 | 0.938 |
| normal_daytime | 0.000 | 0.000 | 0.082 | — | — | 0.447 | 0.493 |
| rare_extreme_daytime | 0.000 | 0.000 | 0.066 | — | — | 0.417 | 0.493 |
| high_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.315 | 0.381 |
| peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.384 | 0.472 |
| extreme_peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.357 | 0.456 |

**PICP PI daytime is materially higher than global** (0.443 vs 0.213, delta 0.230).
It nevertheless remains low relative to the 0.95 coverage target.

## Residual bias diagnostics by stratum

`residual = y_pred_mean - y_true`: positive means overprediction, negative means underprediction.

| stratum | count | MAE | RMSE | mean_residual | median_residual | overprediction% | underprediction% | PICP PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 10037664 | 22.2783 | 47.3943 | -6.8293 | 1.0049 | 65.9% | 34.1% | 0.213 | 19.2% | 59.5% |
| daytime | 4747545 | 45.0628 | 67.2769 | -16.4773 | -18.4611 | 28.2% | 71.8% | 0.443 | 40.5% | 15.1% |
| nighttime | 5290119 | 1.8306 | 14.1456 | 1.8291 | 1.1653 | 99.8% | 0.2% | 0.007 | 0.0% | 99.3% |
| normal | 9054285 | 21.2304 | 44.6586 | -7.7532 | 1.0016 | 66.0% | 34.0% | 0.211 | 19.2% | 59.7% |
| rare_extreme | 983379 | 31.9264 | 67.5640 | 1.6771 | 1.0442 | 64.8% | 35.2% | 0.238 | 19.2% | 57.0% |
| normal_daytime | 4195053 | 44.3622 | 65.5903 | -18.1921 | -19.1373 | 26.9% | 73.1% | 0.447 | 41.4% | 13.9% |
| rare_extreme_daytime | 552492 | 50.3829 | 78.9160 | -3.4566 | -12.3255 | 37.5% | 62.5% | 0.417 | 34.1% | 24.2% |
| normal_nighttime | 4859232 | 1.2604 | 1.4560 | 1.2588 | 1.1634 | 99.8% | 0.2% | 0.007 | 0.0% | 99.3% |
| rare_extreme_nighttime | 430887 | 8.2612 | 49.3229 | 8.2596 | 1.1880 | 99.8% | 0.2% | 0.008 | 0.0% | 99.2% |
| label:unusually_low_solar_potential | 118147 | 91.7179 | 144.6682 | 91.3464 | 43.6978 | 98.3% | 1.7% | 0.161 | 0.1% | 83.8% |
| label:unusually_high_solar_potential | 38401 | 72.5989 | 90.8841 | -72.4505 | -61.9139 | 0.5% | 99.5% | 0.180 | 82.0% | 0.1% |
| label:extreme_temperature_condition | 436524 | 19.9253 | 41.2427 | -8.4354 | 0.9068 | 58.3% | 41.7% | 0.301 | 19.7% | 50.2% |
| label:extreme_wind_condition | 458693 | 25.2554 | 53.8380 | -5.0714 | 1.0415 | 67.9% | 32.1% | 0.214 | 17.9% | 60.7% |

## Daytime production-bin diagnostics

Bins use physical `y_true` in watts and only samples with target-time `solar_irradiance_poa > 10 W/m²`. Intervals are `[lower, upper)`, with the final bin `y_true >= 100 W`.

| bin | count | MAE | RMSE | mean_residual | median_residual | underprediction% | overprediction% | PICP PI | MPIW PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime_0_20 | 396207 | 13.1528 | 23.9359 | 7.0191 | -1.4347 | 57.5% | 42.5% | 0.562 | 24.8226 | 20.8% | 23.0% |
| daytime_20_40 | 376626 | 24.0848 | 41.9912 | 12.7444 | 1.0926 | 48.2% | 51.8% | 0.619 | 44.8846 | 13.6% | 24.4% |
| daytime_40_60 | 352255 | 34.5306 | 64.2275 | 15.8081 | -2.6536 | 53.3% | 46.7% | 0.614 | 53.6899 | 16.1% | 22.5% |
| daytime_60_80 | 266151 | 39.9947 | 66.2845 | 13.7533 | -6.5540 | 56.1% | 43.9% | 0.572 | 60.4304 | 19.5% | 23.3% |
| daytime_80_100 | 206451 | 45.1573 | 69.6233 | 8.8754 | -12.6726 | 61.2% | 38.8% | 0.528 | 63.6847 | 24.7% | 22.5% |
| daytime_gt_100 | 3149855 | 53.1849 | 73.4084 | -30.7534 | -31.2499 | 80.6% | 19.4% | 0.372 | 59.5205 | 51.8% | 11.0% |

## Automatic interpretation of residual asymmetry

- **Global:** underprediction; mean residual -6.8293 W, over 65.9%, under 34.1%.
- **Daytime:** underprediction; mean residual -16.4773 W, over 28.2%, under 71.8%.
- **Nighttime:** overprediction; mean residual 1.8291 W, over 99.8%, under 0.2%.
- **Unusually low solar potential:** overprediction; mean residual 91.3464 W, over 98.3%, under 1.7%.
- **Unusually high solar potential:** underprediction; mean residual -72.4505 W, over 0.5%, under 99.5%.
- **Rare/extreme daytime:** underprediction; mean residual -3.4566 W, over 37.5%, under 62.5%.
- **Production >= 100 W:** underprediction; mean residual -30.7534 W, over 19.4%, under 80.6%.
- **Nighttime softplus signature:** misses are predominantly below the PI while most targets are zero and most empirical lower bounds remain positive. This is consistent with `softplus` plus `y_true=0`.
- The model tends to **overpredict unusually low solar potential**.
- The model tends to **underpredict unusually high solar potential**.
- The model **underpredicts the >=100 W production bin**. Targets exceed `upper_pi` in 51.8% of these samples.
- The model overpredicts the low daytime production bin.
- **Global PI miss direction:** below 59.5%, above 19.2%; intervals/centres are predominantly too high.
- **Likely cause of low PICP:** softplus/zero-target nighttime misses, high-production targets above the PI. Centre bias and interval width should be interpreted together.

