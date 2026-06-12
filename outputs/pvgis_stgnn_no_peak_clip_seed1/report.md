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
- Device: cuda  |  Generated (UTC): 2026-06-10T18:28:38+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 20.3977 | 46.2066 | 7.2344 | 2.5060 | 17.5127 | 0.759 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 19.3669 | 43.4295 | 7.0806 | 2.3562 | 17.4035 | 0.767 | — |
| group:rare_or_extreme | 983379 | 29.8886 | 66.5358 | 8.6509 | 8.9446 | 18.3884 | 0.684 | — |
| label:unusually_low_solar_potential | 118147 | 92.9074 | 144.4537 | 13.9568 | 13.9206 | 19.7934 | 0.181 | — |
| label:unusually_high_solar_potential | 38401 | 50.2332 | 73.4640 | 16.6540 | 16.0849 | 25.4211 | 0.427 | — |
| label:extreme_temperature_condition | 436524 | 18.2512 | 40.6214 | 8.0522 | 7.9715 | 17.6469 | 0.775 | — |
| label:extreme_wind_condition | 458693 | 23.6823 | 53.2671 | 7.4354 | 2.7103 | 17.7117 | 0.743 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 19.3669  |  MAE rare/extreme: 29.8886  |  ratio: **1.54×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 19.3669  |  MAE rare/extreme: 29.8886  |  rare/normal MAE ratio: **1.54×**
- Mean uncertainty (std) normal: 7.0806  |  rare/extreme: 8.6509  |  rare/normal uncertainty ratio: **1.22×**
- Gaussian coverage@95 (diagnostic) normal: 0.767  |  rare/extreme: 0.684
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
| global | 0.216 | 24.5799 | 0.0275 | 42.2515 |
| normal | 0.215 | 24.0509 | 0.0269 | 42.1048 |
| rare_extreme | 0.233 | 29.4508 | 0.0330 | 42.7827 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.759 | 28.3590 | 0.0317 | 0.2458 |
| normal | 0.767 | 27.7559 | 0.0311 | 0.2242 |
| rare_extreme | 0.684 | 33.9116 | 0.0380 | 0.5797 |

## Daytime-only interval reliability

Eval-only split based on PVGIS `solar_irradiance_poa` at the target timestamp: daytime > **10.0 W/m²**, nighttime <= **10.0 W/m²**. The irradiance is diagnostic metadata and is not added to the model inputs or targets.

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | PICP PI | MPIW PI | NMPIL PI | CLC PI | PICP Gaussian | MPIW Gaussian | NMPIL Gaussian | CLC Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime | 4747545 | 41.0743 | 65.4724 | 13.6761 | 13.4769 | 19.9187 | 0.451 | 46.7648 | 0.0524 | 7.7209 | 0.507 | 53.6102 | 0.0600 | 5.0875 |
| nighttime | 5290119 | 1.8418 | 14.2874 | 1.4535 | 1.2842 | 2.2882 | 0.006 | 4.6704 | 0.0052 | 66.0565 | 0.985 | 5.6976 | 0.0064 | 0.0109 |
| normal_daytime | 4195053 | 40.3367 | 63.7778 | 13.6379 | 13.4475 | 19.8797 | 0.457 | 46.6352 | 0.0522 | 7.2936 | 0.511 | 53.4605 | 0.0598 | 4.8682 |
| rare_extreme_daytime | 552492 | 46.6742 | 77.1346 | 13.9661 | 13.6918 | 20.2279 | 0.410 | 47.7487 | 0.0535 | 11.9006 | 0.475 | 54.7470 | 0.0613 | 7.1109 |
| high_daytime | 1186852 | 43.0665 | 68.6695 | 12.4721 | 11.4314 | 18.2219 | 0.409 | 42.7328 | 0.0478 | 10.7848 | 0.466 | 48.8907 | 0.0547 | 6.9564 |
| peak_daytime | 474719 | 43.4248 | 67.7600 | 14.7282 | 13.7435 | 21.3010 | 0.431 | 50.3854 | 0.0564 | 10.1390 | 0.503 | 57.7345 | 0.0646 | 5.6873 |
| extreme_peak_daytime | 237361 | 45.9563 | 70.0004 | 17.1572 | 16.6069 | 23.6613 | 0.469 | 58.6735 | 0.0657 | 8.1327 | 0.547 | 67.2564 | 0.0753 | 4.3268 |

### Daytime production-tail diagnostics

| stratum | count | MAE | RMSE | mean residual | median residual | fraction underprediction | fraction above PI | PICP PI | PICP Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high_daytime | 1186852 | 43.0665 | 68.6695 | -31.7643 | -20.5247 | 0.786 | 0.503 | 0.409 | 0.466 |
| peak_daytime | 474719 | 43.4248 | 67.7600 | -38.9055 | -26.6007 | 0.880 | 0.542 | 0.431 | 0.503 |
| extreme_peak_daytime | 237361 | 45.9563 | 70.0004 | -43.1018 | -29.9813 | 0.909 | 0.521 | 0.469 | 0.547 |

| stratum | fraction y_true=0 | fraction lower PI <= 0 | fraction lower Gaussian <= 0 | PI coverage y=0 | Gaussian coverage y=0 | PI coverage y>0 | Gaussian coverage y>0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| daytime | 0.000 | 0.000 | 0.033 | — | — | 0.451 | 0.507 |
| nighttime | 0.988 | 0.000 | 0.984 | 0.000 | 0.988 | 0.449 | 0.792 |
| normal_daytime | 0.000 | 0.000 | 0.034 | — | — | 0.457 | 0.511 |
| rare_extreme_daytime | 0.000 | 0.000 | 0.028 | — | — | 0.410 | 0.475 |
| high_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.409 | 0.466 |
| peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.431 | 0.503 |
| extreme_peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.469 | 0.547 |

**PICP PI daytime is materially higher than global** (0.451 vs 0.216, delta 0.235).
It nevertheless remains low relative to the 0.95 coverage target.

## Residual bias diagnostics by stratum

`residual = y_pred_mean - y_true`: positive means overprediction, negative means underprediction.

| stratum | count | MAE | RMSE | mean_residual | median_residual | overprediction% | underprediction% | PICP PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 10037664 | 20.3977 | 46.2066 | -2.6021 | 1.0036 | 70.5% | 29.5% | 0.216 | 16.1% | 62.3% |
| daytime | 4747545 | 41.0743 | 65.4724 | -7.5527 | -8.6227 | 37.7% | 62.3% | 0.451 | 34.0% | 20.9% |
| nighttime | 5290119 | 1.8418 | 14.2874 | 1.8408 | 1.1181 | 99.9% | 0.1% | 0.006 | 0.0% | 99.4% |
| normal | 9054285 | 19.3669 | 43.4295 | -3.5825 | 0.9966 | 70.6% | 29.4% | 0.215 | 16.1% | 62.5% |
| rare_extreme | 983379 | 29.8886 | 66.5358 | 6.4243 | 1.0900 | 69.5% | 30.5% | 0.233 | 16.2% | 60.5% |
| normal_daytime | 4195053 | 40.3367 | 63.7778 | -9.1942 | -9.2474 | 36.6% | 63.4% | 0.457 | 34.7% | 19.6% |
| rare_extreme_daytime | 552492 | 46.6742 | 77.1346 | 4.9112 | -3.2867 | 45.8% | 54.2% | 0.410 | 28.8% | 30.2% |
| normal_nighttime | 4859232 | 1.2633 | 1.6749 | 1.2623 | 1.1144 | 99.9% | 0.1% | 0.005 | 0.0% | 99.4% |
| rare_extreme_nighttime | 430887 | 8.3657 | 49.7446 | 8.3645 | 1.1623 | 99.9% | 0.1% | 0.007 | 0.0% | 99.3% |
| label:unusually_low_solar_potential | 118147 | 92.9074 | 144.4537 | 92.4857 | 47.7756 | 98.5% | 1.5% | 0.076 | 0.2% | 92.1% |
| label:unusually_high_solar_potential | 38401 | 50.2332 | 73.4640 | -49.1883 | -36.4596 | 4.6% | 95.4% | 0.353 | 64.2% | 0.5% |
| label:extreme_temperature_condition | 436524 | 18.2512 | 40.6214 | -4.5744 | 0.9501 | 64.0% | 36.0% | 0.297 | 17.2% | 53.1% |
| label:extreme_wind_condition | 458693 | 23.6823 | 53.2671 | -0.2195 | 1.0661 | 72.8% | 27.2% | 0.210 | 15.0% | 64.0% |

## Daytime production-bin diagnostics

Bins use physical `y_true` in watts and only samples with target-time `solar_irradiance_poa > 10 W/m²`. Intervals are `[lower, upper)`, with the final bin `y_true >= 100 W`.

| bin | count | MAE | RMSE | mean_residual | median_residual | underprediction% | overprediction% | PICP PI | MPIW PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime_0_20 | 396207 | 16.1307 | 27.4865 | 11.9507 | 4.8769 | 38.2% | 61.8% | 0.421 | 23.3984 | 18.8% | 39.1% |
| daytime_20_40 | 376626 | 26.0112 | 45.4604 | 20.2159 | 9.0538 | 30.6% | 69.4% | 0.532 | 39.4716 | 8.5% | 38.3% |
| daytime_40_60 | 352255 | 34.1399 | 65.1306 | 23.0021 | 5.0141 | 42.9% | 57.1% | 0.566 | 46.1450 | 10.3% | 33.1% |
| daytime_60_80 | 266151 | 38.0141 | 66.8924 | 20.3200 | 2.6909 | 46.9% | 53.1% | 0.537 | 51.4130 | 16.3% | 30.0% |
| daytime_80_100 | 206451 | 41.2236 | 69.0709 | 14.2761 | -3.3543 | 53.5% | 46.5% | 0.516 | 53.8340 | 22.8% | 25.6% |
| daytime_gt_100 | 3149855 | 47.0372 | 70.3820 | -20.5291 | -19.2777 | 73.2% | 26.8% | 0.421 | 49.7892 | 43.9% | 14.0% |

## Automatic interpretation of residual asymmetry

- **Global:** underprediction; mean residual -2.6021 W, over 70.5%, under 29.5%.
- **Daytime:** underprediction; mean residual -7.5527 W, over 37.7%, under 62.3%.
- **Nighttime:** overprediction; mean residual 1.8408 W, over 99.9%, under 0.1%.
- **Unusually low solar potential:** overprediction; mean residual 92.4857 W, over 98.5%, under 1.5%.
- **Unusually high solar potential:** underprediction; mean residual -49.1883 W, over 4.6%, under 95.4%.
- **Rare/extreme daytime:** overprediction; mean residual 4.9112 W, over 45.8%, under 54.2%.
- **Production >= 100 W:** underprediction; mean residual -20.5291 W, over 26.8%, under 73.2%.
- **Nighttime softplus signature:** misses are predominantly below the PI while most targets are zero and most empirical lower bounds remain positive. This is consistent with `softplus` plus `y_true=0`.
- The model tends to **overpredict unusually low solar potential**.
- The model tends to **underpredict unusually high solar potential**.
- The model **underpredicts the >=100 W production bin**. Targets exceed `upper_pi` in 43.9% of these samples.
- The model overpredicts the low daytime production bin.
- **Global PI miss direction:** below 62.3%, above 16.1%; intervals/centres are predominantly too high.
- **Likely cause of low PICP:** softplus/zero-target nighttime misses, high-production targets above the PI. Centre bias and interval width should be interpreted together.

