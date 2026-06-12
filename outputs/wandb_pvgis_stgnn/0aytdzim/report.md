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
- Device: cuda  |  Generated (UTC): 2026-06-11T12:21:41+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 19.6734 | 45.7769 | 8.3011 | 1.9686 | 21.1382 | 0.799 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 18.6397 | 42.8128 | 8.0942 | 1.8163 | 20.9512 | 0.805 | — |
| group:rare_or_extreme | 983379 | 29.1917 | 67.1811 | 10.2059 | 10.3553 | 22.6441 | 0.742 | — |
| label:unusually_low_solar_potential | 118147 | 95.3339 | 149.2425 | 16.3832 | 16.3845 | 23.4487 | 0.277 | — |
| label:unusually_high_solar_potential | 38401 | 46.3367 | 71.9605 | 21.9926 | 22.4557 | 31.9580 | 0.625 | — |
| label:extreme_temperature_condition | 436524 | 16.7898 | 39.9623 | 9.5677 | 8.9605 | 22.1006 | 0.839 | — |
| label:extreme_wind_condition | 458693 | 23.4397 | 52.8420 | 8.5674 | 2.2242 | 21.3565 | 0.772 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 18.6397  |  MAE rare/extreme: 29.1917  |  ratio: **1.57×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 18.6397  |  MAE rare/extreme: 29.1917  |  rare/normal MAE ratio: **1.57×**
- Mean uncertainty (std) normal: 8.0942  |  rare/extreme: 10.2059  |  rare/normal uncertainty ratio: **1.26×**
- Gaussian coverage@95 (diagnostic) normal: 0.805  |  rare/extreme: 0.742
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.57×).
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
| global | 0.261 | 28.3006 | 0.0317 | 31.0352 |
| normal | 0.258 | 27.5923 | 0.0309 | 31.2354 |
| rare_extreme | 0.291 | 34.8226 | 0.0390 | 28.5023 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.799 | 32.5403 | 0.0364 | 0.2010 |
| normal | 0.805 | 31.7293 | 0.0355 | 0.1863 |
| rare_extreme | 0.742 | 40.0072 | 0.0448 | 0.4028 |

## Daytime-only interval reliability

Eval-only split based on PVGIS `solar_irradiance_poa` at the target timestamp: daytime > **10.0 W/m²**, nighttime <= **10.0 W/m²**. The irradiance is diagnostic metadata and is not added to the model inputs or targets.

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | PICP PI | MPIW PI | NMPIL PI | CLC PI | PICP Gaussian | MPIW Gaussian | NMPIL Gaussian | CLC Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime | 4747545 | 39.5819 | 64.5581 | 16.3111 | 16.1320 | 24.1862 | 0.546 | 55.7672 | 0.0624 | 3.6260 | 0.597 | 63.9397 | 0.0716 | 2.5251 |
| nighttime | 5290119 | 1.8068 | 15.3572 | 1.1126 | 0.9201 | 1.7314 | 0.006 | 3.6511 | 0.0041 | 51.2399 | 0.981 | 4.3613 | 0.0049 | 0.0085 |
| normal_daytime | 4195053 | 38.8646 | 62.8775 | 16.2255 | 16.0601 | 24.0658 | 0.550 | 55.4754 | 0.0621 | 3.4492 | 0.599 | 63.6038 | 0.0712 | 2.4568 |
| rare_extreme_daytime | 552492 | 45.0287 | 76.1175 | 16.9618 | 16.6898 | 25.0882 | 0.511 | 57.9832 | 0.0649 | 5.3010 | 0.579 | 66.4901 | 0.0744 | 3.1078 |
| high_daytime | 1186852 | 41.7207 | 68.8103 | 16.9672 | 15.6211 | 26.0786 | 0.569 | 58.1330 | 0.0651 | 3.0009 | 0.612 | 66.5114 | 0.0745 | 2.2623 |
| peak_daytime | 474719 | 37.1742 | 65.6321 | 21.8921 | 21.5717 | 29.5204 | 0.720 | 74.9432 | 0.0839 | 0.9187 | 0.761 | 85.8171 | 0.0961 | 0.7332 |
| extreme_peak_daytime | 237361 | 38.7865 | 66.9160 | 24.5226 | 24.5116 | 31.5827 | 0.733 | 83.9311 | 0.0940 | 0.9140 | 0.777 | 96.1286 | 0.1076 | 0.7143 |

### Daytime production-tail diagnostics

| stratum | count | MAE | RMSE | mean residual | median residual | fraction underprediction | fraction above PI | PICP PI | PICP Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high_daytime | 1186852 | 41.7207 | 68.8103 | -25.3277 | -11.8463 | 0.673 | 0.342 | 0.569 | 0.612 |
| peak_daytime | 474719 | 37.1742 | 65.6321 | -25.5313 | -11.2613 | 0.673 | 0.243 | 0.720 | 0.761 |
| extreme_peak_daytime | 237361 | 38.7865 | 66.9160 | -33.0517 | -19.2658 | 0.800 | 0.257 | 0.733 | 0.777 |

| stratum | fraction y_true=0 | fraction lower PI <= 0 | fraction lower Gaussian <= 0 | PI coverage y=0 | Gaussian coverage y=0 | PI coverage y>0 | Gaussian coverage y>0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| daytime | 0.000 | 0.000 | 0.057 | — | — | 0.546 | 0.597 |
| nighttime | 0.988 | 0.000 | 0.980 | 0.000 | 0.982 | 0.511 | 0.893 |
| normal_daytime | 0.000 | 0.000 | 0.058 | — | — | 0.550 | 0.599 |
| rare_extreme_daytime | 0.000 | 0.000 | 0.046 | — | — | 0.511 | 0.579 |
| high_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.569 | 0.612 |
| peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.720 | 0.761 |
| extreme_peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.733 | 0.777 |

**PICP PI daytime is materially higher than global** (0.546 vs 0.261, delta 0.284).
It nevertheless remains low relative to the 0.95 coverage target.

## Residual bias diagnostics by stratum

`residual = y_pred_mean - y_true`: positive means overprediction, negative means underprediction.

| stratum | count | MAE | RMSE | mean_residual | median_residual | overprediction% | underprediction% | PICP PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 10037664 | 19.6734 | 45.7769 | 0.3176 | 1.0377 | 73.4% | 26.6% | 0.261 | 11.7% | 62.1% |
| daytime | 4747545 | 39.5819 | 64.5581 | -1.3400 | -3.6038 | 44.0% | 56.0% | 0.546 | 24.8% | 20.7% |
| nighttime | 5290119 | 1.8068 | 15.3572 | 1.8051 | 1.0810 | 99.8% | 0.2% | 0.006 | 0.0% | 99.4% |
| normal | 9054285 | 18.6397 | 42.8128 | -0.7377 | 1.0309 | 73.4% | 26.6% | 0.258 | 11.8% | 62.3% |
| rare_extreme | 983379 | 29.1917 | 67.1811 | 10.0336 | 1.1205 | 73.6% | 26.4% | 0.291 | 10.8% | 60.1% |
| normal_daytime | 4195053 | 38.8646 | 62.8775 | -2.9560 | -4.2636 | 42.8% | 57.2% | 0.550 | 25.5% | 19.5% |
| rare_extreme_daytime | 552492 | 45.0287 | 76.1175 | 10.9304 | 2.4807 | 53.2% | 46.8% | 0.511 | 19.2% | 29.7% |
| normal_nighttime | 4859232 | 1.1792 | 1.4652 | 1.1774 | 1.0790 | 99.8% | 0.2% | 0.006 | 0.0% | 99.4% |
| rare_extreme_nighttime | 430887 | 8.8852 | 53.5847 | 8.8838 | 1.1048 | 99.8% | 0.2% | 0.008 | 0.0% | 99.2% |
| label:unusually_low_solar_potential | 118147 | 95.3339 | 149.2425 | 95.0794 | 47.4882 | 98.7% | 1.3% | 0.111 | 0.1% | 88.8% |
| label:unusually_high_solar_potential | 38401 | 46.3367 | 71.9605 | -41.8947 | -31.6434 | 16.7% | 83.3% | 0.560 | 42.5% | 1.5% |
| label:extreme_temperature_condition | 436524 | 16.7898 | 39.9623 | 0.6616 | 1.0543 | 70.6% | 29.4% | 0.370 | 9.4% | 53.5% |
| label:extreme_wind_condition | 458693 | 23.4397 | 52.8420 | 2.0710 | 1.0715 | 75.1% | 24.9% | 0.244 | 12.0% | 63.7% |

## Daytime production-bin diagnostics

Bins use physical `y_true` in watts and only samples with target-time `solar_irradiance_poa > 10 W/m²`. Intervals are `[lower, upper)`, with the final bin `y_true >= 100 W`.

| bin | count | MAE | RMSE | mean_residual | median_residual | underprediction% | overprediction% | PICP PI | MPIW PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime_0_20 | 396207 | 15.0320 | 26.8277 | 9.5024 | -0.3468 | 51.3% | 48.7% | 0.491 | 24.9705 | 22.5% | 28.4% |
| daytime_20_40 | 376626 | 26.6949 | 46.2097 | 19.3791 | 8.0520 | 35.6% | 64.4% | 0.585 | 46.4612 | 9.2% | 32.3% |
| daytime_40_60 | 352255 | 35.2677 | 65.9878 | 23.6895 | 4.7466 | 43.2% | 56.8% | 0.628 | 55.3086 | 8.2% | 29.0% |
| daytime_60_80 | 266151 | 41.9538 | 70.1289 | 23.0994 | 2.2801 | 47.8% | 52.2% | 0.575 | 61.4760 | 12.8% | 29.7% |
| daytime_80_100 | 206451 | 45.9390 | 72.8710 | 18.5442 | -2.4204 | 52.1% | 47.9% | 0.534 | 64.4279 | 19.3% | 27.3% |
| daytime_gt_100 | 3149855 | 44.0762 | 68.4494 | -11.3486 | -8.9607 | 61.4% | 38.6% | 0.537 | 59.7550 | 30.2% | 16.1% |

## Automatic interpretation of residual asymmetry

- **Global:** overprediction; mean residual 0.3176 W, over 73.4%, under 26.6%.
- **Daytime:** underprediction; mean residual -1.3400 W, over 44.0%, under 56.0%.
- **Nighttime:** overprediction; mean residual 1.8051 W, over 99.8%, under 0.2%.
- **Unusually low solar potential:** overprediction; mean residual 95.0794 W, over 98.7%, under 1.3%.
- **Unusually high solar potential:** underprediction; mean residual -41.8947 W, over 16.7%, under 83.3%.
- **Rare/extreme daytime:** overprediction; mean residual 10.9304 W, over 53.2%, under 46.8%.
- **Production >= 100 W:** underprediction; mean residual -11.3486 W, over 38.6%, under 61.4%.
- **Nighttime softplus signature:** misses are predominantly below the PI while most targets are zero and most empirical lower bounds remain positive. This is consistent with `softplus` plus `y_true=0`.
- The model tends to **overpredict unusually low solar potential**.
- The model tends to **underpredict unusually high solar potential**.
- The model **underpredicts the >=100 W production bin**. Targets exceed `upper_pi` in 30.2% of these samples.
- The model overpredicts the low daytime production bin.
- **Global PI miss direction:** below 62.1%, above 11.7%; intervals/centres are predominantly too high.
- **Likely cause of low PICP:** softplus/zero-target nighttime misses, high-production targets above the PI. Centre bias and interval width should be interpreted together.

