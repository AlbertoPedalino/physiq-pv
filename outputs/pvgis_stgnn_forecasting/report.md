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
- Training loss: **weighted_mse**
  - weighted_mse params (train-only signals, no anomaly labels): daytime_threshold=10.0, night_weight=0.2, day_weight=1.0, high_threshold_w=80.0, high_weight=1.5, peak_threshold_w=100.0, peak_weight=2.0
- Selected features (11): temperature_2m, solar_irradiance_poa, wind_speed_10m, sin_elev, cos_elev, kt, kt_std_3h, dghi_dt, dni_norm, dhi_norm, pv_lag_pvgis
- MC samples: **20**
- seq_len: **24**  |  horizon: **1**
- Train years: 2016,2017,2018
- Test year: **2019**
- Nodes (locations): **1149**  |  epochs: **5**
- batch_size: 16  |  lr: 0.001
- Anomaly scores: outputs/pvgis_anomaly_2019_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Predictions: **10037664**
- Device: cuda  |  Generated (UTC): 2026-06-11T08:38:52+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 10037664 | 22.3892 | 46.6542 | 7.1372 | 3.5692 | 16.3361 | 0.533 | — |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 9054285 | 21.1556 | 43.3883 | 7.0013 | 3.3974 | 16.2214 | 0.539 | — |
| group:rare_or_extreme | 983379 | 33.7476 | 69.8873 | 8.3887 | 7.9281 | 17.2144 | 0.475 | — |
| label:unusually_low_solar_potential | 118147 | 106.2933 | 155.0985 | 13.7158 | 13.8398 | 19.1901 | 0.099 | — |
| label:unusually_high_solar_potential | 38401 | 65.4197 | 83.4939 | 13.8983 | 13.6840 | 20.1134 | 0.160 | — |
| label:extreme_temperature_condition | 436524 | 20.3630 | 41.3548 | 7.8096 | 6.9034 | 16.5390 | 0.555 | — |
| label:extreme_wind_condition | 458693 | 25.9614 | 53.9175 | 7.3580 | 3.7917 | 16.5956 | 0.519 | — |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

The `normal` vs `rare_extreme` stratification is a PVGIS-only adaptation of the paper's `non-intense` vs `intense` split, not an exact replica. Anomaly labels are used only for evaluation/stratification and are never used as model inputs or targets.

- MAE normal: 21.1556  |  MAE rare/extreme: 33.7476  |  ratio: **1.60×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty by anomaly stratum

- MC samples: **20**
- MAE normal: 21.1556  |  MAE rare/extreme: 33.7476  |  rare/normal MAE ratio: **1.60×**
- Mean uncertainty (std) normal: 7.0013  |  rare/extreme: 8.3887  |  rare/normal uncertainty ratio: **1.20×**
- Gaussian coverage@95 (diagnostic) normal: 0.539  |  rare/extreme: 0.475
- Primary paper-style PI coverage (PICP) is reported in *Interval reliability & sharpness*.

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.60×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.20×).

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
| global | 0.193 | 24.3164 | 0.0272 | 53.0152 |
| normal | 0.193 | 23.8506 | 0.0267 | 51.9718 |
| rare_extreme | 0.192 | 28.6058 | 0.0320 | 62.6740 |

### Gaussian (diagnostic)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.533 | 27.9778 | 0.0313 | 2.0646 |
| normal | 0.539 | 27.4450 | 0.0307 | 1.9034 |
| rare_extreme | 0.475 | 32.8836 | 0.0368 | 4.3085 |

## Daytime-only interval reliability

Eval-only split based on PVGIS `solar_irradiance_poa` at the target timestamp: daytime > **10.0 W/m²**, nighttime <= **10.0 W/m²**. The irradiance is diagnostic metadata and is not added to the model inputs or targets.

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | PICP PI | MPIW PI | NMPIL PI | CLC PI | PICP Gaussian | MPIW Gaussian | NMPIL Gaussian | CLC Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime | 4747545 | 42.2416 | 65.8067 | 12.5498 | 12.2948 | 18.6200 | 0.407 | 42.9102 | 0.0480 | 11.0455 | 0.458 | 49.1952 | 0.0551 | 7.6059 |
| nighttime | 5290119 | 4.5730 | 15.6087 | 2.2797 | 2.0861 | 3.3083 | 0.001 | 7.6297 | 0.0085 | 113.5082 | 0.600 | 8.9365 | 0.0100 | 0.3420 |
| normal_daytime | 4195053 | 41.0754 | 63.5900 | 12.5001 | 12.2372 | 18.5851 | 0.415 | 42.7415 | 0.0478 | 10.1013 | 0.466 | 49.0002 | 0.0549 | 6.9698 |
| rare_extreme_daytime | 552492 | 51.0969 | 80.6745 | 12.9275 | 12.7105 | 18.8760 | 0.341 | 44.1907 | 0.0495 | 21.7780 | 0.394 | 50.6758 | 0.0567 | 14.7761 |
| high_daytime | 1186852 | 49.5012 | 69.8776 | 10.5362 | 9.7348 | 15.1358 | 0.209 | 36.0977 | 0.0404 | 66.7954 | 0.242 | 41.3021 | 0.0462 | 55.1681 |
| peak_daytime | 474719 | 60.6890 | 76.9154 | 12.3551 | 11.8057 | 16.9381 | 0.090 | 42.2983 | 0.0474 | 257.6750 | 0.113 | 48.4322 | 0.0542 | 234.3012 |
| extreme_peak_daytime | 237361 | 67.9313 | 82.6700 | 13.9999 | 13.6579 | 18.4944 | 0.066 | 47.9195 | 0.0536 | 372.2527 | 0.086 | 54.8796 | 0.0614 | 348.7186 |

### Daytime production-tail diagnostics

| stratum | count | MAE | RMSE | mean residual | median residual | fraction underprediction | fraction above PI | PICP PI | PICP Gaussian |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high_daytime | 1186852 | 49.5012 | 69.8776 | -42.2341 | -34.7074 | 0.859 | 0.723 | 0.209 | 0.242 |
| peak_daytime | 474719 | 60.6890 | 76.9154 | -59.5507 | -49.5992 | 0.971 | 0.902 | 0.090 | 0.113 |
| extreme_peak_daytime | 237361 | 67.9313 | 82.6700 | -67.5403 | -56.5775 | 0.988 | 0.933 | 0.066 | 0.086 |

| stratum | fraction y_true=0 | fraction lower PI <= 0 | fraction lower Gaussian <= 0 | PI coverage y=0 | Gaussian coverage y=0 | PI coverage y>0 | Gaussian coverage y>0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| daytime | 0.000 | 0.000 | 0.017 | — | — | 0.407 | 0.458 |
| nighttime | 0.988 | 0.000 | 0.597 | 0.000 | 0.600 | 0.043 | 0.589 |
| normal_daytime | 0.000 | 0.000 | 0.018 | — | — | 0.415 | 0.466 |
| rare_extreme_daytime | 0.000 | 0.000 | 0.013 | — | — | 0.341 | 0.394 |
| high_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.209 | 0.242 |
| peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.090 | 0.113 |
| extreme_peak_daytime | 0.000 | 0.000 | 0.000 | — | — | 0.066 | 0.086 |

**PICP PI daytime is materially higher than global** (0.407 vs 0.193, delta 0.214).
It nevertheless remains low relative to the 0.95 coverage target.

## Residual bias diagnostics by stratum

`residual = y_pred_mean - y_true`: positive means overprediction, negative means underprediction.

| stratum | count | MAE | RMSE | mean_residual | median_residual | overprediction% | underprediction% | PICP PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 10037664 | 22.3892 | 46.6542 | 1.8612 | 3.6758 | 73.7% | 26.3% | 0.193 | 15.5% | 65.2% |
| daytime | 4747545 | 42.2416 | 65.8067 | -1.1606 | -4.1355 | 44.4% | 55.6% | 0.407 | 32.8% | 26.5% |
| nighttime | 5290119 | 4.5730 | 15.6087 | 4.5730 | 3.8185 | 100.0% | 0.0% | 0.001 | 0.0% | 99.9% |
| normal | 9054285 | 21.1556 | 43.3883 | 0.9012 | 3.6673 | 73.9% | 26.1% | 0.193 | 15.3% | 65.4% |
| rare_extreme | 983379 | 33.7476 | 69.8873 | 10.6995 | 3.7778 | 72.3% | 27.7% | 0.192 | 17.2% | 63.6% |
| normal_daytime | 4195053 | 41.0754 | 63.5900 | -2.6401 | -4.6534 | 43.6% | 56.4% | 0.415 | 33.1% | 25.4% |
| rare_extreme_daytime | 552492 | 51.0969 | 80.6745 | 10.0737 | 0.5860 | 50.7% | 49.3% | 0.341 | 30.6% | 35.2% |
| normal_nighttime | 4859232 | 3.9586 | 4.0982 | 3.9586 | 3.8157 | 100.0% | 0.0% | 0.001 | 0.0% | 99.9% |
| rare_extreme_nighttime | 430887 | 11.5020 | 52.9312 | 11.5020 | 3.8522 | 100.0% | 0.0% | 0.001 | 0.0% | 99.9% |
| label:unusually_low_solar_potential | 118147 | 106.2933 | 155.0985 | 105.9633 | 60.9964 | 98.9% | 1.1% | 0.039 | 0.2% | 96.0% |
| label:unusually_high_solar_potential | 38401 | 65.4197 | 83.4939 | -64.7225 | -58.2775 | 4.2% | 95.8% | 0.143 | 85.4% | 0.3% |
| label:extreme_temperature_condition | 436524 | 20.3630 | 41.3548 | -1.4996 | 3.5442 | 67.2% | 32.8% | 0.252 | 18.7% | 56.1% |
| label:extreme_wind_condition | 458693 | 25.9614 | 53.9175 | 4.4753 | 3.7621 | 76.2% | 23.8% | 0.184 | 14.3% | 67.3% |

## Daytime production-bin diagnostics

Bins use physical `y_true` in watts and only samples with target-time `solar_irradiance_poa > 10 W/m²`. Intervals are `[lower, upper)`, with the final bin `y_true >= 100 W`.

| bin | count | MAE | RMSE | mean_residual | median_residual | underprediction% | overprediction% | PICP PI | MPIW PI | above_interval% | below_interval% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| daytime_0_20 | 396207 | 17.0084 | 31.8349 | 14.4249 | 3.2996 | 31.4% | 68.6% | 0.562 | 23.8541 | 6.0% | 37.8% |
| daytime_20_40 | 376626 | 33.1818 | 54.0091 | 26.8730 | 12.9812 | 33.1% | 66.9% | 0.440 | 39.5285 | 9.6% | 46.5% |
| daytime_40_60 | 352255 | 42.3177 | 73.1297 | 33.4448 | 16.4015 | 33.7% | 66.3% | 0.454 | 46.5732 | 8.7% | 45.9% |
| daytime_60_80 | 266151 | 44.9444 | 74.4680 | 32.2965 | 14.4333 | 35.9% | 64.1% | 0.468 | 51.0824 | 11.1% | 42.0% |
| daytime_80_100 | 206451 | 45.3115 | 74.9429 | 27.4694 | 8.0535 | 41.8% | 58.2% | 0.500 | 53.1549 | 14.1% | 35.9% |
| daytime_gt_100 | 3149855 | 46.0607 | 67.9415 | -15.0465 | -16.2651 | 66.3% | 33.7% | 0.367 | 43.9398 | 44.7% | 18.7% |

## Automatic interpretation of residual asymmetry

- **Global:** overprediction; mean residual 1.8612 W, over 73.7%, under 26.3%.
- **Daytime:** underprediction; mean residual -1.1606 W, over 44.4%, under 55.6%.
- **Nighttime:** overprediction; mean residual 4.5730 W, over 100.0%, under 0.0%.
- **Unusually low solar potential:** overprediction; mean residual 105.9633 W, over 98.9%, under 1.1%.
- **Unusually high solar potential:** underprediction; mean residual -64.7225 W, over 4.2%, under 95.8%.
- **Rare/extreme daytime:** overprediction; mean residual 10.0737 W, over 50.7%, under 49.3%.
- **Production >= 100 W:** underprediction; mean residual -15.0465 W, over 33.7%, under 66.3%.
- **Nighttime softplus signature:** misses are predominantly below the PI while most targets are zero and most empirical lower bounds remain positive. This is consistent with `softplus` plus `y_true=0`.
- The model tends to **overpredict unusually low solar potential**.
- The model tends to **underpredict unusually high solar potential**.
- The model **underpredicts the >=100 W production bin**. Targets exceed `upper_pi` in 44.7% of these samples.
- The model overpredicts the low daytime production bin.
- **Global PI miss direction:** below 65.2%, above 15.5%; intervals/centres are predominantly too high.
- **Likely cause of low PICP:** softplus/zero-target nighttime misses, high-production targets above the PI. Centre bias and interval width should be interpreted together.

