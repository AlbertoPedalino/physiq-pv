# PVGIS-only ST-GNN forecasting report

Reuses the existing STGNN architecture on a **PVGIS-only** input. No real plant production, no ENERGIA, no quality score, no kWp/UPN. Anomaly labels are used **only** for stratified evaluation.

## Experiment

- Mode: **pvgis_stgnn**
- Model type: **stgnn**
- Feature set: **full**
- W&B enabled: **False**
- MC Dropout: **enabled** (experimental)

## Parameters

- Target variable: **pv_power_output**
- Selected features (11): temperature_2m, solar_irradiance_poa, wind_speed_10m, sin_elev, cos_elev, kt, kt_std_3h, dghi_dt, dni_norm, dhi_norm, pv_lag_pvgis
- MC samples: **3**
- seq_len: **24**  |  horizon: **1**
- Train years: 2016,2017
- Test year: **2019**
- Nodes (locations): **1149**  |  epochs: **1**
- batch_size: 8  |  lr: 0.001
- Anomaly scores: outputs/pvgis_anomaly_2019_2005_2023_w15_q0975/pvgis_climatology_scores.csv
- Predictions: **1149000**
- Device: cuda  |  Generated (UTC): 2026-06-05T18:04:54+00:00

## Global metrics

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| all | 1149000 | 74.5783 | 113.6421 | 0.6456 | 0.1292 | 2.2186 | 0.007 | 0.951 |

## Metrics by anomaly stratum

| stratum | count | MAE | RMSE | mean_std | median_std | p90_std | coverage_95_raw | coverage_95_calibrated |
|---|---|---|---|---|---|---|---|---|
| group:normal | 1044838 | 72.8821 | 110.2991 | 0.6302 | 0.1274 | 2.1800 | 0.007 | 0.950 |
| group:rare_or_extreme | 104162 | 91.5927 | 142.9124 | 0.7998 | 0.1534 | 2.5465 | 0.006 | 0.963 |
| label:unusually_low_solar_potential | 13080 | 164.6430 | 242.8448 | 1.0227 | 0.2809 | 3.0578 | 0.000 | 0.985 |
| label:unusually_high_solar_potential | 4488 | 179.8775 | 220.5694 | 1.8395 | 1.5625 | 3.7015 | 0.005 | 0.971 |
| label:extreme_temperature_condition | 43307 | 83.4644 | 124.2948 | 0.8765 | 0.1828 | 2.5881 | 0.009 | 0.970 |
| label:extreme_wind_condition | 49880 | 74.0154 | 116.6199 | 0.5915 | 0.1189 | 2.0929 | 0.006 | 0.953 |

## Does ST-GNN degrade on rare/extreme PVGIS conditions?

- MAE normal: 72.8821  |  MAE rare/extreme: 91.5927  |  ratio: **1.26×**
- Verdict: **yes** — ST-GNN is worse on rare/extreme conditions.

## Uncertainty calibration

MC Dropout std is a relative uncertainty measure. Raw intervals `mean ± 1.96 std` are not guaranteed to be calibrated. Post-hoc calibration scales std with a factor estimated on a separate calibration set; the test year is used only for evaluation.

- Calibration years: 2018
- Coverage target: **0.950**
- Calibration factor: **2128.0758**
- Calibration predictions: **10037664**

| stratum | raw coverage | calibrated coverage |
|---|---|---|
| global | 0.007 | 0.951 |
| normal | 0.007 | 0.950 |
| rare/extreme | 0.006 | 0.963 |

## Uncertainty by anomaly stratum

- MC samples: **3**
- MAE normal: 72.8821  |  MAE rare/extreme: 91.5927  |  rare/normal MAE ratio: **1.26×**
- Mean uncertainty (std) normal: 0.6302  |  rare/extreme: 0.7998  |  rare/normal uncertainty ratio: **1.27×**
- Raw coverage@95 normal: 0.007  |  rare/extreme: 0.006
- Calibrated coverage@95 normal: 0.950  |  rare/extreme: 0.963

1. Does the model err more on rare/extreme? **yes** (MAE ratio 1.26×).
2. Is the model also more uncertain on rare/extreme? **yes** (uncertainty ratio 1.27×).

