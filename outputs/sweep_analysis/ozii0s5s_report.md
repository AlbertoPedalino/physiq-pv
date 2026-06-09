# Sweep `ozii0s5s` — PVGIS-only ST-GNN: CLC / interval-metrics report

Seed-only sweep (5 seeds) over the PVGIS-only ST-GNN with MC-Dropout + **group** post-hoc calibration. Interval reliability/sharpness metrics (PICP / MPIW / NMPIL / CLC) are **eval-only** — they do not touch the model, training, features or data. Source CSV: `outputs/sweep_analysis/ozii0s5s.csv`.

## 1. Setup

| param | value |
|---|---|
| sweep_id | ozii0s5s |
| train_years | 2016,2017 |
| calibration_year | 2018 |
| test_year | 2019 |
| feature_set | full |
| target | pv_power_output |
| model_type | stgnn |
| batch_size | 16 |
| epochs | 5 |
| lr | 0.001 |
| dropout | 0.2 |
| mc_samples | 20 |
| calibration_strategy | group |
| coverage_target (gamma) | 0.95 |
| clc_eta (eta) | 10 |
| seeds | 1,2,3,4,5 |
| skip_predictions_csv | true |
| max_calibration_samples | 200000 |

## 2. Per-seed results

| seed | run_name | state | mae/global | mae/normal | mae/rare_extreme | ratio_mae_r/n | std_normal | std_rare | std_ratio_r/n | picp_raw/g | picp_cal/g | nmpil_raw/g | nmpil_cal/g | clc_raw/g | clc_cal/g | clc_cal/norm | clc_cal/rare | k_global | k_normal | k_rare_or_extreme |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | blooming-sweep-1 | finished | 21.706 | 20.581 | 32.066 | 1.558 | 0.771 | 0.915 | 1.187 | 0.021 | 0.949 | 0.0034 | 0.1448 | 37.353 | 0.291 | 0.279 | 0.405 | 82.148 | 81.248 | 91.727 |
| 2 | faithful-sweep-2 | finished | 21.815 | 20.675 | 32.314 | 1.563 | 0.842 | 0.993 | 1.180 | 0.027 | 0.948 | 0.0038 | 0.1412 | 38.495 | 0.285 | 0.277 | 0.368 | 73.279 | 72.764 | 80.084 |
| 3 | scarlet-sweep-3 | finished | 19.940 | 18.796 | 30.481 | 1.622 | 0.877 | 1.042 | 1.188 | 0.033 | 0.950 | 0.0039 | 0.1178 | 37.541 | 0.235 | 0.222 | 0.365 | 57.877 | 56.647 | 76.490 |
| 4 | pleasant-sweep-4 | finished | 21.102 | 19.832 | 32.792 | 1.653 | 0.859 | 1.004 | 1.168 | 0.029 | 0.947 | 0.0038 | 0.1573 | 38.174 | 0.319 | 0.307 | 0.430 | 79.828 | 78.772 | 93.356 |
| 5 | polar-sweep-5 | finished | 19.959 | 18.905 | 29.663 | 1.569 | 0.874 | 1.038 | 1.188 | 0.031 | 0.949 | 0.0039 | 0.1613 | 38.145 | 0.324 | 0.316 | 0.399 | 80.843 | 80.694 | 82.781 |

## 3. Aggregated (mean / std / cv), n=5

### Point forecast

| metric | mean | std | cv |
|---|---|---|---|
| mae/global | 20.9045 | 0.9130 | 0.044 |
| mae/normal | 19.7577 | 0.8912 | 0.045 |
| mae/rare_extreme | 31.4633 | 1.3285 | 0.042 |
| ratio/mae_rare_normal | 1.5931 | 0.0424 | 0.027 |
| rmse/global | 47.3504 | 0.8081 | 0.017 |
| rmse/normal | 44.3472 | 0.8962 | 0.020 |
| rmse/rare_extreme | 69.0963 | 1.8516 | 0.027 |
| ratio/rmse_rare_normal | 1.5586 | 0.0516 | 0.033 |

### Raw interval metrics

| metric | mean | std | cv |
|---|---|---|---|
| picp_raw/global | 0.0283 | 0.0048 | 0.171 |
| picp_raw/normal | 0.0283 | 0.0049 | 0.172 |
| picp_raw/rare_extreme | 0.0285 | 0.0048 | 0.169 |
| mpiw_raw/global | 3.3697 | 0.1738 | 0.052 |
| mpiw_raw/normal | 3.3106 | 0.1710 | 0.052 |
| mpiw_raw/rare_extreme | 3.9136 | 0.2009 | 0.051 |
| nmpil_raw/global | 0.0038 | 0.0002 | 0.052 |
| nmpil_raw/normal | 0.0037 | 0.0002 | 0.052 |
| nmpil_raw/rare_extreme | 0.0044 | 0.0002 | 0.051 |
| clc_raw/global | 37.9417 | 0.4764 | 0.013 |
| clc_raw/normal | 37.2833 | 0.4679 | 0.013 |
| clc_raw/rare_extreme | 43.9921 | 0.6253 | 0.014 |

### Calibrated interval metrics

| metric | mean | std | cv |
|---|---|---|---|
| picp_calibrated/global | 0.9487 | 0.0011 | 0.001 |
| picp_calibrated/normal | 0.9494 | 0.0014 | 0.001 |
| picp_calibrated/rare_extreme | 0.9426 | 0.0041 | 0.004 |
| mpiw_calibrated/global | 129.0545 | 15.2635 | 0.118 |
| mpiw_calibrated/normal | 124.7034 | 16.0371 | 0.129 |
| mpiw_calibrated/rare_extreme | 169.1161 | 11.5930 | 0.069 |
| nmpil_calibrated/global | 0.1445 | 0.0171 | 0.118 |
| nmpil_calibrated/normal | 0.1396 | 0.0180 | 0.129 |
| nmpil_calibrated/rare_extreme | 0.1893 | 0.0130 | 0.069 |
| clc_calibrated/global | 0.2909 | 0.0354 | 0.122 |
| clc_calibrated/normal | 0.2802 | 0.0370 | 0.132 |
| clc_calibrated/rare_extreme | 0.3933 | 0.0270 | 0.069 |

### Calibration factors (k)

| metric | mean | std | cv |
|---|---|---|---|
| calibration/factor_global | 74.7951 | 10.0559 | 0.134 |
| calibration/factor_normal | 74.0252 | 10.2809 | 0.139 |
| calibration/factor_rare_extreme | 84.8877 | 7.3574 | 0.087 |

## 4. Sharpness / reliability interpretation

- **`picp_raw/global ~ 0.028`** — raw MC-Dropout intervals are heavily **under-covered** (target gamma=0.95). Raw std is an uncalibrated diagnostic, not a predictive std.
- **`nmpil_raw/global ~ 0.0038`** — raw intervals are extremely **sharp** (tiny normalized width), i.e. far too narrow.
- **`clc_raw/global ~ 37.9`** — CLC penalises hard because PICP is far from gamma: the sigma penalty `1+exp(-eta*(PICP-gamma))` dominates (see section 6), not the width.
- **`picp_calibrated/global ~ 0.949`** — group calibration restores coverage to ~gamma.
- **`nmpil_calibrated/global ~ 0.1445`** — calibrated intervals are much **wider** (~38.3x the raw width) — the price of reliability.
- **`clc_calibrated/global ~ 0.291`** — a far better sharpness/reliability trade-off than raw (~130x lower CLC).
- **`clc_calibrated/rare_extreme (0.393) > clc_calibrated/normal (0.280)`** — rare/extreme cases need wider intervals (or are less efficient) at equal coverage.

## 5. Normal vs rare/extreme — cost of covering rare events

| quantity | value |
|---|---|
| mae_rare / mae_normal | 1.592 |
| mpiw_cal_rare / mpiw_cal_normal | 1.356 |
| nmpil_cal_rare / nmpil_cal_normal | 1.356 |
| clc_cal_rare / clc_cal_normal | 1.404 |
| picp_cal_rare - picp_cal_normal | -0.0068 |

**Answer:** at essentially equal coverage (delta_picp = -0.0068), covering rare/extreme costs ~35.6% wider calibrated intervals and a ~40.4% worse CLC than normal. The model errs ~59.2% more on rare/extreme (MAE), and pays for reliability there with wider bands.

## 6. CLC components (sigma = CLC / NMPIL)

sigma is the coverage penalty `1+exp(-eta*(PICP-gamma))`. Shows raw CLC is large because of sigma, not width.

| group | interval_type | PICP | NMPIL | sigma = CLC/NMPIL | CLC |
|---|---|---|---|---|---|
| global | raw | 0.028 | 0.0038 | 10058.00 | 37.942 |
| global | calibrated | 0.949 | 0.1445 | 2.01 | 0.291 |
| normal | raw | 0.028 | 0.0037 | 10059.83 | 37.283 |
| normal | calibrated | 0.949 | 0.1396 | 2.01 | 0.280 |
| rare_extreme | raw | 0.028 | 0.0044 | 10041.15 | 43.992 |
| rare_extreme | calibrated | 0.943 | 0.1893 | 2.08 | 0.393 |

sigma_raw ~ 1+exp(-10*(0.028-0.95)) ~ 1e4 -> raw CLC blows up despite tiny NMPIL; sigma_calibrated ~ 2 (PICP~gamma) -> CLC ~ 2*NMPIL.

## 7. Sanity check vs previous sweep `ltxbtupy` (no CLC metrics)

| metric | ozii0s5s (mean) | ltxbtupy (mean) |
|---|---|---|
| mae/global | 20.904 | 20.900 |
| mae/rare_extreme | 31.463 | 31.604 |
| ratio/mae_rare_normal | 1.593 | 1.601 |

Point-forecast metrics match `ltxbtupy` (MAE/global ~ 20.9, MAE rare/extreme ~ 31.5-31.6, rare/normal ratio ~ 1.59-1.60). **Confirms the new CLC/interval metrics are eval-only and did not change the model or training.**

## 8. Best seeds

- Best by **mae/rare_extreme**: seed **5** (29.663)
- Best by **clc_calibrated/global**: seed **3** (0.2354)
- Best by **clc_calibrated/rare_extreme**: seed **3** (0.3651)

The best seed for point accuracy (seed 5) differs from the best seed for interval efficiency (seed 3). Point MAE and interval CLC optimise different things (error vs width-at-coverage), so the best seed is not unique — report MAE and CLC separately.

## 9. Operational conclusion

- PVGIS-only ST-GNN is **seed-robust** on the point forecast (MAE/global cv 0.044).
- Rare/extreme conditions are **harder**: MAE rare/normal ~ 1.59.
- **Raw MC-Dropout does not give reliable predictive intervals** (PICP raw ~ 0.028).
- **Group calibration fixes reliability** (PICP calibrated ~ 0.949).
- The cost of calibration is **wider intervals** (NMPIL up ~38x).
- **CLC confirms calibrated >> raw** (CLC 0.291 vs 37.9).
- **CLC rare/extreme > normal** -> rare/extreme stays more expensive to cover even after calibration.
- This **closes the raw-vs-calibrated uncertainty fix**.
- **Next step:** event-centered analysis around rare/extreme events, or simple baselines (persistence) for context.

