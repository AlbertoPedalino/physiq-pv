# Sweep `7c8llckm` — PVGIS-only ST-GNN: paper-style uncertainty report

Paper-style protocol (uncertainty-aware rainfall prediction): MC-Dropout in test, predictive intervals built **directly from the MC sample quantiles** (q0.025/q0.975), **no post-hoc calibration**. PICP is evaluated, not forced. Source CSV: `outputs/sweep_analysis/7c8llckm.csv`.

## 1. Setup

| param | value |
|---|---|
| sweep_id | 7c8llckm |
| train_years | 2016,2017,2018 |
| calibration | none |
| test_year | 2019 |
| MC Dropout | true (mc_samples=20) |
| primary PI | MC sample quantiles q0.025 / q0.975 |
| Gaussian band | diagnostic only (mean ± 1.96·std_raw) |
| post-hoc calibration | OFF (enable_posthoc_calibration=False) |
| feature_set | full |
| target | pv_power_output |
| model_type | stgnn |
| batch_size | 16 |
| epochs | 5 |
| lr | 0.001 |
| dropout | 0.2 |
| coverage_target (gamma) | 0.95 |
| clc_eta (eta) | 10 |
| seeds | 1,2,3,4,5 |

## 2. Per-seed results

| seed | run_name | mae/global | mae/normal | mae/rare | ratio_mae | rmse/global | rmse/rare | std_norm | std_rare | std_ratio | picp_pi/g | picp_pi/n | picp_pi/r | nmpil_pi/g | clc_pi/g | clc_pi/n | clc_pi/r | picp_gauss/g | clc_gauss/g |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | balmy-sweep-1 | 19.937 | 18.846 | 29.980 | 1.591 | 45.924 | 66.491 | 0.913 | 1.094 | 1.199 | 0.030 | 0.030 | 0.030 | 0.0036 | 35.494 | 34.825 | 41.642 | 0.034 | 38.892 |
| 2 | feasible-sweep-2 | 20.927 | 19.788 | 31.414 | 1.588 | 47.205 | 68.195 | 0.889 | 1.014 | 1.140 | 0.026 | 0.026 | 0.028 | 0.0035 | 35.561 | 35.137 | 39.398 | 0.030 | 39.199 |
| 3 | brisk-sweep-3 | 19.828 | 18.598 | 31.155 | 1.675 | 45.726 | 68.647 | 0.886 | 1.060 | 1.196 | 0.028 | 0.028 | 0.028 | 0.0035 | 34.808 | 34.134 | 41.040 | 0.032 | 38.281 |
| 4 | astral-sweep-4 | 20.978 | 20.028 | 29.724 | 1.484 | 46.180 | 64.703 | 0.879 | 1.030 | 1.173 | 0.024 | 0.024 | 0.026 | 0.0034 | 36.071 | 35.560 | 40.646 | 0.027 | 39.868 |
| 5 | golden-sweep-5 | 20.053 | 18.817 | 31.435 | 1.671 | 46.222 | 68.936 | 0.971 | 1.124 | 1.158 | 0.033 | 0.034 | 0.030 | 0.0038 | 36.214 | 35.532 | 42.705 | 0.038 | 39.535 |

## 3. Aggregated (mean / std / cv), n=5

### Point forecast

| metric | mean | std | cv |
|---|---|---|---|
| mae/global | 20.3446 | 0.5608 | 0.028 |
| mae/normal | 19.2154 | 0.6450 | 0.034 |
| mae/rare_extreme | 30.7416 | 0.8247 | 0.027 |
| ratio/mae_rare_normal | 1.6016 | 0.0779 | 0.049 |
| rmse/global | 46.2514 | 0.5695 | 0.012 |
| rmse/normal | 43.3341 | 0.7007 | 0.016 |
| rmse/rare_extreme | 67.3945 | 1.7781 | 0.026 |
| ratio/rmse_rare_normal | 1.5557 | 0.0542 | 0.035 |

### Uncertainty (MC std)

| metric | mean | std | cv |
|---|---|---|---|
| uncertainty/mean_std_normal | 0.9074 | 0.0376 | 0.041 |
| uncertainty/mean_std_rare_extreme | 1.0644 | 0.0453 | 0.043 |
| uncertainty/ratio_rare_normal | 1.1731 | 0.0250 | 0.021 |

### PI quantile metrics (primary)

| metric | mean | std | cv |
|---|---|---|---|
| picp_pi/global | 0.0282 | 0.0036 | 0.126 |
| picp_pi/normal | 0.0282 | 0.0038 | 0.134 |
| picp_pi/rare_extreme | 0.0283 | 0.0016 | 0.056 |
| mpiw_pi/global | 3.1613 | 0.1299 | 0.041 |
| mpiw_pi/normal | 3.1087 | 0.1290 | 0.041 |
| mpiw_pi/rare_extreme | 3.6461 | 0.1555 | 0.043 |
| nmpil_pi/global | 0.0035 | 0.0001 | 0.041 |
| nmpil_pi/normal | 0.0035 | 0.0001 | 0.041 |
| nmpil_pi/rare_extreme | 0.0041 | 0.0002 | 0.043 |
| clc_pi/global | 35.6295 | 0.5556 | 0.016 |
| clc_pi/normal | 35.0376 | 0.5893 | 0.017 |
| clc_pi/rare_extreme | 41.0863 | 1.2222 | 0.030 |

### Gaussian diagnostic metrics

| metric | mean | std | cv |
|---|---|---|---|
| picp_gaussian/global | 0.0323 | 0.0040 | 0.125 |
| nmpil_gaussian/global | 0.0040 | 0.0002 | 0.041 |
| clc_gaussian/global | 39.1550 | 0.6097 | 0.016 |

## 4. Paper-style interpretation

- **`picp_pi/global ≈ 0.028`** — MC Dropout is strongly **under-dispersed**: the empirical MC quantile interval covers only ~3% of the truth vs the 0.95 target.
- Intervals are extremely **sharp** (tiny NMPIL) but **not reliable**.
- **Gaussian diagnostic ≈ PI quantile** (`picp_gaussian/global ≈ 0.032` vs `picp_pi/global ≈ 0.028`): the problem is NOT the Gaussian assumption — the **MC spread itself is too narrow**.
- Rare/extreme MAE ≈ **1.60×** normal.
- But MC std rises only ~**1.17×** (rare/normal) → the spread **does not track** the error increase on rare/extreme.
- **PICP rare ≈ normal** (0.028 vs 0.028) → MC Dropout does **not discriminate** coverage across strata.
- **CLC rare > normal** (41.086 vs 35.038) → rare/extreme stays more expensive in the sharpness/reliability trade-off.

## 5. Direct comparison: paper-style vs calibrated sweeps

| sweep | protocol | train | calibration | test | primary PI | mae/global | mae/rare | ratio r/n | primary PICP | NMPIL/CLC (primary) | interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `7c8llckm` | paper-style main | 2016,2017,2018 | none | 2019 | MC quantile PI | 20.34 | 30.74 | 1.60 | picp_pi **0.028** | nmpil 0.0035 / clc 35.63 | honest uncalibrated coverage |
| `ozii0s5s` | post-hoc calibrated variant | 2016,2017 | 2018 (group) | 2019 | calibrated band mean±k·std | 20.90 | 31.46 | 1.59 | coverage_95_calibrated **0.949** | clc_calibrated 0.291 (k≈75) | coverage forced to target |
| `ltxbtupy` | legacy calibrated | 2016,2017 | 2018 (group) | 2019 | gaussian raw + calibrated band | 20.90 | 31.60 | 1.60 | coverage_95_calibrated **0.949** | (raw 0.028, k≈76) | coverage forced to target |

Key contrast: the **~3% coverage is identical** whether read as `picp_pi` here (0.028) or as `coverage_95_raw` in `ltxbtupy` (0.028). The **~0.95 coverage only ever came from post-hoc calibration on 2018** (k≈75), never from the model's own intervals.

## 6. Operational conclusion

- The paper-style protocol is **clean**: no post-hoc calibration, no calibration year, no `k`/`*_calibrated` keys logged.
- MC Dropout produces **sharp but under-reliable** intervals (PICP ≈ 0.028, stable across 5 seeds).
- Adding 2018 to training **slightly improves** the point forecast (mae/global 20.34 vs 20.90 for the train-2016,2017 calibrated sweep), but does **not** fix under-dispersion.
- The 95% coverage exists **only** in the post-hoc calibrated variant, **not** in the paper-style protocol.
- **Next step:** Deep Ensemble — check whether independently trained seed models give larger per-sample dispersion (and thus higher PICP) than a single MC-Dropout model.

