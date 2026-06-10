# PVGIS-only ST-GNN — Deep Ensemble report

**Deep Ensemble** = several ST-GNNs trained independently with different seeds, combined PER TEST SAMPLE. This differs from seed robustness: seed robustness aggregates metrics per seed; the Deep Ensemble combines the seeds' predictions for each sample. The predictive interval is built **directly from the seed predictions** (empirical quantiles) — **PICP is evaluated, not forced**, and **no post-hoc calibration** is used. With only 5 models the empirical quantiles are coarse (close to min/max), so a min/max diagnostic interval is reported alongside.

## Setup

- Members (seeds): **5**  |  samples per member: **10037664**
- coverage_target (gamma): **0.95**  |  clc_eta (eta): **10.0**
- PI quantiles: q0.025 / q0.975  |  target_range (global): **893.2700**
- Pipeline: PVGIS-only; anomaly labels eval-only (stratification), never input/target. No post-hoc calibration.

## Loaded seed files

| file | seed | n_samples |
|---|---|---|
| 0wy9x013_seed4.npz | 4 | 10037664 |
| 1ct8hiug_seed2.npz | 2 | 10037664 |
| g9rpzz8q_seed3.npz | 3 | 10037664 |
| tz5lfk5v_seed1.npz | 1 | 10037664 |
| v09urucu_seed5.npz | 5 | 10037664 |

## Point forecast & uncertainty

| stratum | count | MAE | RMSE | mean ensemble_std |
|---|---|---|---|---|
| global | 10037664 | 19.1925 | 45.0606 | 5.7790 |
| normal | 9054285 | 18.0754 | 42.1137 | 5.5781 |
| rare_extreme | 983379 | 29.4780 | 66.3007 | 7.6285 |

- MAE rare/normal ratio: **1.631**  |  RMSE rare/normal: **1.574**  |  ensemble_std rare/normal: **1.368**

## Ensemble prediction interval (primary, empirical quantiles)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.173 | 15.1574 | 0.0170 | 40.3814 |
| normal | 0.170 | 14.6325 | 0.0164 | 39.8026 |
| rare_extreme | 0.192 | 19.9903 | 0.0224 | 43.9733 |

## Min/max diagnostic interval (secondary)

With few members the quantile band ~ min/max; this is a diagnostic, not the primary PI.

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.180 | 16.1403 | 0.0181 | 39.7862 |
| normal | 0.178 | 15.5825 | 0.0174 | 39.2379 |
| rare_extreme | 0.200 | 21.2757 | 0.0238 | 43.1092 |

## Vs MC-Dropout single model (paper-style)

- No single-model paper-style metrics passed; compare against `picp_pi/global` from the seed-only paper-style sweep manually.

## Operational conclusion

- Deep Ensemble of 5 independent ST-GNN seeds, combined per sample.
- Primary PI coverage (PICP global) = **0.173** — evaluated, not forced; no calibration.
- With 5 members the empirical quantile band is coarse; min/max is the diagnostic upper bound on width.
- If PICP stays far below gamma, the independent-seed disagreement alone does not yield calibrated intervals in the PVGIS-only setting (consistent with the under-dispersed MC-Dropout result).

