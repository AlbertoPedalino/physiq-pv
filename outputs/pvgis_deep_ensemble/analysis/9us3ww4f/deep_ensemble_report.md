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
| 0aytdzim_seed2.npz | 2 | 10037664 |
| dqb8j69z_seed1.npz | 1 | 10037664 |
| o6d5xg45_seed4.npz | 4 | 10037664 |
| pp7jx7a3_seed3.npz | 3 | 10037664 |
| qub1w3vt_seed5.npz | 5 | 10037664 |

## Point forecast & uncertainty

| stratum | count | MAE | RMSE | mean ensemble_std |
|---|---|---|---|---|
| global | 10037664 | 19.5824 | 45.2329 | 6.2427 |
| normal | 9054285 | 18.5178 | 42.3414 | 6.0082 |
| rare_extreme | 983379 | 29.3849 | 66.1628 | 8.4013 |

- MAE rare/normal ratio: **1.587**  |  RMSE rare/normal: **1.563**  |  ensemble_std rare/normal: **1.398**

## Ensemble prediction interval (primary, empirical quantiles)

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.178 | 16.3159 | 0.0183 | 41.0992 |
| normal | 0.176 | 15.7029 | 0.0176 | 40.5349 |
| rare_extreme | 0.201 | 21.9601 | 0.0246 | 44.1594 |

## Min/max diagnostic interval (secondary)

With few members the quantile band ~ min/max; this is a diagnostic, not the primary PI.

| stratum | PICP | MPIW | NMPIL | CLC |
|---|---|---|---|---|
| global | 0.186 | 17.4089 | 0.0195 | 40.4826 |
| normal | 0.184 | 16.7495 | 0.0188 | 39.9553 |
| rare_extreme | 0.210 | 23.4806 | 0.0263 | 43.1744 |

## Vs MC-Dropout single model (paper-style)

- No single-model paper-style metrics passed; compare against `picp_pi/global` from the seed-only paper-style sweep manually.

## Operational conclusion

- Deep Ensemble of 5 independent ST-GNN seeds, combined per sample.
- Primary PI coverage (PICP global) = **0.178** — evaluated, not forced; no calibration.
- With 5 members the empirical quantile band is coarse; min/max is the diagnostic upper bound on width.
- If PICP stays far below gamma, the independent-seed disagreement alone does not yield calibrated intervals in the PVGIS-only setting (consistent with the under-dispersed MC-Dropout result).

