# Sweep 236ajuh7 - seed 1 analysis

## Run

- Run ID: `mpgx2aja`
- Run name: `skilled-sweep-1`
- State: `finished`
- Model: `stgnn_enhanced_dropout`
- Seed: 1
- Feature set: `full` (11 features)
- Train years: 2016, 2017, 2018
- Test year: 2019
- Dropout: 0.2
- MC samples: 20
- Post-hoc calibration: disabled
- Git commit: `4f9ce80252eed5c26b530dfe1395d0558ce94a15`

The enhanced variant reactivates four dropout modules during MC inference:
`temporal_dropout`, `representation_dropout`, `gat.0.dropout`, and
`head_pv.2`.

## Main metrics

| metric | value |
|---|---:|
| mae/global | 20.4104 |
| mae/normal | 19.3839 |
| mae/rare_extreme | 29.8615 |
| ratio/mae_rare_normal | 1.5405 |
| rmse/global | 46.2350 |
| uncertainty/mean_std_global | 7.2214 |
| uncertainty/ratio_rare_normal | 1.2224 |
| picp_pi/global | 0.2159 |
| picp_pi/rare_extreme | 0.2330 |
| mpiw_pi/global | 24.5362 |
| clc_pi/global | 42.3880 |
| picp_gaussian/global | 0.7587 |
| clc_gaussian/global | 0.2464 |

## Seed-1 comparison

| metric | enhanced ST-GNN | standard ST-GNN | LSTM |
|---|---:|---:|---:|
| mae/global | 20.4104 | 19.9366 | 23.2150 |
| mae/normal | 19.3839 | 18.8458 | 21.9943 |
| mae/rare_extreme | 29.8615 | 29.9796 | 34.4550 |
| ratio/mae_rare_normal | 1.5405 | 1.5908 | 1.5665 |
| rmse/global | 46.2350 | 45.9243 | 49.4920 |
| mean std global | 7.2214 | 0.9307 | 9.0131 |
| PICP PI global | 0.2159 | 0.0296 | 0.6669 |
| MPIW PI global | 24.5362 | 3.1897 | 30.8285 |
| PICP Gaussian global | 0.7587 | 0.0339 | 0.6929 |

Compared with standard ST-GNN seed 1, enhanced dropout:

- keeps point accuracy broadly stable: global MAE is 2.4% worse;
- slightly improves rare/extreme MAE by 0.4%;
- reduces the rare/normal MAE ratio from 1.5908 to 1.5405;
- increases mean MC standard deviation by 7.76x;
- increases empirical-PI width by 7.69x;
- raises empirical PICP from 0.0296 to 0.2159.

Compared with LSTM seed 1, enhanced ST-GNN is 12.1% better on global MAE
and 13.3% better on rare/extreme MAE. Its MC dispersion and PI width are
about 20% lower, but its empirical PICP is much worse (0.2159 vs 0.6669).

## Interpretation

The added dropout locations solve the near-zero-dispersion failure of the
standard ST-GNN without materially damaging the point forecast. They do not,
however, produce reliable paper-style empirical intervals: global PICP is
0.2159 against a 0.95 target.

The main diagnostic issue is the large disagreement between intervals built
from the same 20 MC samples:

- empirical quantiles: PICP 0.2159, MPIW 24.5362;
- Gaussian mean +/- 1.96 std: PICP 0.7587, MPIW 28.3078.

A 15.4% width increase alone is unlikely to explain the full coverage jump.
The likely cause is a strongly skewed or heavy-tailed MC distribution. Dropout
on normalized GAT attention weights, several hidden representations, and the
PV head followed by `softplus` can generate asymmetric samples or occasional
outliers. The Gaussian band uses the outlier-sensitive mean and standard
deviation and may extend beyond the empirical sample range; the empirical
2.5/97.5 percentiles are also poorly resolved with only 20 samples.

This is an inference from the aggregate outputs, not a definitive distribution
diagnosis. The run used `skip_predictions_csv=true`, and W&B contains neither
the per-sample predictions nor a model checkpoint. Therefore skewness,
quantile centers, tail behavior, and coverage by error magnitude cannot be
recomputed from the downloaded artifact.

## Anomaly strata

The weakest condition is `unusually_low_solar_potential`:

- MAE: 92.6164 W
- mean std: 13.9547 W
- Gaussian coverage: 0.1816

`extreme_temperature_condition` is comparatively well handled:

- MAE: 18.2407 W
- Gaussian coverage: 0.7749

The aggregate rare/extreme label therefore hides substantial heterogeneity.

## Conclusion

Seed 1 supports the enhanced-dropout hypothesis only partially:

1. Dispersion collapse is substantially reduced.
2. Point accuracy remains close to standard ST-GNN and beats LSTM.
3. Rare/extreme relative degradation improves slightly.
4. Primary empirical PI coverage remains inadequate.
5. The empirical-vs-Gaussian discrepancy must be treated as unresolved until
   a run saves predictions or sufficient MC-distribution diagnostics.
