# PVGIS ST-GNN interval-miss diagnostics (post-hoc, eval-only)

Computed from saved predictions only — no training, no model, no loss change. Distances are in **watt** and **conditional on the missed rows** (`mean_outside_distance` = mean watt outside the band, over outside rows only). `required_multiplier = |y_true - y_pred_mean| / (std + eps)` is computed on all rows of the stratum. Anomaly labels are eval-only.

## Inputs

- predictions: `outputs/pvgis_stgnn_enhanced_mse_dropout03_seed1/predictions.csv`
- interval: `pi`
- eps: `1e-06`
- daytime_threshold_wm2: `10.0`
- multipliers: `[1.0, 1.96, 2.5, 3.0, 4.0, 5.0, 8.0, 10.0]`
- n_rows: `10037664`

## 1. Interval-miss distances per stratum

| group | n | inside_interval_count | outside_interval_count | inside_interval_pct | outside_interval_pct | above_interval_pct | below_interval_pct | mean_residual | median_residual | mean_outside_distance | median_outside_distance | p90_outside_distance | p95_outside_distance | mean_above_distance | median_above_distance | p90_above_distance | p95_above_distance | mean_below_distance | median_below_distance | p90_below_distance | p95_below_distance | p50_required_multiplier | p90_required_multiplier | p95_required_multiplier |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| global | 10037664 | 2599290 | 7438374 | 0.259 | 0.741 | 0.1284 | 0.6127 | -1.324 | 0.9847 | 13.61 | 0.1286 | 41.61 | 81.42 | 39.22 | 21.12 | 99.75 | 140.4 | 8.237 | 0.1129 | 14.48 | 49.7 | 0.8868 | 3.529 | 5.742 |
| daytime | 4747545 | 2563735 | 2183810 | 0.54 | 0.46 | 0.2714 | 0.1886 | -4.812 | -6.023 | 44.73 | 23.5 | 114.4 | 162.3 | 39.23 | 21.12 | 99.76 | 140.4 | 52.66 | 27.47 | 136 | 192.1 | 1.521 | 5.884 | 8.667 |
| nighttime | 5290119 | 35555 | 5254564 | 0.006721 | 0.9933 | 2.439e-05 | 0.9933 | 1.806 | 1.082 | 0.6699 | 0.1042 | 0.1859 | 0.2554 | 0.2632 | 0.185 | 0.5978 | 0.7531 | 0.6699 | 0.1041 | 0.1859 | 0.2553 | 0.8015 | 1.111 | 1.237 |
| normal | 9054285 | 2317875 | 6736410 | 0.256 | 0.744 | 0.1288 | 0.6152 | -2.298 | 0.978 | 12.53 | 0.1265 | 39.18 | 76.8 | 39.16 | 21.34 | 99.08 | 139.1 | 6.95 | 0.1115 | 10.18 | 42.88 | 0.8806 | 3.442 | 5.582 |
| rare_extreme | 983379 | 281415 | 701964 | 0.2862 | 0.7138 | 0.1249 | 0.589 | 7.644 | 1.065 | 23.98 | 0.1669 | 68.5 | 136.9 | 39.85 | 19.05 | 106.4 | 152.5 | 20.62 | 0.1317 | 52.11 | 128.8 | 0.9636 | 4.327 | 7.534 |
| normal_daytime | 4195053 | 2286383 | 1908670 | 0.545 | 0.455 | 0.2779 | 0.1771 | -6.367 | -6.564 | 43.86 | 23.51 | 111.9 | 157.2 | 39.16 | 21.34 | 99.09 | 139.1 | 51.22 | 27.52 | 131.5 | 182.8 | 1.507 | 5.849 | 8.547 |
| rare_extreme_daytime | 552492 | 277352 | 275140 | 0.502 | 0.498 | 0.2222 | 0.2758 | 6.994 | -1.564 | 50.81 | 23.37 | 136.2 | 202.8 | 39.85 | 19.05 | 106.4 | 152.6 | 59.64 | 27.29 | 164.1 | 241.7 | 1.627 | 6.178 | 9.782 |
| normal_nighttime | 4859232 | 31492 | 4827740 | 0.006481 | 0.9935 | 2.428e-05 | 0.9935 | 1.215 | 1.079 | 0.1384 | 0.1038 | 0.1832 | 0.2455 | 0.2623 | 0.2009 | 0.561 | 0.7666 | 0.1384 | 0.1038 | 0.1832 | 0.2454 | 0.8002 | 1.105 | 1.225 |
| rare_extreme_nighttime | 430887 | 4063 | 426824 | 0.009429 | 0.9906 | 2.553e-05 | 0.9905 | 8.476 | 1.112 | 6.682 | 0.1085 | 0.2341 | 0.5574 | 0.2726 | 0.121 | 0.6687 | 0.68 | 6.682 | 0.1085 | 0.2341 | 0.5573 | 0.8181 | 1.194 | 1.459 |
| label:unusually_low_solar_potential | 118147 | 15443 | 102704 | 0.1307 | 0.8693 | 0.002091 | 0.8672 | 92.88 | 47.41 | 79.65 | 28.26 | 271.9 | 343 | 11.4 | 7.716 | 26.24 | 33.38 | 79.81 | 28.36 | 272.3 | 343.2 | 2.754 | 15.7 | 26.09 |
| label:unusually_high_solar_potential | 38401 | 18358 | 20043 | 0.4781 | 0.5219 | 0.5192 | 0.00276 | -49.37 | -36.7 | 42.95 | 20.19 | 115.1 | 169.7 | 43.04 | 20.24 | 115.4 | 169.9 | 24.27 | 10.38 | 67.04 | 113.5 | 1.796 | 6.048 | 9.294 |
| label:extreme_temperature_condition | 436524 | 157450 | 279074 | 0.3607 | 0.6393 | 0.1219 | 0.5174 | -2.997 | 0.9297 | 11.26 | 0.1326 | 29.01 | 62.15 | 32.35 | 14.73 | 86.07 | 133.2 | 6.287 | 0.1141 | 6.703 | 30.31 | 0.8715 | 2.736 | 4.265 |
| label:extreme_wind_condition | 458693 | 114698 | 343995 | 0.2501 | 0.7499 | 0.1242 | 0.6258 | 0.8156 | 1.036 | 17.34 | 0.1363 | 56.47 | 107.4 | 47.97 | 25.77 | 123.6 | 169.3 | 11.26 | 0.1192 | 26.04 | 74.97 | 0.9068 | 4.155 | 7.051 |
| daytime_0_20 | 396207 | 216025 | 180182 | 0.5452 | 0.4548 | 0.1725 | 0.2822 | 10.73 | 1.051 | 13.82 | 6.492 | 33.11 | 48.5 | 3.552 | 2.885 | 7.599 | 9.233 | 20.1 | 13.37 | 42.96 | 62.91 | 1.463 | 4.168 | 5.612 |
| daytime_20_40 | 376626 | 221820 | 154806 | 0.589 | 0.411 | 0.07736 | 0.3337 | 20.61 | 10.35 | 27.49 | 13.38 | 65.4 | 103.5 | 8.715 | 7.496 | 17.82 | 21.12 | 31.85 | 16.32 | 75.68 | 118.2 | 1.237 | 3.785 | 5.347 |
| daytime_40_60 | 352255 | 227224 | 125031 | 0.6451 | 0.3549 | 0.06081 | 0.2941 | 24.85 | 6.626 | 48.38 | 20.04 | 129.8 | 214.7 | 11.28 | 8.364 | 26.11 | 31.54 | 56.05 | 25.09 | 152.3 | 235.7 | 1.093 | 4.125 | 6.607 |
| daytime_60_80 | 266151 | 167603 | 98548 | 0.6297 | 0.3703 | 0.09161 | 0.2787 | 23.44 | 5.438 | 49.52 | 21.4 | 133.1 | 205.4 | 11.81 | 9.066 | 26.13 | 32.68 | 61.91 | 31.64 | 161.5 | 233.7 | 1.232 | 4.309 | 6.749 |
| daytime_80_100 | 206451 | 123894 | 82557 | 0.6001 | 0.3999 | 0.1531 | 0.2468 | 18.03 | 0.6556 | 49.61 | 21.87 | 138.9 | 208.8 | 14.97 | 12.28 | 31.52 | 37.82 | 71.1 | 38.9 | 188.3 | 253.4 | 1.384 | 4.668 | 7.405 |
| daytime_gt_100 | 3149855 | 1607169 | 1542686 | 0.5102 | 0.4898 | 0.3536 | 0.1362 | -17.01 | -15.91 | 49.21 | 29.12 | 121.7 | 165.8 | 44.04 | 25.95 | 108.2 | 149.2 | 62.63 | 40.35 | 151.6 | 199.7 | 1.684 | 6.702 | 9.605 |

## 2. PICP curve — mean ± k·std

Diagnostic band from the MC std; shows the coverage a pure std rescale would buy (k=1.96 = the Gaussian diagnostic band).

| group | n | picp_k_1 | picp_k_1.96 | picp_k_2.5 | picp_k_3 | picp_k_4 | picp_k_5 | picp_k_8 | picp_k_10 |
|---|---|---|---|---|---|---|---|---|---|
| global | 10037664 | 0.5978 | 0.8058 | 0.8485 | 0.8774 | 0.9152 | 0.9381 | 0.9716 | 0.9817 |
| daytime | 4747545 | 0.3592 | 0.594 | 0.6825 | 0.743 | 0.8227 | 0.871 | 0.9416 | 0.9628 |
| nighttime | 5290119 | 0.8118 | 0.9959 | 0.9975 | 0.9979 | 0.9983 | 0.9984 | 0.9986 | 0.9986 |
| normal | 9054285 | 0.6058 | 0.8123 | 0.8534 | 0.8812 | 0.918 | 0.9405 | 0.9735 | 0.9834 |
| rare_extreme | 983379 | 0.5242 | 0.7462 | 0.8028 | 0.8416 | 0.8897 | 0.9162 | 0.9537 | 0.9656 |
| normal_daytime | 4195053 | 0.3629 | 0.5966 | 0.684 | 0.7438 | 0.8231 | 0.8716 | 0.9429 | 0.9642 |
| rare_extreme_daytime | 552492 | 0.3312 | 0.5742 | 0.6707 | 0.7372 | 0.8202 | 0.8663 | 0.9314 | 0.9518 |
| normal_nighttime | 4859232 | 0.8154 | 0.9984 | 0.9997 | 0.9999 | 1 | 1 | 1 | 1 |
| rare_extreme_nighttime | 430887 | 0.7716 | 0.9667 | 0.9722 | 0.9754 | 0.9787 | 0.9802 | 0.9823 | 0.9833 |
| label:unusually_low_solar_potential | 118147 | 0.07813 | 0.2983 | 0.4405 | 0.55 | 0.686 | 0.7519 | 0.8332 | 0.8583 |
| label:unusually_high_solar_potential | 38401 | 0.228 | 0.5455 | 0.6606 | 0.7343 | 0.8218 | 0.8688 | 0.9349 | 0.9562 |
| label:extreme_temperature_condition | 436524 | 0.6081 | 0.84 | 0.8854 | 0.913 | 0.9448 | 0.9609 | 0.9816 | 0.9875 |
| label:extreme_wind_condition | 458693 | 0.5781 | 0.7867 | 0.8282 | 0.8571 | 0.8955 | 0.9199 | 0.9591 | 0.9724 |
| daytime_0_20 | 396207 | 0.3712 | 0.6171 | 0.7254 | 0.8015 | 0.8906 | 0.9336 | 0.9817 | 0.9914 |
| daytime_20_40 | 376626 | 0.4186 | 0.6935 | 0.7873 | 0.8455 | 0.9103 | 0.9426 | 0.9783 | 0.9867 |
| daytime_40_60 | 352255 | 0.4683 | 0.7158 | 0.7928 | 0.8399 | 0.8953 | 0.9249 | 0.9618 | 0.9718 |
| daytime_60_80 | 266151 | 0.4217 | 0.6856 | 0.7707 | 0.8247 | 0.8874 | 0.9206 | 0.9609 | 0.972 |
| daytime_80_100 | 206451 | 0.3787 | 0.6432 | 0.7348 | 0.7952 | 0.8709 | 0.9107 | 0.9551 | 0.9666 |
| daytime_gt_100 | 3149855 | 0.3319 | 0.5547 | 0.6413 | 0.7022 | 0.787 | 0.8417 | 0.9274 | 0.9543 |

## 3. Widen-vs-bias verdict per stratum

Heuristic on `p95_required_multiplier`: <= 3 -> widening/calibration plausible; >= 6 -> predictive center too biased (or std too small) — fix the point forecast. Above/below asymmetry + mean_residual tell the direction of the bias.

| group | n | p95_required_multiplier | above% | below% | mean_residual | verdict |
|---|---|---|---|---|---|---|
| global | 10037664 | 5.74 | 0.128 | 0.613 | -1.32 | borderline — partial fix from widening, residual center bias likely |
| daytime | 4747545 | 8.67 | 0.271 | 0.189 | -4.81 | center biased or std collapsed — widening alone will not fix it |
| nighttime | 5290119 | 1.24 | 2.44e-05 | 0.993 | 1.81 | moderate widening/calibration could suffice |
| normal | 9054285 | 5.58 | 0.129 | 0.615 | -2.3 | borderline — partial fix from widening, residual center bias likely |
| rare_extreme | 983379 | 7.53 | 0.125 | 0.589 | 7.64 | center biased or std collapsed — widening alone will not fix it |
| normal_daytime | 4195053 | 8.55 | 0.278 | 0.177 | -6.37 | center biased or std collapsed — widening alone will not fix it |
| rare_extreme_daytime | 552492 | 9.78 | 0.222 | 0.276 | 6.99 | center biased or std collapsed — widening alone will not fix it |
| normal_nighttime | 4859232 | 1.23 | 2.43e-05 | 0.993 | 1.21 | moderate widening/calibration could suffice |
| rare_extreme_nighttime | 430887 | 1.46 | 2.55e-05 | 0.991 | 8.48 | moderate widening/calibration could suffice |
| label:unusually_low_solar_potential | 118147 | 26.1 | 0.00209 | 0.867 | 92.9 | center biased or std collapsed — widening alone will not fix it |
| label:unusually_high_solar_potential | 38401 | 9.29 | 0.519 | 0.00276 | -49.4 | center biased or std collapsed — widening alone will not fix it |
| label:extreme_temperature_condition | 436524 | 4.26 | 0.122 | 0.517 | -3 | borderline — partial fix from widening, residual center bias likely |
| label:extreme_wind_condition | 458693 | 7.05 | 0.124 | 0.626 | 0.816 | center biased or std collapsed — widening alone will not fix it |
| daytime_0_20 | 396207 | 5.61 | 0.173 | 0.282 | 10.7 | borderline — partial fix from widening, residual center bias likely |
| daytime_20_40 | 376626 | 5.35 | 0.0774 | 0.334 | 20.6 | borderline — partial fix from widening, residual center bias likely |
| daytime_40_60 | 352255 | 6.61 | 0.0608 | 0.294 | 24.8 | center biased or std collapsed — widening alone will not fix it |
| daytime_60_80 | 266151 | 6.75 | 0.0916 | 0.279 | 23.4 | center biased or std collapsed — widening alone will not fix it |
| daytime_80_100 | 206451 | 7.41 | 0.153 | 0.247 | 18 | center biased or std collapsed — widening alone will not fix it |
| daytime_gt_100 | 3149855 | 9.61 | 0.354 | 0.136 | -17 | center biased or std collapsed — widening alone will not fix it |

