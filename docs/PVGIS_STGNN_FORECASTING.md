# PVGIS-only ST-GNN forecasting

Reuses the existing **STGNN** architecture on a real **PVGIS-only** dataset, to
check the deterministic behaviour of ST-GNN on PVGIS data before touching plant
data, uncertainty, or continual learning.

## Key point: PVGIS-only input

The main model uses 16 features, 6 of which depend on real plant data
(`m1..m5` quality-score channels + a real `pv_lag`). This pipeline **removes**
those entirely (not neutralised) and uses **11 PVGIS-only features**:

```
temperature_2m, solar_irradiance_poa, wind_speed_10m   (meteo, z-scored)
sin_elev, cos_elev                                     (solar geometry, pvlib)
kt, kt_std_3h, dghi_dt, dni_norm, dhi_norm             (derived irradiance)
pv_lag_pvgis                                           (lag of PVGIS pv_power_output)
```

- `pv_lag_pvgis` is the past **PVGIS** `pv_power_output` (normalised by the
  per-location p99 of daytime pv), sliced causally inside each window. It does
  **not** come from observed plant production.
- No `ENERGIA`, no `compute_qs`, no `kWp`/`UPN`/`load_kwp`, no Sentinel/SCADA,
  no plant-quality filters.

`STGNN` is reused as-is via `make_model(n_nodes, seq_len, n_features=11)` — the
architecture already takes `n_features` as a constructor argument, so no model
code is modified. The model's GHI head is unused here (`ghi_cs=None`); training
optimises only `pred_pv` against the PVGIS target.

## Task

```
input : last seq_len hours of PVGIS-only features, per location node
target: PVGIS pv_power_output, horizon hours ahead (default +1h)
```

Nodes = PVGIS locations; the geographic graph is built from location lat/lon
(`build_graph`, edges ≤ `--max-dist-km`).

## Anomaly labels (stratified evaluation only)

Pass `--anomaly-scores <pvgis_climatology_scores.csv>`. A prediction whose
`(location, timestamp)` appears in that file is tagged `rare_or_extreme`
(`anomaly_group`), otherwise `normal`; the specific labels are kept in
`anomaly_label`. Anomaly labels are **never** model input nor a supervised
target.

## Normalisation

Fitted on **train years only**: per-location `pv_scale` = p99 of daytime
`pv_power_output`; global z-score stats for `temperature_2m`,
`solar_irradiance_poa`, `wind_speed_10m`, `dghi_dt`. Predictions are
inverse-scaled back to physical `pv_power_output` units before metrics.

## Outputs

```
outputs/pvgis_stgnn_forecasting_2019/predictions.csv                 # timestamp, location, y_true, y_pred, error, abs_error, squared_error, anomaly_group, anomaly_label
outputs/pvgis_stgnn_forecasting_2019/metrics_global.csv              # MAE / RMSE / count (all)
outputs/pvgis_stgnn_forecasting_2019/metrics_by_anomaly_label.csv    # MAE / RMSE / count per stratum
outputs/pvgis_stgnn_forecasting_2019/report.md                       # report + verdict
```

The report answers: **does ST-GNN degrade on rare/extreme PVGIS conditions vs
normal ones?** (ratio of `rare_or_extreme` MAE to `normal` MAE).

## Example (server)

```bash
PYTHONPATH=$PWD python scripts/run_pvgis_stgnn_forecasting.py \
  --pvgis-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance \
  --train-years 2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018 \
  --test-year 2019 \
  --anomaly-scores outputs/pvgis_anomaly_2019_2005_2023_w15_q099/pvgis_climatology_scores.csv \
  --out-dir outputs/pvgis_stgnn_forecasting_2019 \
  --seq-len 24 \
  --horizon 1 \
  --target-variable pv_power_output \
  --epochs 10 \
  --batch-size 8
```

Memory knobs: `--max-train-samples`, `--max-test-samples` (random subsample of
windows), `--device cpu|cuda`.

## Not in scope yet

No Monte Carlo Dropout, no ensemble, no uncertainty, no continual learning, no
real plant data, no comparison with real production. This is the deterministic
PVGIS-only check only.
