# PVGIS forecasting baseline

A minimal, self-contained pipeline that trains/evaluates a simple **PVGIS-only**
forecasting baseline and evaluates it **stratified** by the rare/extreme labels
produced by the climatology anomaly pipeline.

## What it does

- Task: `input = last seq_len hours of PVGIS variables` →
  `target = one PVGIS variable, horizon hours ahead` (default `pv_power_output`).
- Baselines:
  - `persistence` (default, mandatory): `y_hat(t+horizon) = y(t)` — the last
    observed target value. No training.
  - `mlp` (optional): a small torch MLP on flattened windows, trained on the
    train years and evaluated on the test year. Uses torch, already a project
    dependency; no new dependencies are added.
- Evaluation is split into `normal` vs `rare_or_extreme` (and per specific
  label) so we can answer:

  > Does a PVGIS-only forecast do worse on the rare / extreme PVGIS conditions
  > found by the climatology anomaly pipeline?

## Why it stays separate from the main training

- It does **not** touch `train.py` or the ST-GNN, and is not linked to the main
  model.
- It uses **only** PVGIS data — no real plant production, no Sentinel/SCADA
  energy.
- Anomaly labels are used **only** for stratified evaluation, never as a
  supervised target.
- The `persistence` baseline is a first, simple, interpretable reference — not a
  performance-maximising model.

## Inputs

- Annual PVGIS NetCDF files in `--pvgis-dir` (e.g.
  `piedmont_pvgis_2005.nc … piedmont_pvgis_2023.nc`). One or more years.
- Optional `--anomaly-scores`: a `pvgis_climatology_scores.csv` from the
  climatology anomaly pipeline. A prediction is tagged `rare_or_extreme` when
  its `(location, timestamp)` appears in that file, otherwise `normal`. The
  specific labels are kept for per-label metrics.

Input variables (used if present, else ignored):

```
solar_irradiance_poa, pv_power_output, temperature_2m, wind_speed_10m, sun_height
```

No NetCDF is written, no API download, no heavy data copied into the repo.

## Outputs

```
outputs/pvgis_forecasting_2019/predictions.csv                 # per-point y_true/y_pred/errors + anomaly_label
outputs/pvgis_forecasting_2019/metrics_global.csv              # MAE / RMSE / count (all points)
outputs/pvgis_forecasting_2019/metrics_by_anomaly_label.csv    # MAE / RMSE / count per stratum
outputs/pvgis_forecasting_2019/report.md                       # human-readable report + verdict
```

`predictions.csv` columns: `location, timestamp, target_variable, y_true,
y_pred, error, abs_error, squared_error, anomaly_label, anomaly_labels_detail`.

## Example (server)

```bash
PYTHONPATH=$PWD python scripts/run_pvgis_forecasting_baseline.py \
  --pvgis-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance \
  --train-years 2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018 \
  --test-year 2019 \
  --anomaly-scores outputs/pvgis_anomaly_2019_2005_2023_w15_q099/pvgis_climatology_scores.csv \
  --out-dir outputs/pvgis_forecasting_2019 \
  --seq-len 24 \
  --horizon 1 \
  --target-variable pv_power_output \
  --baseline persistence
```

For the optional MLP baseline use `--baseline mlp` (the `--train-years` are then
required; MLP training is subsampled to `--mlp-max-train-samples` to stay
tractable). Persistence ignores `--train-years` and needs no training.
