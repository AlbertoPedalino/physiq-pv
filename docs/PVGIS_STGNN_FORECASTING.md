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

`STGNN` is reused as-is via `make_model(n_nodes, seq_len, n_features)` — the
architecture already takes `n_features` as a constructor argument, so no model
code is modified. The model's GHI head is unused here (`ghi_cs=None`); training
optimises only `pred_pv` against the PVGIS target. `n_features` is **not**
hardcoded to 11: it is `len(selected_features)` for the chosen `--feature-set`
(see *Feature-set ablation* below).

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
outputs/<out-dir>/predictions.csv                 # per-(location, timestamp) predictions
outputs/<out-dir>/metrics_global.csv              # count / MAE / RMSE (+ uncertainty cols)
outputs/<out-dir>/metrics_by_anomaly_label.csv    # same, per anomaly stratum
outputs/<out-dir>/report.md                       # report + verdict (+ uncertainty section)
```

`predictions.csv` columns:

- **deterministic:** `timestamp, location, y_true, y_pred, error, abs_error,
  squared_error, anomaly_group, anomaly_label`
- **MC Dropout (`--mc-dropout`):** the above **plus** `y_pred_mean, y_pred_std,
  y_pred_lower, y_pred_upper` (inserted after `y_pred`). For back-compat
  `y_pred == y_pred_mean`, and `y_pred_lower/upper = y_pred_mean ∓ 1.96 *
  y_pred_std`.

`metrics_global.csv` / `metrics_by_anomaly_label.csv` columns: `stratum, count,
MAE, RMSE, mean_pred_std, median_pred_std, p90_pred_std, coverage_95`. The four
uncertainty columns are `NaN` in the deterministic path and populated under
`--mc-dropout` (`coverage_95` = fraction of `y_true` inside the 95% band).

The report answers: **does ST-GNN degrade on rare/extreme PVGIS conditions vs
normal ones?** (ratio of `rare_or_extreme` MAE to `normal` MAE). Under
`--mc-dropout` it adds an **Uncertainty by anomaly stratum** section answering
(1) is the error higher on rare/extreme, and (2) is the model also more
*uncertain* there.

## Integration with `main.py`

This pipeline is wired into the main entrypoint as an experimental **mode**, so
sweep / ablation runs go through `main.py` (the default pipeline is untouched):

```bash
# default mode = existing real Piedmont pipeline, unchanged
python main.py

# PVGIS-only ST-GNN experiment
python main.py --mode pvgis_stgnn ...
```

Both `main.py --mode pvgis_stgnn` and `scripts/run_pvgis_stgnn_forecasting.py`
delegate to the same runner (`physiq_pv/experiments/pvgis_stgnn_runner.py`), so
they accept identical flags.

## Feature-set ablation

`--feature-set` selects a subset of the 11 features; the model is then built
with `STGNN(n_features=len(selected))` — **never hardcoded to 11**.

| feature_set            | features                                                                                                   | n_features |
|------------------------|------------------------------------------------------------------------------------------------------------|:----------:|
| `full`                 | temperature_2m, solar_irradiance_poa, wind_speed_10m, sin_elev, cos_elev, kt, kt_std_3h, dghi_dt, dni_norm, dhi_norm, pv_lag_pvgis | 11 |
| `no_pv_lag`            | `full` minus pv_lag_pvgis                                                                                   | 10 |
| `no_derived_irradiance`| temperature_2m, solar_irradiance_poa, wind_speed_10m, sin_elev, cos_elev, pv_lag_pvgis                     | 6  |
| `meteo_only`           | temperature_2m, solar_irradiance_poa, wind_speed_10m, sin_elev, cos_elev                                   | 5  |
| `irradiance_only`      | solar_irradiance_poa, sin_elev, cos_elev, kt, kt_std_3h, dghi_dt, dni_norm, dhi_norm                       | 8  |

Selection always keeps the canonical feature order (channel subsetting is by
index), so results are comparable across sets.

## Model types

`--model-type stgnn` is implemented. `persistence` and `mlp` are **scaffolded**
in the runner's registry but not implemented yet — selecting them fails with a
clean "not implemented yet" message (never a silent fallback).

## W&B (optional, sweep-ready)

Off by default. Enable with `--wandb [--wandb-project P] [--wandb-run-name N]`
(lazily imported — W&B is never a hard dependency).

Logged **config:** `mode, model_type, feature_set, selected_features,
n_features, target_variable, train_years, test_year, seq_len, horizon, epochs,
batch_size, lr, dropout, device, max_train_samples, max_test_samples,
mc_dropout, mc_samples, anomaly_scores`.

Logged **metrics** (namespaced for sweep dashboards):

```
mae/global   rmse/global
mae/normal   rmse/normal
mae/rare_extreme   rmse/rare_extreme
ratio/mae_rare_normal   ratio/rmse_rare_normal
# only when --mc-dropout:
uncertainty/mean_std_global   uncertainty/mean_std_normal   uncertainty/mean_std_rare_extreme
uncertainty/ratio_rare_normal
coverage_95/global   coverage_95/normal   coverage_95/rare_extreme
```

The same scalar dict is printed to stdout (under `Key metrics:`) even without
W&B, so nothing requires the dependency.

## MC Dropout (uncertainty estimation)

Implemented. Run with `--mc-dropout --mc-samples N` (needs `--dropout > 0`).

Inference path (`predict_mc` in `physiq_pv/data/pvgis_stgnn_dataset.py`):

1. train the model normally (deterministic, unchanged);
2. `model.eval()` (whole model stays in eval — BiLSTM/LayerNorm deterministic);
3. `enable_dropout_only(model)` reactivates **only** `nn.Dropout` (and
   `Dropout2d`/`Dropout3d`) via `module.train()`; `model.train()` is **never**
   called on the whole model — confirmed at runtime by printing
   `model.training=False` with the count of reactivated dropout layers;
4. run `--mc-samples` forward passes per batch;
5. aggregate per-(location, timestamp): `y_pred_mean`, `y_pred_std`, and the
   band `y_pred_mean ∓ 1.96 * y_pred_std`;
6. `y_pred_mean` drives MAE/RMSE; `y_pred_std` is the uncertainty;
   `coverage_95` checks calibration.

With `gat_layers=1`, the active stochastic layer is the GAT attention dropout;
raise `--dropout` for a wider predictive band. The BiLSTM's *internal* dropout
is an `nn.LSTM` argument (not a module), so it deliberately stays off — only
true `nn.Dropout` modules are sampled, per spec.

## Sweeps (W&B)

Ready-made sweep configs live in `configs/sweeps/`:

| file | purpose | optimises |
|------|---------|-----------|
| `pvgis_stgnn_ablation.yaml`  | feature-set + lr/dropout/batch_size grid (no MC) | `mae/rare_extreme` |
| `pvgis_stgnn_mc_dropout.yaml`| MC-Dropout uncertainty grid (dropout × mc_samples) | `mae/rare_extreme` (monitor `uncertainty/ratio_rare_normal`) |
| `pvgis_stgnn_debug.yaml`     | tiny/fast smoke of both branches | `mae/global` |

Each sweep runs `main.py --mode pvgis_stgnn`; the swept params are emitted by
`${args_no_boolean_flags}` as `--param=value` and matched by the underscore CLI
aliases (`--feature_set`, `--max_train_samples`, …). Boolean `mc_dropout` is
emitted as a bare `--mc_dropout` only on its `true` runs. Edit the fixed
`--pvgis-dir` / `--anomaly-scores` / `--train-years` in each YAML's `command:`
block before launching.

```bash
# create + run an agent (PYTHONPATH so main.py / physiq_pv import)
PYTHONPATH=$PWD wandb sweep configs/sweeps/pvgis_stgnn_ablation.yaml
PYTHONPATH=$PWD wandb agent <SWEEP_ID>

# or the helper (creates the sweep and launches the agent in one step):
scripts/experiments/run_pvgis_stgnn_sweep.sh configs/sweeps/pvgis_stgnn_debug.yaml
scripts/experiments/run_pvgis_stgnn_sweep.sh configs/sweeps/pvgis_stgnn_ablation.yaml 20
```

## Example (server)

```bash
PYTHONPATH=$PWD python main.py \
  --mode pvgis_stgnn \
  --pvgis-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance \
  --train-years 2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018 \
  --test-year 2019 \
  --anomaly-scores outputs/pvgis_anomaly_2019_2005_2023_w15_q099/pvgis_climatology_scores.csv \
  --out-dir outputs/pvgis_stgnn_forecasting_2019_full \
  --seq-len 24 --horizon 1 --target-variable pv_power_output \
  --model-type stgnn --feature-set full \
  --epochs 10 --batch-size 8 --lr 0.001 --device cuda
```

Ablation (drop the PVGIS pv lag):

```bash
PYTHONPATH=$PWD python main.py \
  --mode pvgis_stgnn \
  --pvgis-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance \
  --train-years 2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018 \
  --test-year 2019 \
  --anomaly-scores outputs/pvgis_anomaly_2019_2005_2023_w15_q099/pvgis_climatology_scores.csv \
  --out-dir outputs/pvgis_stgnn_forecasting_2019_no_pv_lag \
  --seq-len 24 --horizon 1 --target-variable pv_power_output \
  --model-type stgnn --feature-set no_pv_lag \
  --epochs 10 --batch-size 8 --lr 0.001 --device cuda
```

MC Dropout (uncertainty):

```bash
PYTHONPATH=$PWD python main.py \
  --mode pvgis_stgnn \
  --pvgis-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance \
  --train-years 2016,2017,2018 --test-year 2019 \
  --anomaly-scores outputs/pvgis_anomaly_2019_2005_2023_w15_q0975/pvgis_climatology_scores.csv \
  --out-dir outputs/pvgis_stgnn_2019_q0975_full_e10_mc20 \
  --seq-len 24 --horizon 1 --target-variable pv_power_output \
  --model-type stgnn --feature-set full \
  --epochs 10 --batch-size 8 --lr 0.001 --dropout 0.2 --device cuda \
  --max-train-samples 50000 \
  --mc-dropout --mc-samples 20
```

Memory knobs: `--max-train-samples`, `--max-test-samples` (random subsample of
windows), `--device cpu|cuda`.

## Baselines (`--model-type`)

Only `stgnn` is implemented in this PVGIS-only runner. `persistence` / `mlp` are
scaffolded (clean "not implemented yet"). A **real-data** persistence baseline
already exists separately at `scripts/experiments/persistence_baseline.py`, but
it uses `ENERGIA`/Sentinel and is therefore **not** PVGIS-only — keep it out of
PVGIS comparisons.

## Not in scope yet

No ensemble, no continual learning, no real plant data, no comparison with real
production. MC Dropout uncertainty **is** implemented (above); this remains a
PVGIS-only experiment runnable from `main.py` for sweep / ablation / uncertainty.
