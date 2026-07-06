# PVGIS-only ST-GNN forecasting

Reuses the existing **STGNN** architecture on a real **PVGIS-only** dataset, as a
deterministic forecasting check before touching plant data.

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

## Normalisation

Fitted on **train years only**: per-location `pv_scale` = p99 of daytime
`pv_power_output`; global z-score stats for `temperature_2m`,
`solar_irradiance_poa`, `wind_speed_10m`, `dghi_dt`. Predictions are
inverse-scaled back to physical `pv_power_output` units before metrics.

## Outputs

```
outputs/<out-dir>/predictions.csv        # per-(location, timestamp) predictions
outputs/<out-dir>/metrics_global.csv     # count / MAE / RMSE
outputs/<out-dir>/metrics.json           # machine-readable global metrics
outputs/<out-dir>/report.md              # report
```

`predictions.csv` columns: `timestamp, location, y_true, y_pred, error,
abs_error, squared_error`.

`metrics_global.csv` columns: `stratum, count, MAE, RMSE` (`stratum` = `all`).

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

Off by default. Enable with `--wandb [--wandb-entity E] [--wandb-project P]
[--wandb-run-name N] [--wandb-log-predictions]` (lazily imported — W&B is never a
hard dependency). For the thesis runs use `--wandb-entity
albertopedalino-politecnico-di-torino --wandb-project PhysiQ-PV`.

Logged **config:** `mode, model_type, feature_set, selected_features,
n_features, target_variable, train_years, test_year, seq_len, horizon, epochs,
batch_size, lr, dropout, device, max_train_samples, max_test_samples,
skip_predictions_csv, wandb_log_predictions`.

Speed knobs: `--batch-size`, `--epochs`, `--max-train-samples`,
`--skip-predictions-csv` (metrics + report.md still written; predictions.csv
omitted). A `[config] …` banner at the start of every run echoes the effective
seed/batch_size/epochs/…, and `[time] …` lines report per-phase wall-clock
(dataset build, per-epoch + total training, test inference, writing outputs,
total run).

Logged **metrics** (namespaced for sweep dashboards):

```
mae/global   rmse/global
```

The numeric scalar dict is printed to stdout (under `Key metrics:`) even without
W&B, so nothing requires the dependency.

**Per-run output dir.** With `--wandb`, output goes to a unique folder so sweep
runs never overwrite each other: pass an explicit `--out-dir
outputs/wandb_pvgis_stgnn/{wandb_run_id}` (or `{wandb_run_name}`), or leave
`--out-dir` at its default and it is auto-redirected to
`outputs/wandb_pvgis_stgnn/<run_id>/`.

**Artifacts.** Every run always writes `predictions.csv`, `metrics_global.csv`,
`report.md` locally. Under `--wandb` a single artifact (`pvgis_stgnn_<run_id>`,
type `pvgis_stgnn_outputs`) is logged with `report.md` + `metrics_global.csv`;
`predictions.csv` is added **only** with `--wandb-log-predictions` (off by
default — it can be very large).

## Sweeps (W&B)

Ready-made sweep configs live in `configs/sweeps/`:

| file | purpose | optimises |
|------|---------|-----------|
| `pvgis_stgnn_ablation.yaml`  | feature-set + lr/dropout/batch_size grid | `mae/global` |
| `pvgis_stgnn_debug.yaml`     | tiny/fast smoke of the pipeline | `mae/global` |

Each sweep runs `main.py --mode pvgis_stgnn`; the swept params are emitted by
`${args_no_boolean_flags}` as `--param=value` and matched by the underscore CLI
aliases (`--feature_set`, `--max_train_samples`, …). `--out-dir
outputs/wandb_pvgis_stgnn/{wandb_run_id}` keeps every run's files unique. Edit the
fixed `--pvgis-dir` / `--train-years` in each YAML's `command:` block before
launching.

```bash
# create + run an agent (PYTHONPATH so main.py / physiq_pv import)
PYTHONPATH=$PWD wandb sweep configs/sweeps/pvgis_stgnn_ablation.yaml
PYTHONPATH=$PWD wandb agent <SWEEP_ID> --count 10
```

## Example (server)

```bash
PYTHONPATH=$PWD python main.py \
  --mode pvgis_stgnn \
  --pvgis-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance \
  --train-years 2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018 \
  --test-year 2019 \
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
  --out-dir outputs/pvgis_stgnn_forecasting_2019_no_pv_lag \
  --seq-len 24 --horizon 1 --target-variable pv_power_output \
  --model-type stgnn --feature-set no_pv_lag \
  --epochs 10 --batch-size 8 --lr 0.001 --device cuda
```

Memory knobs: `--max-train-samples`, `--max-test-samples` (random subsample of
windows), `--device cpu|cuda`.

## Baselines (`--model-type`)

Only `stgnn` is implemented in this PVGIS-only runner. `persistence` / `mlp` are
scaffolded (clean "not implemented yet"). Real-plant baselines use
`ENERGIA`/Sentinel and are therefore **not** PVGIS-only.

## Not in scope

No real plant data and no comparison with real production.
