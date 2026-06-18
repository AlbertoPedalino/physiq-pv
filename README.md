# PhysiQ-PV — PVGIS-only ST-GNN (Huber + MC-Dropout + SDE-proxy)

Minimal branch that reproduces **one** experiment: a PVGIS-only Spatio-Temporal
GNN forecaster trained with Huber loss, MC-Dropout uncertainty, a train-time
MC-Dropout uncertainty penalty in **SDE-proxy** mode, and anomaly-aware input
noise injection — plus the post-hoc daytime report.

Everything not needed for this run (continual learning, replay, deep ensemble,
conformal runtime, sweeps, old data pipelines) has been removed.

---

## Repository structure

```
physiq_pv/
  data/
    pvgis_dataset.py        PVGIS .nc loading, windowing, normalization,
                            datasets, make_model, anomaly-label attach,
                            run metrics + report writers
    pvgis_anomaly_scores.py climatology anomaly scoring (generates the
                            --anomaly-scores / --train-anomaly-scores CSVs)
  model/
    st_gnn.py               STGNN (BiLSTM temporal encoder + GAT + dual head)
    bilstm_encoder.py       temporal encoder
    graph_builder.py        geographic graph (edges <= 20 km, weight = 1/dist)
  training/
    losses.py               Huber/MSE point loss, under-dispersion penalty,
                            SDE-proxy OOD penalty
    noise.py                train-only input noise (random + anomaly-aware)
    uncertainty.py          MC-Dropout inference + predictive intervals
    train_loop.py           train_model (the training loop)
  experiments/
    pvgis_stgnn_runner.py   CLI entrypoint for the run
  reporting/
    daytime_bin_anomaly_report.py  post-hoc daytime bin x anomaly report

scripts/
  run_pvgis_climatology_anomaly.py        single-year anomaly scoring
  run_pvgis_climatology_anomaly_years.py  multi-year + aggregation
  analyze_pvgis_huber_daytime_report.py   thin wrapper -> reporting module

tests/
  test_pvgis_huber_noise_uncertainty.py   Huber, anomaly noise, SDE-proxy, MC
```

Dependencies (`pyproject.toml`): `numpy`, `pandas`, `torch`, `xarray`,
`netcdf4` (xarray `.nc` backend), `pvlib`, `wandb`.

---

## 1. Generate the anomaly scores (input prerequisite)

The run reads two anomaly-score CSVs produced from the PVGIS climatology. They
live under `outputs/` (git-ignored). Regenerate them with:

```bash
# Test year (2019)
PYTHONPATH=$PWD python scripts/run_pvgis_climatology_anomaly_years.py \
  --years 2019 \
  --pvgis-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance \
  --climatology-start-year 2005 --climatology-end-year 2023 \
  --quantile 0.975 --climatology-window-days 15 --min-climatology-years 3 \
  --variables solar_irradiance_poa pv_power_output temperature_2m wind_speed_10m \
  --out-root outputs

# Train years (2016-2018), aggregated into ONE CSV
PYTHONPATH=$PWD python scripts/run_pvgis_climatology_anomaly_years.py \
  --years 2016,2017,2018 \
  --pvgis-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance \
  --climatology-start-year 2005 --climatology-end-year 2023 \
  --quantile 0.975 --climatology-window-days 15 --min-climatology-years 3 \
  --variables solar_irradiance_poa pv_power_output temperature_2m wind_speed_10m \
  --out-root outputs \
  --aggregate-out-dir outputs/pvgis_anomaly_train_2016_2018_2005_2023_w15_q0975
```

Produces, among others:
- `outputs/pvgis_anomaly_2019_2005_2023_w15_q0975/pvgis_climatology_scores.csv`
- `outputs/pvgis_anomaly_train_2016_2018_2005_2023_w15_q0975/pvgis_climatology_scores.csv`

---

## 2. Run the final experiment

```bash
python -m physiq_pv.experiments.pvgis_stgnn_runner \
  --pvgis-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance \
  --train-years 2016,2017,2018 \
  --test-year 2019 \
  --anomaly-scores outputs/pvgis_anomaly_2019_2005_2023_w15_q0975/pvgis_climatology_scores.csv \
  --train-anomaly-scores outputs/pvgis_anomaly_train_2016_2018_2005_2023_w15_q0975/pvgis_climatology_scores.csv \
  --seq-len 24 --horizon 1 --target-variable pv_power_output \
  --model-type stgnn_enhanced_dropout --feature-set full \
  --use-irradiance-head --use-irradiance-loss --irradiance-loss-weight 0.1 \
  --epochs 5 --batch-size 16 --lr 0.001 --dropout 0.3 \
  --mc-dropout --mc-samples 20 --pv-target-clip-max none --device cuda --seed 1 \
  --loss-type huber --huber-delta 0.1 \
  --train-mc-uncertainty-penalty --train-mc-samples 5 \
  --uncertainty-penalty-mode sde_proxy \
  --sde-proxy-in-weight 0.00005 --sde-proxy-out-weight 0.5 --sde-proxy-std-min-ood 0.10 \
  --train-noise-mode anomaly --train-noise-std 0.0 --train-noise-prob 0.0 \
  --anomaly-noise-std 0.05 --anomaly-noise-prob 0.7 \
  --skip-posthoc-analysis \
  --out-dir outputs/pvgis_stgnn_huber_d01_sdeproxy_in005e4_out05_ood010_seed1 \
  --wandb --wandb-project PhysiQ-PV \
  --wandb-entity albertopedalino-politecnico-di-torino \
  --wandb-run-name pvgis_stgnn_huber_d01_sdeproxy_in005e4_out05_ood010_ktaux_w01_seed1
```

Writes `predictions.csv`, `metrics.json` and `report.md` under `--out-dir`.

---

## 3. Post-hoc daytime report

```bash
PRED_DIR="outputs/pvgis_stgnn_huber_d01_sdeproxy_in005e4_out05_ood010_seed1"

python scripts/analyze_pvgis_huber_daytime_report.py \
  --predictions "$PRED_DIR/predictions.csv" \
  --out-dir "$PRED_DIR" \
  --loss-type huber --huber-delta 0.1 --epochs 5 --dropout 0.3 --mc-samples 20
```

Outputs (daytime = `solar_irradiance_poa_target` above threshold), stratified by
production bin and anomaly category:
`daytime_anomaly_overview.csv`, `daytime_bin_summary.csv`,
`daytime_bin_anomaly_metrics.csv`, `uncertainty_response.csv`,
`sharpness_overview.csv`, `daytime_bin_anomaly_report.md`.

---

## Metrics

- **PICP** (Prediction Interval Coverage Probability) — fraction of test points
  whose true value falls inside `[lower_pi, upper_pi]`. Should track the nominal
  coverage (here 0.95). Measures interval **coverage** (reliability).
- **MPIW** (Mean Prediction Interval Width) — `mean(upper_pi - lower_pi)`.
  Measures **sharpness**: narrower is better *at equal coverage*.
- **NMPIL** (Normalized Mean Prediction Interval Length) — `MPIW / target_range`,
  with `target_range = max(y_true) - min(y_true)` over valid daytime rows. Scale-
  free width, comparable across bins/scopes.
- `*_ratio_vs_normal` columns (e.g. `mpiw_ratio_vs_normal`, `nmpil_ratio_vs_normal`)
  express a category's value relative to the `normal` daytime subset.

Plus standard `MAE`, `RMSE`, and `mean_std` (mean MC predictive std) per scope.

---

## Tests

```bash
uv run --with pytest pytest tests/test_pvgis_huber_noise_uncertainty.py
# or, with pytest installed:
pytest tests/test_pvgis_huber_noise_uncertainty.py
```

---

## Notebook pipeline

`notebooks/pvgis_sde_proxy_pipeline.ipynb` is a thin orchestrator for the whole
experiment. It does **not** reimplement the runner or the analysis script — it
builds their commands and reads the CSVs they produce, via
`physiq_pv.experiments.sde_proxy_pipeline` (also covered by
`tests/test_sde_proxy_pipeline.py`).

**Open** (on server `newzealand`, repo root):
```bash
jupyter lab notebooks/pvgis_sde_proxy_pipeline.ipynb
```
Run the **Setup** cell first — it sets the repo root / `PYTHONPATH`, defines the
PVGIS + anomaly-score paths and checks they exist (printing the regeneration
commands if the anomaly scores are missing). Nothing trains until you flip a
switch; every action is gated (`RUN_TRAINING`, `RUN_ANALYSIS`, `RUN_SWEEP`,
`CREATE_SWEEP`, `LOG_TO_WANDB`), all default `False`.

**Single run** — set `RUN_TRAINING = True` (training) and `RUN_ANALYSIS = True`
(post-hoc), then run sections 2–7. `out_dir` and `wandb_run_name` are derived
from the config so re-running the same config reuses the dir and different
configs never collide.

**Manual sweep** — edit `SWEEP_CONFIGS`, set `RUN_SWEEP = True` (section 8): each
config trains + analyses sequentially, generates the standard figures, resumes
the matching W&B training run when `wandb_run.json` is available, logs post-hoc
scalars/images, uploads a compact artifact, and collects a post-hoc summary
table.

**Real W&B sweep** — section 9 builds a sweep config (`make_sweep_config`) whose
`program` is `scripts/run_pvgis_sde_proxy_sweep_member.py`. Set
`CREATE_SWEEP = True` to register it, then launch agents:
```bash
wandb agent albertopedalino-politecnico-di-torino/PhysiQ-PV/<sweep_id>
```
Each agent run reads `wandb.config`, trains (runner W&B off — the wrapper owns
the run), runs the analysis, generates the standard figures, logs the post-hoc
scalars/images, and uploads a compact artifact with reports, metrics CSV/JSON,
post-hoc CSVs and figures (`predictions.csv` excluded by default).

**Outputs & figures** — under `<out_dir>` (git-ignored `outputs/`):
`predictions.csv`, the metrics CSVs, `report.md`, the post-hoc CSVs, and
`<out_dir>/figures/*.png` (`residual_histogram`, `interval_width_histogram`,
`absolute_error_by_bin_boxplot`, `interval_width_by_bin_boxplot`,
`picp_mpiw_nmpil_by_bin`, `uncertainty_response_ratios`).
The training runner writes `<out_dir>/wandb_run.json` so later post-hoc logging
can resume the same W&B run instead of creating a duplicate display-name run.

**W&B post-hoc scalars** logged (per scope/bin, from the analysis CSVs):
`posthoc/daytime_picp`, `posthoc/daytime_mpiw`, `posthoc/daytime_nmpil`,
`posthoc/normal_picp`, `posthoc/rare_extreme_picp`, `posthoc/unusually_low_picp`,
`posthoc/gt100_picp`. Metrics to watch: **PICP** (coverage), **MPIW** /
**NMPIL** (sharpness), **MAE**, **RMSE**, **mean_std**.

Only the notebook itself is tracked (`.gitignore`: `notebooks/*` +
`!notebooks/pvgis_sde_proxy_pipeline.ipynb`); commit it with outputs cleared.

---

## Methodological note on anomaly labels

Anomaly labels are used in **two distinct** ways:

- **Training run** — they *are* used: to target anomaly-aware input-noise
  injection (`--train-noise-mode anomaly`) and as the in-distribution vs OOD
  split for the SDE-proxy penalty (`--uncertainty-penalty-mode sde_proxy`).
- **Post-hoc report** — they are used **only** to stratify already-written
  predictions; the report never trains or modifies the model.

So this is *not* an "anomaly-labels-eval-only" configuration.

Outputs (`outputs/`, `wandb/`, `checkpoints/`, generated CSV/MD/NPZ/logs) are
git-ignored and must never be committed.
