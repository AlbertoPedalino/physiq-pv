# PhysiQ-PV - Enhanced MC Dropout

PVGIS-only ST-GNN pipeline for uncertainty experiments with enhanced MC Dropout and Deep Ensemble runs.

The supervised target is PVGIS `pv_power_output`; anomaly labels are used only for stratified post-hoc evaluation, never as model inputs or targets. Post-hoc production bins use a single global daytime reference peak by default: `100 * y_true / q99_daytime(y_true)`.

## Pipeline

```text
PVGIS yearly NetCDF files
        |
        v
climatology anomaly scores
        |
        v
ST-GNN / enhanced MC Dropout training
        |
        v
MC predictive intervals + post-hoc diagnostics
        |
        v
Deep Ensemble aggregation
```

## Core Files

```text
notebooks/
  run_training.ipynb                 central notebook for single MC run + W&B sweep ensemble

scripts/
  run_pvgis_climatology_anomaly.py   single-year past-only anomaly scores
  run_pvgis_climatology_anomaly_years.py  multi-year past-only scores + aggregation
  run_pvgis_stgnn_forecasting.py     single training/evaluation wrapper
  run_pvgis_stgnn_sweep_member.py    W&B sweep member for ensemble seeds
  analyze_pvgis_deep_ensemble.py     aggregates ensemble predictions
  analyze_pvgis_interval_miss_distance.py
  analyze_pvgis_daytime_report.py     reference-peak bin/anomaly post-hoc
  interval_miss_utils.py

physiq_pv/
  data/
    pvgis_stgnn_dataset.py
    pvgis_climatology_anomaly.py
  experiments/
    pvgis_stgnn_runner.py
  model/
    st_gnn.py
    bilstm_encoder.py
    graph_builder.py
```

## Main Notebook

Open:

```text
notebooks/run_training.ipynb
```

The notebook centralizes the configuration for:

- one-tag execution via `PIPELINE_TAG = "mc_dropout"` or `"deep_ensemble"`
- one enhanced MC-Dropout run logged to W&B
- a W&B sweep where only `seed` changes for Deep Ensemble members
- Deep Ensemble aggregation logged to W&B
- MC-Dropout and Deep Ensemble post-hoc analysis
- summary tables and figures

Default model:

```text
model_type = stgnn_enhanced_dropout
feature_set = full
seq_len = 24
horizon = 1
dropout = 0.3
mc_samples = 30
```

For Deep Ensemble, the sweep member does not pass `--mc-dropout`; each seed is a deterministic model and the ensemble interval is built from between-seed predictions.

To run end-to-end from the notebook, set:

```python
PIPELINE_TAG = "mc_dropout"      # or "deep_ensemble"
RUN_PIPELINE = True
```

## CLI

Generate anomaly scores:

```powershell
python scripts/run_pvgis_climatology_anomaly_years.py ^
  --years 2016,2017,2018 ^
  --pvgis-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance ^
  --climatology-start-year 2005 ^
  --climatology-end-year 2018 ^
  --quantile 0.975 ^
  --climatology-window-days 15 ^
  --out-root outputs
```

The anomaly generator is always past-only: target year `Y` uses climatology
years `2005..min(end_year, Y - 1)`. For example, 2018 writes
`outputs/pvgis_anomaly_2018_2005_2017_w15_q0975/pvgis_climatology_scores.csv`.

Single run:

```powershell
python scripts/run_pvgis_stgnn_forecasting.py ^
  --pvgis-dir /data/SentinelPV/pvgis_data/data/pvgis_summed_irradiance ^
  --train-years 2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018 ^
  --test-year 2019 ^
  --anomaly-scores outputs/pvgis_anomaly_2019_2005_2018_w15_q0975/pvgis_climatology_scores.csv ^
  --model-type stgnn_enhanced_dropout ^
  --feature-set full ^
  --seq-len 24 ^
  --epochs 60 ^
  --batch-size 8 ^
  --dropout 0.3 ^
  --mc-dropout ^
  --mc-samples 30 ^
  --wandb
```

Deep Ensemble:

Use the W&B sweep cell in `notebooks/run_training.ipynb`. The sweep varies only `seed`; all other hyperparameters are fixed by the notebook command.
