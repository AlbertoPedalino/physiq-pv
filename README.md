# PhysiQ-PV — PVGIS ST-GNN with neural-SDE uncertainty

PVGIS-only photovoltaic forecasting with a BiLSTM+GAT ST-GNN, neural-SDE
Brownian-path uncertainty, and post-hoc anomaly-stratified evaluation.

## Main components

- `physiq_pv/data/pvgis_dataset.py`: PVGIS loading, windows, normalisation, and normal-only filtering.
- `physiq_pv/data/pvgis_anomaly_scores.py`: seasonal, univariate climatology labels.
- `physiq_pv/model/st_gnn.py`: ST-GNN with drift, diffusion, and aleatoric heads.
- `physiq_pv/training/uncertainty.py`: stochastic SDE inference and prediction intervals.
- `physiq_pv/experiments/pvgis_stgnn_runner.py`: training/evaluation CLI.
- `physiq_pv/reporting/daytime_bin_anomaly_report.py`: daytime post-hoc report.

## Run

Generate seasonal anomaly scores for the test year, and optionally for train
years when running the `--train-normal-only` ablation.

```bash
python -m physiq_pv.experiments.pvgis_stgnn_runner \
  --pvgis-dir <pvgis-dir> \
  --train-years 2016,2017,2018 --test-year 2019 \
  --anomaly-scores <test-scores.csv> \
  --seq-len 24 --horizon 1 --target-variable pv_power_output \
  --model-type stgnn --feature-set full \
  --epochs 10 --batch-size 16 --lr 0.001 --dropout 0.3 \
  --loss-type huber --huber-delta 0.1 \
  --use-irradiance-loss --irradiance-loss-weight 0.1 \
  --sde-uncertainty --mc-samples 20 \
  --out-dir outputs/pvgis_stgnn_sde_seed1
```

To follow the paper-style normal-only protocol, dropping any training window
with a labelled rare/extreme target cell or rare/extreme input history before
SDE training/noise injection, add:

```bash
--train-normal-only --train-anomaly-scores <train-scores.csv>
```

Anomaly labels are never model inputs or prediction targets. In the
normal-only ablation they filter training windows; otherwise they are
evaluation metadata.

With the default aleatoric head, PV is trained with Gaussian NLL; `--loss-type`
controls the optional irradiance auxiliary loss. Use `--no-aleatoric` to train
the PV head with MSE or Huber directly.

## Post-hoc report

```bash
python scripts/analyze_pvgis_huber_daytime_report.py \
  --predictions outputs/pvgis_stgnn_sde_seed1/predictions.csv \
  --out-dir outputs/pvgis_stgnn_sde_seed1 \
  --loss-type huber --huber-delta 0.1 --epochs 10 --dropout 0.3 --mc-samples 20
```

The report computes daytime PICP, MPIW, NMPIL, error metrics, and anomaly
strata. The current labels identify individual meteorological variables that
are unusual for the same location, hour, and seasonal window; they are not a
multivariate OOD definition.

## Tests

```bash
.\\.venv\\Scripts\\python.exe tests\\test_sde_net.py
.\\.venv\\Scripts\\python.exe tests\\test_sde_proxy_pipeline.py
```
