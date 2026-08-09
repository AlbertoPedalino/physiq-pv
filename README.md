# PhysiQ-PV — PVGIS ST-GNN with neural-SDE uncertainty

PVGIS-only photovoltaic forecasting with a BiLSTM+GAT ST-GNN, neural-SDE
Brownian-path uncertainty, and post-hoc anomaly-stratified evaluation.

## Main components

- `physiq_pv/data/pvgis_dataset.py`: PVGIS loading, windows, normalisation, and normal-only filtering.
- `physiq_pv/data/pvgis_anomaly_scores.py`: seasonal, univariate climatology labels.
- `physiq_pv/model/st_gnn.py`: Monaco-style parallel drift/diffusion encoders on
  a BiLSTM+GAT backbone.
- `docs/CODE_FLOW_SDE.md`: runtime map and exact Monaco-to-STGNN correspondence.
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

## Monaco SDE adaptation

`n_sde_steps` is the number of aligned stochastic encoder stages.  The first
stage is temporal (a drift BiLSTM and a dedicated diffusion BiLSTM); each
remaining stage is a paired drift/diffusion GAT.  Every diffusion stage emits
a bounded per-node/per-feature sigmoid gate and injects an independent Brownian
kick into its matching drift stage.  The diffusion gates are trained with the
sum of Monaco's per-stage BCE losses (ID=0, Gaussian pseudo-OOD=1), separately
from the MSE forecasting path.

Reference: [Monaco et al. (2025)](https://doi.org/10.1016/j.cageo.2025.105992),
*Uncertainty-aware methods for enhancing rainfall prediction with deep-learning
based post-processing segmentation*, and the authors'
[`probabilistic-rainprediction`](https://github.com/simone7monaco/probabilistic-rainprediction)
SDE U-Net implementation.

The PV point head and optional irradiance auxiliary head are trained with MSE,
matching Monaco's public SDE U-Net regression objective.  Predictive intervals
come from repeated stochastic SDE forwards, not from a separate variance head.

## Post-hoc report

```bash
python scripts/analyze_pvgis_daytime_report.py \
  --predictions outputs/pvgis_stgnn_sde_seed1/predictions.csv \
  --out-dir outputs/pvgis_stgnn_sde_seed1
```

The report computes daytime PICP, MPIW, NMPIL, error metrics, and anomaly
strata. The current labels identify individual meteorological variables that
are unusual for the same location, hour, and seasonal window; they are not a
multivariate OOD definition.

## Geographical STGAN anomaly detector

The STGAN detector trains jointly on all PVGIS locations. Locations are graph
nodes connected through Haversine KNN subgraphs; meteorological variables are
node attributes. Training and threshold fitting use only the historical split,
while the test split is scored without anomaly labels.

```bash
python scripts/prepare_pvgis_stgan.py \
  --pvgis-dir <pvgis-dir> \
  --out-dir outputs/pvgis_stgan/prepared \
  --train-start 2005 --train-end 2018 --test-year 2019

python scripts/run_pvgis_stgan.py \
  --manifest outputs/pvgis_stgan/prepared/manifest.csv \
  --out-dir outputs/pvgis_stgan/run \
  --device cuda
```

Each seed produces location-level scores, feature residuals for flagged test
points, per-location diagnostic files, and a reloadable checkpoint.
`locations.csv` maps every score location to latitude/longitude for spatial
joins and heatmaps. Defaults reproduce the published STGAN architecture and
complete shuffled training product; replacement sampling remains an explicit
scalability option for PVGIS. See `notebooks/stgan_pvgis_workflow.ipynb` for
the paper-to-code correspondence and the full workflow.

## Tests

```bash
.\\.venv\\Scripts\\python.exe tests\\test_sde_net.py
.\\.venv\\Scripts\\python.exe tests\\test_sde_proxy_pipeline.py
```
