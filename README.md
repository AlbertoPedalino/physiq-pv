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

## Tests

```bash
.\\.venv\\Scripts\\python.exe tests\\test_sde_net.py
.\\.venv\\Scripts\\python.exe tests\\test_sde_proxy_pipeline.py
```

## CATCH label-unaware anomaly detection

The CATCH implementation follows Wu et al. (ICLR 2025): it applies RevIN,
patches the full complex FFT, discovers a binary channel mask for each frequency
patch, reconstructs the signal in both frequency and time domains, and combines
point-aligned errors from both domains.

The implementation policy is paper-first. Explicit equations and Algorithm 1
take precedence; details omitted by the paper follow the
[official CATCH repository](https://github.com/decisionintelligence/CATCH).
PVGIS-specific departures are limited to leakage-safe train-only scaling and
threshold calibration, year-boundary-safe windows, and complete tail coverage
during non-overlapping scoring.

For PVGIS, each location is treated as an independent multivariate asset. The
default protocol trains on complete years from 2005 through 2018 and keeps 2019
strictly held out:

```bash
python -m physiq_pv.experiments.pvgis_catch_runner \
  --pvgis-dir <pvgis-dir> \
  --train-years 2005-2018 --test-year 2019 \
  --seq-len 192 --patch-size 16 --patch-stride 8 \
  --inference-patch-size 32 --inference-patch-stride 1 \
  --epochs 3 --batch-size 32 \
  --out-dir outputs/pvgis_catch_2005_2019
```

Outputs include all timestamp scores, anomaly-only rows, merged hourly
intervals, per-location summaries, metadata, and a Markdown report. The same
orchestration is available through Python:

```python
from physiq_pv.experiments.pvgis_catch_pipeline import (
    PVGISCATCHConfig,
    run_pvgis_catch,
)

config = PVGISCATCHConfig(
    pvgis_dir="/path/to/pvgis",
    train_years=tuple(range(2005, 2019)),
    test_year=2019,
)
paths = run_pvgis_catch(config)
```
