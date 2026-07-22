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

## M2AD label-unaware anomaly detection

The implementation of Alnegheimish et al. (AISTATS 2025) is split by
responsibility:

- `anomaly_detection/m2ad_preprocessing.py`: train-only interpolation/scaling;
- `anomaly_detection/m2ad_forecaster.py`: windows, stacked LSTM, early stopping;
- `anomaly_detection/m2ad_errors.py`: point/area discrepancy and EWMA;
- `anomaly_detection/m2ad_calibration.py`: GMM, BIC, Fisher, Gamma threshold;
- `anomaly_detection/m2ad.py`: small public facade composing those stages;
- `data/pvgis_m2ad.py`: PVGIS-to-asset adapter;
- `experiments/pvgis_m2ad_pipeline.py`: typed, argparse-free Python pipeline;
- `experiments/pvgis_m2ad_runner.py`: thin CLI adapter;
- `reporting/m2ad_outputs.py`: intervals, CSV contract, and Markdown report.

The LSTM predicts the next multivariate observation, per-sensor residual
distributions are fitted with GMMs, weighted Fisher scores are aggregated
globally, and a moment-matched Gamma distribution supplies the threshold. The
implementation never accepts anomaly labels.

For PVGIS, each location is treated as an independent asset and the default
sensors are PV output, plane-of-array irradiance, temperature, and wind. The
default temporal protocol uses all complete years from 2005 through 2018 for
training/calibration and keeps 2019 strictly held out:

```bash
python -m physiq_pv.experiments.pvgis_m2ad_runner \
  --pvgis-dir <pvgis-dir> \
  --train-years 2005-2018 --test-year 2019 \
  --window-size 120 --error area --area-half-window 2 \
  --epochs 30 --gmm-components bic \
  --out-dir outputs/pvgis_m2ad_2005_2019
```

The 120-hour history, area half-window `l=2`, 30 epochs, `(-1, 1)` scaling,
and Gamma significance `0.001` follow the paper's hourly case study and public
implementation. BIC chooses one to three GMM components per sensor as described
in Appendix A.4. Windows, interpolation, and EWMA state never cross annual
boundaries; every preprocessing statistic is fitted on training only.

As in the paper, calibration assumes that the training interval predominantly
represents normal operation. Because this protocol is deliberately label-unaware,
2005–2018 is not filtered with climatology annotations; substantial contamination
can make the learned threshold conservative, while a full distribution shift in
2019 calls for retraining.

Outputs include all timestamp scores, anomaly-only rows, merged hourly
intervals, per-location calibration summaries, metadata, and a Markdown report.
See `notebooks/pvgis_m2ad_pipeline.ipynb` for a reproducible launcher.

The same orchestration is available without CLI or notebook coupling:

```python
from physiq_pv.experiments.pvgis_m2ad_pipeline import (
    PVGISM2ADConfig,
    run_pvgis_m2ad,
)

config = PVGISM2ADConfig(
    pvgis_dir="/path/to/pvgis",
    train_years=tuple(range(2005, 2019)),
    test_year=2019,
)
paths = run_pvgis_m2ad(config)
```
