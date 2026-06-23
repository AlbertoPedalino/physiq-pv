# PhysiQ-PV — PVGIS ST-GNN with paper-faithful SDE-Net uncertainty

PVGIS-only photovoltaic forecasting with a BiLSTM+GAT ST-GNN, neural-SDE
Brownian-path uncertainty, and post-hoc anomaly-stratified evaluation.

## Main components

- `physiq_pv/data/pvgis_dataset.py`: PVGIS loading, windows, normalisation, and normal-only masks.
- `physiq_pv/data/pvgis_anomaly_scores.py`: seasonal, univariate climatology labels.
- `physiq_pv/model/sde_net.py`: faithful vector SDE-Net core plus the authors'
  YearMSD regression architecture.
- `physiq_pv/model/st_gnn.py`: PV ST-GNN with SDE-Net's drift/diffusion block.
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
  --ood-noise-std 2.0 --sde-uncertainty --mc-samples 10 \
  --out-dir outputs/pvgis_stgnn_sde_seed1
```

To use only label-defined normal target/history cells in the PV and irradiance
task losses, add:

```bash
--train-normal-only --train-anomaly-scores <train-scores.csv>
```

Anomaly labels are never model inputs or prediction targets. In the
normal-only ablation they select training cells; otherwise they are evaluation
metadata.

## SDE-Net fidelity

The implementation follows [Kong, Sun & Zhang (ICML 2020)](https://arxiv.org/abs/2008.10546)
and the [authors' reference repository](https://github.com/Lingkai-Kong/SDE-Net):

- Euler–Maruyama dynamics use a scalar, initial-state diffusion per example,
  broadcast over the latent state for every time step;
- Algorithm 1 drives diffusion down on ID and up on pseudo-OOD data; the
  authors' public repository implements that step as BCE (ID=0, pseudo-OOD=1),
  while the encoder, drift and PV heads receive the task loss;
- pseudo-OOD inputs are `x + 2·N(0, I)` by default; and
- the two optimisation paths use separate SGD optimisers (momentum 0.9,
  weight decay `5e-4`), with one path per training update and 10 paths at test
  time.

For the v1 YearMSD setup, sigma is scheduled from `0.01` to `0.5` at epoch 30
via `--sde-sigma-initial` and `--sde-sigma-warmup-epochs`. The archived Kong
repository sets `0.1` rather than `0.01` in its training script; both values
remain configurable, and the default follows the supplied v1 PDF.

`YearMSDSDENet` is a direct dependency-free implementation of the paper's
YearMSD regressor (`90 → 50 → {mean, aleatoric sigma}`) and is covered by CPU
tests. The ST-GNN remains a PV-domain adaptation: its temporal/GAT encoder and
forecast heads replace the paper's raw-feature encoder/head, while the SDE and
diffusion-training semantics are retained.

## Post-hoc report

```bash
python scripts/analyze_pvgis_huber_daytime_report.py \
  --predictions outputs/pvgis_stgnn_sde_seed1/predictions.csv \
  --out-dir outputs/pvgis_stgnn_sde_seed1 \
  --loss-type huber --huber-delta 0.1 --epochs 10 --dropout 0.3 --mc-samples 10
```

The report computes daytime PICP, MPIW, NMPIL, error metrics, anomaly
strata, and frequency-weighted production-bin calibration. The current labels identify individual meteorological variables that
are unusual for the same location, hour, and seasonal window; they are not a
multivariate OOD definition.

## Tests

```bash
.\\.venv\\Scripts\\python.exe tests\\test_sde_net.py
.\\.venv\\Scripts\\python.exe tests\\test_sde_proxy_pipeline.py
```
