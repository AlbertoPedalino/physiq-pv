# Replay-Based Continual Adaptation Protocol

## Objective

Adapt a pre-trained ST-GNN PV forecasting model to new temporal windows
using replay-based continual learning. The goal is to maintain prediction
performance under apparent performance shifts and heterogeneous plant-level
changes without catastrophic forgetting of previously observed regimes.

## Data Modes

| Mode | Description | Use case |
|------|-------------|----------|
| `--data-mode synthetic` | 20-plant synthetic dataset (8760h) | Debug, fast CI, unit tests |
| `--data-mode real` | Piedmont Sentinel/SCADA + PVGIS weather | Thesis experiments |

Real mode uses the same data loading chain as the main training pipeline
(`main.py`):
1. `load_sentinel_hourly()` — hourly ENERGIA from Sentinel CSV files
2. `merge_with_weather()` — temperature, irradiance, wind from PVGIS
3. Coordinate normalization — `latitude`/`longitude` renamed to `lat`/`lon`
4. Missing-coordinate plant filtering

Synthetic mode exists only for debugging and testing. It should not be
used for thesis results.

## Method: Replay-Based Continual Adaptation

At each new temporal window the model is updated using a combination of:

- **Recent data**: samples from the current time window.
- **Replay data**: examples uniformly sampled from a FIFO ring buffer
  containing past observations.

The combined loss is:

```
loss = loss_recent + replay_loss_weight * loss_replay
```

where both `loss_recent` and `loss_replay` use `physics_loss_full`
(MSE_ghi + MSE_pv + lambda * physics_constraint), the same forecast loss
used in the base ST-GNN training.

The replay buffer mitigates catastrophic forgetting by exposing the model
to previously observed regimes during each gradient update.

This protocol implements **only** replay-based continual adaptation.
It does not implement ablation baselines, drift detectors, regime-aware
sampling, or any other advanced continual learning component.

## Features Used by the Model

The model receives the same 16 input features as the base ST-GNN:

| Channel | Feature         | Description                          |
|---------|-----------------|--------------------------------------|
| 0       | temperature_2m  | Normalized 2m temperature            |
| 1       | solar_poa       | Normalized plane-of-array irradiance |
| 2       | wind_speed_10m  | Normalized 10m wind speed            |
| 3       | sin_solar_elev  | sin(apparent solar elevation)        |
| 4       | cos_solar_elev  | cos(apparent solar elevation)        |
| 5       | m1              | Correlation score (QS component)     |
| 6       | m2              | Bias score (QS component)            |
| 7       | m3              | Completeness score (QS component)    |
| 8       | m4              | Variance ratio (QS component)        |
| 9       | m5              | Physical consistency (QS component)  |
| 10      | pv_lag          | Lagged normalized PV output          |
| 11      | kt              | Clearness index                      |
| 12      | kt_std_3h       | 3h rolling std of clearness index    |
| 13      | dghi_dt         | Solar irradiance first difference    |
| 14      | dni_norm        | Direct normal irradiance (Erbs)      |
| 15      | dhi_norm        | Diffuse horizontal irradiance (Erbs) |

### m1..m5 as contextual features

The quality metrics m1..m5 are used as **contextual input features** — the
model learns autonomously how to interpret data quality, missingness, bias,
physical consistency, and reliability of the input signal.

They are **NOT** used as:
- Loss gates (`loss = QS * forecast_loss`)
- Soft weighting of the loss function
- Sampling weights in the replay buffer

## Batch shapes (real data)

Each sample from `PVDataset`:

```
x:      (N_plants, seq_len, 16)    float32  — input features
y_ghi:  (N_plants,)                float32  — GHI target [kW/m^2]
y_pv:   (N_plants,)                float32  — ENERGIA target (normalized)
eta:    (N_plants,)                float32  — per-plant efficiency proxy
ghi_cs: (N_plants,)                float32  — clear-sky GHI [kW/m^2]
```

Batched by DataLoader:

```
x:      (B, N_plants, seq_len, 16)
y_ghi:  (B, N_plants)
y_pv:   (B, N_plants)
eta:    (B, N_plants)
ghi_cs: (B, N_plants)
```

Where N_plants ~ 80-90 for real Piedmont data (after coord filtering).

## What Is Implemented

- Initial offline training on a configurable historical window.
- FIFO ring buffer with uniform random sampling.
- Temporal stream with configurable monthly windows.
- Continual update loop: recent + replay combined loss.
- Per-window evaluation (MAE, RMSE, loss).
- Config, metrics CSV, checkpoints, and summary JSON output.
- Debug mode for fast pipeline verification.
- Real data loading (Sentinel/SCADA + PVGIS weather).
- Synthetic data for debug/testing.
- Sanity check test script.

## What Is NOT Implemented (Future Work)

- Static model baseline (frozen after initial training).
- Naive fine-tuning baseline (no replay).
- Regime-aware / QS-weighted replay sampling.
- DER++ consistency loss (stored-prediction distillation).
- Adapter-based update (frozen backbone + lightweight heads).
- QS-weighted loss or QS-gated updates.
- Automatic drift detection triggers.
- BWT / FWT / Average Forgetting metrics (infrastructure exists in
  `physiq_pv/eval/cl_metrics.py`, not wired into this pipeline yet).

## Example Commands

### Real data — full run on Piedmont 2019

```bash
python -m physiq_pv.continual.train_replay_continual \
    --data-mode real \
    --initial-train-start 2019-03-01 \
    --initial-train-end 2019-05-31 \
    --window-months 1 \
    --replay-buffer-size 5000 \
    --replay-batch-size 64 \
    --replay-loss-weight 1.0 \
    --initial-epochs 5 \
    --update-epochs 1 \
    --seed 42
```

### Real data — debug (few plants, 1 window)

```bash
python -m physiq_pv.continual.train_replay_continual \
    --data-mode real --debug \
    --max-plants 10 --max-windows 1 \
    --seed 42
```

### Synthetic — debug

```bash
python -m physiq_pv.continual.train_replay_continual \
    --data-mode synthetic --debug --seed 42
```

### Sanity checks

```bash
python tests/test_replay_continual.py
```

## Output Structure

```
outputs/continual_replay/<run_name>/
  config.json               — All CLI arguments + device info
  metrics_per_window.csv    — Per-window metrics table
  final_summary.json        — Aggregate summary
  checkpoint_initial.pt     — Model state after initial training
  checkpoint_final.pt       — Model state after last window
  checkpoint_window_<id>.pt — Per-window checkpoints (non-debug only)
```

## Metrics in metrics_per_window.csv

| Column              | Description                                  |
|---------------------|----------------------------------------------|
| window_id           | -1 for initial training, 0+ for stream       |
| window_start        | Window start timestamp                       |
| window_end          | Window end timestamp                         |
| phase               | `initial_train` or `continual_update`        |
| mae                 | Mean absolute error (PV head, normalized)    |
| rmse                | Root mean squared error (PV head, normalized)|
| loss                | physics_loss_full value                      |
| num_recent_samples  | Samples in current window's dataset          |
| num_replay_samples  | Total replay samples used during update      |
| replay_buffer_size  | Buffer occupancy after this window           |
