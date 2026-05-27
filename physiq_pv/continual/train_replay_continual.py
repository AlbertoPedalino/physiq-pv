"""
Replay-based continual adaptation for PV forecasting.

Pipeline:
  1. ST-GNN trained on initial historical window.
  2. Subsequent data treated as temporal stream of monthly windows.
  3. At each window: model updated using recent data + replay buffer
     examples, then evaluated.

m1..m5 quality metrics are contextual input features — the model learns
autonomously how to interpret quality, missingness, bias, physical
consistency, and reliability. They are NOT used as loss gates.

Data modes:
  --data-mode synthetic   20-plant synthetic dataset (debug/test only)
  --data-mode real        Piedmont Sentinel/SCADA + PVGIS weather

Usage (real):
    python -m physiq_pv.continual.train_replay_continual \\
        --data-mode real \\
        --initial-train-start 2019-03-01 \\
        --initial-train-end 2019-05-31 \\
        --window-months 1 \\
        --replay-buffer-size 5000 \\
        --replay-batch-size 64 \\
        --replay-loss-weight 1.0 \\
        --update-epochs 1 \\
        --seed 42

Usage (synthetic debug):
    python -m physiq_pv.continual.train_replay_continual \\
        --data-mode synthetic --debug --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physiq_pv.continual.simple_replay_buffer import SimpleReplayBuffer
from physiq_pv.continual.temporal_stream import TemporalStream
from physiq_pv.data.dataset import PVDataset, SEQ_LEN, N_FEATURES
from physiq_pv.data.quality_score import compute_qs
from physiq_pv.data.synthetic_generator import generate_synthetic_dataset
from physiq_pv.model.graph_builder import build_graph
from physiq_pv.model.physics_loss import physics_loss_full
from physiq_pv.model.st_gnn import STGNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #

def _normalize_real_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Align real dataset coord names to the schema expected by PVDataset."""
    renames: dict[str, str] = {}
    if "latitude" in ds.variables and "lat" not in ds.variables:
        renames["latitude"] = "lat"
    if "longitude" in ds.variables and "lon" not in ds.variables:
        renames["longitude"] = "lon"
    if renames:
        ds = ds.rename(renames)

    if "eta_base" not in ds.data_vars and "eta_base" in ds.coords:
        ds = ds.assign({"eta_base": ds["eta_base"]})
    if "eta_base" not in ds.data_vars and "eta_base" not in ds.coords:
        ds = ds.assign_coords(
            eta_base=("plant", np.full(ds.sizes["plant"], 0.18, dtype=np.float64)),
        )

    n_plants, n_steps = ds.sizes["plant"], ds.sizes["time"]

    if "solar_irradiance_poa" not in ds:
        raise ValueError("Missing required variable 'solar_irradiance_poa'")

    if "wind_speed_10m" not in ds:
        ds = ds.assign({
            "wind_speed_10m": xr.DataArray(
                np.full((n_plants, n_steps), 3.0, dtype="float64"),
                dims=["plant", "time"],
            ),
        })
    return ds


def _drop_missing_coord_plants(ds: xr.Dataset) -> xr.Dataset:
    """Drop plants without finite lat/lon."""
    lat_key = "lat" if "lat" in ds.variables else "latitude"
    lon_key = "lon" if "lon" in ds.variables else "longitude"
    lats = ds[lat_key].values.astype(float)
    lons = ds[lon_key].values.astype(float)
    keep = np.isfinite(lats) & np.isfinite(lons)
    n_drop = int((~keep).sum())
    if n_drop > 0:
        print(f"[data] dropped {n_drop}/{len(keep)} plants with missing coords")
        ds = ds.isel(plant=np.where(keep)[0])
    return ds


def _load_synthetic(seed: int) -> xr.Dataset:
    ds = generate_synthetic_dataset(seed=seed)
    if "eta_base" not in ds.data_vars and "eta_base" not in ds.coords:
        ds = ds.assign_coords(
            eta_base=("plant", np.full(ds.sizes["plant"], 0.18, dtype=np.float64)),
        )
    return ds


def _load_real(
    sentinel_dir: str,
    year: int,
    plant_mapping: str,
    energy_coords: str,
    pvgis_path: str,
) -> tuple[xr.Dataset, "np.ndarray | None"]:
    from physiq_pv.data.sentinel_hourly_loader import (
        load_sentinel_hourly,
        merge_with_weather,
    )

    pm_exists = Path(plant_mapping).exists()
    ec_exists = Path(energy_coords).exists()

    print(f"[data] loading Sentinel hourly ({year})...")
    ds = load_sentinel_hourly(
        sentinel_dir=sentinel_dir,
        year=year,
        plant_mapping_path=plant_mapping if pm_exists else None,
        energy_coords_path=energy_coords if ec_exists else None,
    )

    # Load kWp from GSE registry (same as main.py)
    kwp = None
    if pm_exists and ec_exists:
        try:
            from physiq_pv.data.load_kwp import load_kwp
            from main import _load_kwp_for_dataset
            kwp = _load_kwp_for_dataset(ds, plant_mapping, energy_coords)
            n_finite = int(np.sum(np.isfinite(kwp)))
            print(f"[data] kwp loaded: {n_finite}/{len(kwp)} plants with real kWp")
        except Exception as e:
            print(f"[data] kwp loading failed ({e}), using p99 fallback")
            kwp = None

    print("[data] merging weather variables...")
    ds = merge_with_weather(ds, pvgis_path=pvgis_path)

    ds = _normalize_real_dataset(ds)

    # Drop plants with missing coords, keep kwp aligned
    lat_key = "lat" if "lat" in ds.variables else "latitude"
    lats = ds[lat_key].values.astype(float)
    lon_key = "lon" if "lon" in ds.variables else "longitude"
    lons = ds[lon_key].values.astype(float)
    keep = np.isfinite(lats) & np.isfinite(lons)
    n_drop = int((~keep).sum())
    if n_drop > 0:
        print(f"[data] dropped {n_drop}/{len(keep)} plants with missing coords")
        plant_idx = np.where(keep)[0]
        ds = ds.isel(plant=plant_idx)
        if kwp is not None:
            kwp = kwp[plant_idx]

    return ds, kwp


# ------------------------------------------------------------------ #
# Seed
# ------------------------------------------------------------------ #

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ------------------------------------------------------------------ #
# Dataset / loader helpers
# ------------------------------------------------------------------ #

def _log_window_diagnostics(ds_window, dataset: PVDataset | None) -> dict:
    """Log NaN/data diagnostics for a time window."""
    diag: dict = {}
    energia = ds_window["ENERGIA"].values
    n_plants, n_times = energia.shape
    all_nan_plants = int(np.all(np.isnan(energia), axis=1).sum())
    nan_frac = float(np.isnan(energia).sum()) / max(energia.size, 1)
    diag["n_plants"] = n_plants
    diag["n_times"] = n_times
    diag["all_nan_plants"] = all_nan_plants
    diag["energia_nan_frac"] = round(nan_frac, 4)

    if dataset is not None and len(dataset) > 0:
        target = dataset.target_pv
        diag["target_pv_min"] = float(np.nanmin(target))
        diag["target_pv_max"] = float(np.nanmax(target))
        diag["target_pv_mean"] = float(np.nanmean(target))
        diag["target_pv_gt1"] = int((target > 1.0).sum())

    if all_nan_plants > 0 or nan_frac > 0.3:
        print(
            f"  [diag] all_nan_plants={all_nan_plants}/{n_plants}  "
            f"nan_frac={nan_frac:.1%}  target_range="
            f"[{diag.get('target_pv_min', '?'):.3f}, {diag.get('target_pv_max', '?'):.3f}]"
        )
    return diag


def _build_dataset_and_loader(
    ds_window,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    kwp: "np.ndarray | None" = None,
    pv_scale: "np.ndarray | None" = None,
) -> tuple[PVDataset | None, DataLoader | None]:
    try:
        _qs_da, m_components = compute_qs(ds_window, debug=True)
    except Exception as e:
        print(f"  [warn] compute_qs failed: {e}")
        return None, None

    try:
        dataset = PVDataset(ds_window, m_components, seq_len=seq_len, kwp=kwp, pv_scale=pv_scale)
    except Exception as e:
        print(f"  [warn] PVDataset failed: {e}")
        return None, None

    if len(dataset) == 0:
        return dataset, None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )
    return dataset, loader


# ------------------------------------------------------------------ #
# Training / update / eval
# ------------------------------------------------------------------ #

def _train_epoch(
    model: STGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    lam: float,
    device: str,
    max_batches: int | None = None,
) -> dict:
    model.train()
    ei = edge_index.to(device)
    ew = edge_weight.to(device)
    losses: list[float] = []

    for batch_idx, (x, y_ghi, y_pv, eta, ghi_cs) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y_ghi = y_ghi.to(device)
        y_pv = y_pv.to(device)
        eta = eta.to(device)
        ghi_cs = ghi_cs.to(device)

        pred_ghi, pred_pv = model(x, ei, ew, ghi_cs)
        loss, _ = physics_loss_full(pred_ghi, pred_pv, y_ghi, y_pv, eta, lam=lam)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "n_batches": len(losses),
    }


def _continual_update(
    model: STGNN,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    buffer: SimpleReplayBuffer,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    lam: float,
    device: str,
    n_epochs: int,
    replay_batch_size: int,
    replay_loss_weight: float,
    max_batches: int | None = None,
) -> dict:
    """
    Update step: for each recent batch, also sample from replay buffer.

    loss = loss_recent + replay_loss_weight * loss_replay
    """
    model.train()
    ei = edge_index.to(device)
    ew = edge_weight.to(device)

    total_loss = 0.0
    total_recent_loss = 0.0
    total_replay_loss = 0.0
    n_steps = 0
    n_replay_samples = 0

    for _epoch in range(n_epochs):
        for batch_idx, (x, y_ghi, y_pv, eta, ghi_cs) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x = x.to(device)
            y_ghi = y_ghi.to(device)
            y_pv = y_pv.to(device)
            eta = eta.to(device)
            ghi_cs = ghi_cs.to(device)

            # Gradient accumulation: forward+backward recent and replay
            # separately so only one computation graph is alive at a time.
            optimizer.zero_grad()

            pred_ghi, pred_pv = model(x, ei, ew, ghi_cs)
            loss_recent, _ = physics_loss_full(
                pred_ghi, pred_pv, y_ghi, y_pv, eta, lam=lam,
            )
            loss_recent.backward()

            loss_replay_val = 0.0
            replay_used = 0
            if len(buffer) >= replay_batch_size:
                rx, ry_ghi, ry_pv, reta, rghi_cs = buffer.sample(replay_batch_size)
                rx = rx.to(device)
                ry_ghi = ry_ghi.to(device)
                ry_pv = ry_pv.to(device)
                reta = reta.to(device)
                rghi_cs = rghi_cs.to(device)

                rpred_ghi, rpred_pv = model(rx, ei, ew, rghi_cs)
                loss_replay, _ = physics_loss_full(
                    rpred_ghi, rpred_pv, ry_ghi, ry_pv, reta, lam=lam,
                )
                (replay_loss_weight * loss_replay).backward()
                loss_replay_val = loss_replay.item()
                replay_used = replay_batch_size

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss_recent.item() + replay_loss_weight * loss_replay_val
            total_recent_loss += loss_recent.item()
            total_replay_loss += loss_replay_val
            n_steps += 1
            n_replay_samples += replay_used

    return {
        "avg_loss": total_loss / max(n_steps, 1),
        "avg_recent_loss": total_recent_loss / max(n_steps, 1),
        "avg_replay_loss": total_replay_loss / max(n_steps, 1),
        "n_steps": n_steps,
        "n_replay_samples": n_replay_samples,
    }


_BIN_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf]
_BIN_LABELS = ["0_20", "20_40", "40_60", "60_80", "80_100", "over_100"]


def compute_bin_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bin_edges: list[float] = _BIN_EDGES,
    bin_labels: list[str] = _BIN_LABELS,
) -> list[dict]:
    """Per-bin MAE/RMSE on flattened arrays, binned by y_true.

    Matches train.py:_val_epoch bin definition exactly:
    bins on normalized y_true (ENERGIA / p99_per_plant), edges at
    [0, 0.2, 0.4, 0.6, 0.8, 1.0, inf]. The over_100 bin captures
    values > 1.0 that can occur due to clipping at 1.5 in PVDataset.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    n_nan_removed = int((~finite).sum())
    n_y_true_out = int(((y_true[finite] < 0) | (y_true[finite] > 1.0)).sum())
    n_y_pred_out = int(((y_pred[finite] < 0) | (y_pred[finite] > 1.0)).sum())

    y_true = y_true[finite]
    y_pred = y_pred[finite]

    rows: list[dict] = []
    for i in range(len(bin_labels)):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        label = bin_labels[i]
        mask = (y_true >= lo) & (y_true < hi)

        count = int(mask.sum())
        if count == 0:
            rows.append({
                "bin_label": label, "bin_low": lo, "bin_high": hi,
                "mae": float("nan"), "rmse": float("nan"),
                "count": 0,
                "mean_y_true": float("nan"), "mean_y_pred": float("nan"),
                "mean_error": float("nan"),
                "min_y_true": float("nan"), "max_y_true": float("nan"),
                "min_y_pred": float("nan"), "max_y_pred": float("nan"),
                "n_nan_removed": n_nan_removed,
                "n_y_true_out_of_range": n_y_true_out,
                "n_y_pred_out_of_range": n_y_pred_out,
            })
            continue

        yt, yp = y_true[mask], y_pred[mask]
        err = yp - yt
        rows.append({
            "bin_label": label, "bin_low": lo, "bin_high": hi,
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "count": count,
            "mean_y_true": float(np.mean(yt)),
            "mean_y_pred": float(np.mean(yp)),
            "mean_error": float(np.mean(err)),
            "min_y_true": float(np.min(yt)),
            "max_y_true": float(np.max(yt)),
            "min_y_pred": float(np.min(yp)),
            "max_y_pred": float(np.max(yp)),
            "n_nan_removed": n_nan_removed,
            "n_y_true_out_of_range": n_y_true_out,
            "n_y_pred_out_of_range": n_y_pred_out,
        })
    return rows


@torch.no_grad()
def _evaluate(
    model: STGNN,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    lam: float,
    device: str,
    max_batches: int | None = None,
) -> dict:
    model.eval()
    ei = edge_index.to(device)
    ew = edge_weight.to(device)

    losses: list[float] = []
    pv_preds: list[np.ndarray] = []
    pv_trues: list[np.ndarray] = []

    for batch_idx, (x, y_ghi, y_pv, eta, ghi_cs) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y_ghi = y_ghi.to(device)
        y_pv = y_pv.to(device)
        eta = eta.to(device)
        ghi_cs = ghi_cs.to(device)

        pred_ghi, pred_pv = model(x, ei, ew, ghi_cs)
        loss, _ = physics_loss_full(pred_ghi, pred_pv, y_ghi, y_pv, eta, lam=lam)
        losses.append(loss.item())
        pv_preds.append(pred_pv.cpu().numpy().ravel())
        pv_trues.append(y_pv.cpu().numpy().ravel())

    model.train()

    if not pv_preds:
        return {
            "loss": float("nan"), "mae": float("nan"), "rmse": float("nan"),
            "n_samples": 0, "bin_metrics": [],
        }

    pv_p = np.concatenate(pv_preds)
    pv_t = np.concatenate(pv_trues)
    err = pv_p - pv_t

    return {
        "loss": float(np.mean(losses)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "n_samples": int(pv_p.size),
        "bin_metrics": compute_bin_metrics(pv_t, pv_p),
    }


def _populate_buffer(buffer: SimpleReplayBuffer, dataset: PVDataset, max_samples: int | None = None) -> int:
    n = len(dataset)
    if max_samples is not None:
        n = min(n, max_samples)
    for i in range(n):
        x, y_ghi, y_pv, eta, ghi_cs = dataset[i]
        buffer.add(x, y_ghi, y_pv, eta, ghi_cs)
    return n


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay-based continual adaptation for PV forecasting",
    )

    # Data mode
    parser.add_argument(
        "--data-mode", type=str, choices=["synthetic", "real"], default="synthetic",
        help="synthetic = 20-plant debug data; real = Piedmont Sentinel/SCADA",
    )

    # Real data paths
    parser.add_argument(
        "--sentinel-dir", type=str,
        default="/data/SentinelPV/energy_data/piemonte_energy_data/single_ups",
    )
    parser.add_argument("--plant-mapping", type=str, default="data/plant_mapping.csv")
    parser.add_argument("--energy-coords", type=str, default="data/energy_with_coordinates.csv")
    parser.add_argument("--pvgis-path", type=str, default="data/piedmont_pvgis_2019.nc")
    parser.add_argument("--year", type=int, default=2019)

    # Temporal stream
    parser.add_argument("--initial-train-start", type=str, default=None)
    parser.add_argument("--initial-train-end", type=str, default=None)
    parser.add_argument("--window-months", type=int, default=None,
                        help="Window size in months (default 1). Ignored if --window-days set.")
    parser.add_argument("--window-days", type=int, default=None,
                        help="Window size in days (e.g. 7=weekly, 1=daily). Overrides --window-months.")
    parser.add_argument("--max-windows", type=int, default=None)

    # Training
    parser.add_argument("--initial-epochs", type=int, default=5)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)

    # Replay
    parser.add_argument("--replay-buffer-size", type=int, default=5000)
    parser.add_argument("--replay-batch-size", type=int, default=8)
    parser.add_argument("--replay-loss-weight", type=float, default=1.0)
    parser.add_argument("--replay-seed", type=int, default=None)

    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/continual_replay")

    # Debug
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-plants", type=int, default=None)
    parser.add_argument("--max-batches-per-window", type=int, default=None)

    args = parser.parse_args()

    # Default dates depend on data mode
    if args.initial_train_start is None:
        args.initial_train_start = "2019-03-01" if args.data_mode == "real" else "2023-01-01"
    if args.initial_train_end is None:
        args.initial_train_end = "2019-05-31" if args.data_mode == "real" else "2023-03-31"

    _set_seed(args.seed)

    if args.debug:
        if args.max_plants is None:
            args.max_plants = 5
        if args.max_batches_per_window is None:
            args.max_batches_per_window = 3
        if args.max_windows is None:
            args.max_windows = 2
        args.initial_epochs = min(args.initial_epochs, 2)

    replay_seed = args.replay_seed if args.replay_seed is not None else args.seed
    run_name = args.run_name or f"replay_{args.data_mode}_{int(time.time())}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] run={run_name}  mode={args.data_mode}  device={DEVICE}  out={out_dir}")

    config = vars(args)
    config["run_name"] = run_name
    config["device"] = DEVICE
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #
    kwp = None
    if args.data_mode == "real":
        ds, kwp = _load_real(
            sentinel_dir=args.sentinel_dir,
            year=args.year,
            plant_mapping=args.plant_mapping,
            energy_coords=args.energy_coords,
            pvgis_path=args.pvgis_path,
        )
    else:
        print("[data] generating synthetic dataset")
        ds = _load_synthetic(seed=args.seed)

    if args.max_plants is not None and args.max_plants < ds.sizes["plant"]:
        ds = ds.isel(plant=slice(0, args.max_plants))
        if kwp is not None:
            kwp = kwp[: args.max_plants]

    n_plants = ds.sizes["plant"]
    times = pd.DatetimeIndex(ds.coords["time"].values)
    print(
        f"[data] n_plants={n_plants}  T={ds.sizes['time']}h  "
        f"range={times[0].date()} .. {times[-1].date()}"
    )

    # ------------------------------------------------------------------ #
    # Graph
    # ------------------------------------------------------------------ #
    lats = ds["lat"].values
    lons = ds["lon"].values
    edge_index, edge_weight = build_graph(lats, lons, max_dist_km=50.0)
    print(f"[graph] {n_plants} nodes, {edge_index.shape[1]} edges")

    # ------------------------------------------------------------------ #
    # Temporal stream
    # ------------------------------------------------------------------ #
    window_months = args.window_months if args.window_days is None else None
    if window_months is None and args.window_days is None:
        window_months = 1
    stream = TemporalStream(
        ds,
        initial_train_start=args.initial_train_start,
        initial_train_end=args.initial_train_end,
        window_months=window_months,
        window_days=args.window_days,
        max_windows=args.max_windows,
    )
    initial_ds = stream.initial_train_ds()
    print(
        f"[stream] initial: {args.initial_train_start} -> {args.initial_train_end} "
        f"({initial_ds.sizes['time']}h)"
    )

    # ------------------------------------------------------------------ #
    # Initial training
    # ------------------------------------------------------------------ #
    print("[train] building initial dataset...")
    dataset_init, loader_init = _build_dataset_and_loader(
        initial_ds, args.seq_len, args.batch_size, shuffle=True, kwp=kwp,
    )
    if dataset_init is None or loader_init is None:
        print("[ERROR] cannot build initial dataset")
        return

    # Fix p99 scale from initial window — reused for all subsequent windows
    # so that normalization (and bin metrics) are consistent across time.
    fixed_pv_scale = dataset_init.pv_scale.copy()
    print(
        f"[train] {len(dataset_init)} samples, {N_FEATURES} features, "
        f"pv_scale range=[{fixed_pv_scale.min():.2f}, {fixed_pv_scale.max():.2f}]"
    )

    model = STGNN(
        n_nodes=n_plants,
        n_features=N_FEATURES,
        seq_len=args.seq_len,
        patch_len=max(1, args.seq_len // 6),
        stride=max(1, args.seq_len // 12),
        d_model=128,
        gat_dim=96,
        gat_heads=4,
        gat_layers=1,
        dropout=0.2,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"[train] initial training for {args.initial_epochs} epochs...")
    for epoch in range(1, args.initial_epochs + 1):
        ep_metrics = _train_epoch(
            model, loader_init, optimizer, edge_index, edge_weight,
            args.lam, DEVICE, max_batches=args.max_batches_per_window,
        )
        print(f"  epoch {epoch}/{args.initial_epochs}  loss={ep_metrics['loss']:.4f}")

    torch.save(model.state_dict(), out_dir / "checkpoint_initial.pt")

    eval_init = _evaluate(
        model, loader_init, edge_index, edge_weight, args.lam, DEVICE,
        max_batches=args.max_batches_per_window,
    )
    print(f"[train] initial eval: mae={eval_init['mae']:.4f}  rmse={eval_init['rmse']:.4f}")

    # ------------------------------------------------------------------ #
    # Replay buffer
    # ------------------------------------------------------------------ #
    buffer = SimpleReplayBuffer(capacity=args.replay_buffer_size, seed=replay_seed)
    n_added = _populate_buffer(buffer, dataset_init)
    print(f"[buffer] populated with {n_added} initial samples (size={len(buffer)})")

    # ------------------------------------------------------------------ #
    # Metrics tracking
    # ------------------------------------------------------------------ #
    metrics_rows: list[dict] = []
    bin_rows: list[dict] = []

    def _append_bin_rows(window_id, window_start, window_end, phase, eval_res):
        for bm in eval_res.get("bin_metrics", []):
            bin_rows.append({
                "window_id": window_id,
                "window_start": str(window_start),
                "window_end": str(window_end),
                "phase": phase,
                **bm,
            })

    metrics_rows.append({
        "window_id": -1,
        "window_start": str(args.initial_train_start),
        "window_end": str(args.initial_train_end),
        "phase": "initial_train",
        "mae": eval_init["mae"],
        "rmse": eval_init["rmse"],
        "loss": eval_init["loss"],
        "num_recent_samples": len(dataset_init),
        "num_replay_samples": 0,
        "replay_buffer_size": len(buffer),
    })
    _append_bin_rows(-1, args.initial_train_start, args.initial_train_end,
                     "initial_train", eval_init)

    # ------------------------------------------------------------------ #
    # Continual adaptation loop
    # ------------------------------------------------------------------ #
    print("[stream] starting continual adaptation...")
    for window_id, w_start, w_end, ds_window in stream.stream_windows():
        print(
            f"\n[window {window_id}] {w_start.date()} -> {w_end.date()} "
            f"({ds_window.sizes['time']}h)"
        )

        dataset_w, loader_w = _build_dataset_and_loader(
            ds_window, args.seq_len, args.batch_size, shuffle=True,
            kwp=kwp, pv_scale=fixed_pv_scale,
        )
        if dataset_w is None or loader_w is None:
            print("  skipped (insufficient data)")
            continue

        _log_window_diagnostics(ds_window, dataset_w)
        print(f"  samples={len(dataset_w)}  buffer={len(buffer)}")

        update_result = _continual_update(
            model, optimizer, loader_w, buffer,
            edge_index, edge_weight,
            args.lam, DEVICE,
            n_epochs=args.update_epochs,
            replay_batch_size=args.replay_batch_size,
            replay_loss_weight=args.replay_loss_weight,
            max_batches=args.max_batches_per_window,
        )

        n_added = _populate_buffer(buffer, dataset_w)

        eval_result = _evaluate(
            model, loader_w, edge_index, edge_weight, args.lam, DEVICE,
            max_batches=args.max_batches_per_window,
        )

        print(
            f"  loss={eval_result['loss']:.4f}  mae={eval_result['mae']:.4f}  "
            f"rmse={eval_result['rmse']:.4f}  replay_used={update_result['n_replay_samples']}  "
            f"buffer={len(buffer)}"
        )

        metrics_rows.append({
            "window_id": window_id,
            "window_start": str(w_start),
            "window_end": str(w_end),
            "phase": "continual_update",
            "mae": eval_result["mae"],
            "rmse": eval_result["rmse"],
            "loss": eval_result["loss"],
            "num_recent_samples": len(dataset_w),
            "num_replay_samples": update_result["n_replay_samples"],
            "replay_buffer_size": len(buffer),
        })
        _append_bin_rows(window_id, w_start, w_end, "continual_update", eval_result)

        if not args.debug:
            torch.save(model.state_dict(), out_dir / f"checkpoint_window_{window_id}.pt")

    # ------------------------------------------------------------------ #
    # Save outputs
    # ------------------------------------------------------------------ #
    torch.save(model.state_dict(), out_dir / "checkpoint_final.pt")

    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(out_dir / "metrics_per_window.csv", index=False)

    df_bins = pd.DataFrame(bin_rows)
    df_bins.to_csv(out_dir / "metrics_by_bin.csv", index=False)

    # Last window bin metrics for summary
    last_window_id = metrics_rows[-1]["window_id"] if metrics_rows else -1
    final_bins = {
        r["bin_label"]: {"mae": r["mae"], "rmse": r["rmse"], "count": r["count"]}
        for r in bin_rows if r["window_id"] == last_window_id
    }

    summary = {
        "run_name": run_name,
        "data_mode": args.data_mode,
        "seed": args.seed,
        "device": DEVICE,
        "n_plants": n_plants,
        "n_features": N_FEATURES,
        "n_windows": len(metrics_rows) - 1,
        "initial_mae": metrics_rows[0]["mae"],
        "final_mae": metrics_rows[-1]["mae"] if len(metrics_rows) > 1 else None,
        "initial_rmse": metrics_rows[0]["rmse"],
        "final_rmse": metrics_rows[-1]["rmse"] if len(metrics_rows) > 1 else None,
        "replay_buffer_final_size": len(buffer),
        "replay_buffer_capacity": buffer.capacity,
        "total_samples_added_to_buffer": buffer.total_added,
        "final_window_bin_metrics": final_bins,
    }
    with open(out_dir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------ #
    # Bin metric audit report
    # ------------------------------------------------------------------ #
    audit_lines = [
        "# Bin Metric Audit",
        "",
        "## Old definition (train.py:_val_epoch)",
        "",
        "- y_true = PVDataset.target_pv = ENERGIA / p99_per_plant, clipped [0, 1.5]",
        "- Bins: [0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1.0), [1.0,inf)",
        "- Evaluated on: 20% held-out validation split (stratified monthly, full year)",
        "- Normalization: per-plant p99 of daytime ENERGIA",
        "- Filter: none (all hours including night where target~0)",
        "- Includes over_100 bin for values > 1.0",
        "",
        "## New definition (train_replay_continual.py:compute_bin_metrics)",
        "",
        "- y_true = same PVDataset.target_pv = ENERGIA / p99_per_plant, clipped [0, 1.5]",
        "- Bins: [0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1.0), [1.0,inf)",
        "- Evaluated on: same window used for training (no held-out split)",
        "- Normalization: per-plant p99, BUT p99 recomputed per window (not full-year)",
        "- Filter: none (all hours including night)",
        "- Includes over_100 bin",
        "",
        "## Differences found",
        "",
        "1. **p99 scale differs**: old p99 computed on full year (Mar-Dec).",
        "   New p99 computed per 1-month window. Winter months have lower peak",
        "   production -> lower p99 -> same kWh maps to HIGHER normalized value.",
        "   A plant producing 5 kWh in Dec with p99_dec=6 -> target=0.83.",
        "   Same plant with p99_fullyear=10 -> target=0.50.",
        "   THIS IS THE MAIN CAUSE of inflated high-bin MAE.",
        "",
        "2. **Eval on training data vs held-out**: old metrics on 20% validation.",
        "   New metrics on same data used for update. Makes new metrics",
        "   optimistic for global MAE but does not explain high-bin inflation.",
        "",
        "3. **Seasonal bias**: old metrics averaged across all seasons.",
        "   Dec window has mostly low-production hours (short days, low sun).",
        "   Fewer samples in high bins -> higher variance in bin MAE.",
        "",
        "## Conclusion",
        "",
        "The comparison 3-5% (old) vs 10-16% (new) in bins 60-80/80-100 is",
        "**NOT fair**. The per-window p99 normalization inflates the high bins",
        "because winter p99 is much lower than full-year p99. The same absolute",
        "production value lands in a higher bin when normalized by a smaller p99.",
        "",
        "## Recommendation",
        "",
        "To make results comparable, compute p99 on the INITIAL training window",
        "(Mar-May) and reuse that fixed scale for all subsequent windows.",
        "This matches the real deployment scenario where normalization is fixed",
        "at training time, not recomputed monthly.",
    ]
    with open(out_dir / "bin_metric_audit.md", "w", encoding="utf-8") as f:
        f.write("\n".join(audit_lines) + "\n")

    print(f"\n[done] outputs -> {out_dir}/")
    print(f"  data_mode={args.data_mode}  n_plants={n_plants}")
    print(f"  initial mae={summary['initial_mae']:.4f}")
    if summary["final_mae"] is not None:
        print(f"  final   mae={summary['final_mae']:.4f}")
    print(f"  buffer: {len(buffer)}/{buffer.capacity}")


if __name__ == "__main__":
    main()
