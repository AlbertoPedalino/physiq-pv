"""
One-hour-training sanity check (anti-leakage).

Trains the ST-GNN on a single training-set window for `--train-steps`
iterations, then evaluates on the full validation partition. The
validation MAE/RMSE must be much worse than the production model. If
validation metrics are close to the real model, suspect:
    - target leakage (target value reachable from input features)
    - train/val split overlap
    - feature that encodes future information

This experiment is a sanity check, NOT a tuning run.

Usage:
    python scripts/one_hour_training_sanity.py
    python scripts/one_hour_training_sanity.py --seq-len 1 --train-steps 500 --wandb
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physiq_pv.data.sentinel_hourly_loader import (  # noqa: E402
    load_sentinel_hourly,
    merge_with_weather,
)
from physiq_pv.data.dataset import PVDataset  # noqa: E402
from physiq_pv.data.load_kwp import load_kwp  # noqa: E402
from physiq_pv.data.quality_score import compute_qs  # noqa: E402
from physiq_pv.model.graph_builder import build_graph  # noqa: E402
from physiq_pv.model.physics_loss import physics_loss_full  # noqa: E402
from physiq_pv.model.st_gnn import STGNN  # noqa: E402
from main import _normalize_dataset  # noqa: E402


BANDS: list[tuple[str, float, float]] = [
    ("base campana: 0-20%",    0.0, 0.2),
    ("bassa-media: 20-40%",    0.2, 0.4),
    ("media: 40-60%",          0.4, 0.6),
    ("medio-alta: 60-80%",     0.6, 0.8),
    ("picco: 80-100%",         0.8, 1.0),
    ("oltre p99: >100%",       1.0, np.inf),
]


def split_train_val(times: pd.DatetimeIndex, valid_starts: np.ndarray) -> tuple[list[int], list[int]]:
    """Monthly 80/20 stratified split — same logic as train.py."""
    train_idx: list[int] = []
    val_idx:   list[int] = []
    for month in range(1, 13):
        month_mask = np.where(times[valid_starts].month == month)[0]
        if len(month_mask) == 0:
            continue
        split = int(len(month_mask) * 0.8)
        train_idx.extend(month_mask[:split].tolist())
        val_idx.extend(month_mask[split:].tolist())
    return sorted(train_idx), sorted(val_idx)


def band_metrics(true_vals: np.ndarray, pred_vals: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    n_total = int(true_vals.size)
    for label, lo, hi in BANDS:
        band_mask = (true_vals >= lo) & (true_vals < hi)
        n = int(band_mask.sum())
        if n == 0:
            rows.append({
                "fascia": label, "n": 0, "share_%": 0.0,
                "MAE": float("nan"), "RMSE": float("nan"), "bias": float("nan"),
                "actual_mean": float("nan"), "pred_mean": float("nan"),
            })
            continue
        err = pred_vals[band_mask] - true_vals[band_mask]
        rows.append({
            "fascia":      label,
            "n":           n,
            "share_%":     100.0 * n / n_total if n_total > 0 else 0.0,
            "MAE":         float(np.mean(np.abs(err))),
            "RMSE":        float(np.sqrt(np.mean(err ** 2))),
            "bias":        float(np.mean(err)),
            "actual_mean": float(np.mean(true_vals[band_mask])),
            "pred_mean":   float(np.mean(pred_vals[band_mask])),
        })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    from physiq_pv.data.dataset import N_FEATURES, SEQ_LEN as DEFAULT_SEQ_LEN

    seq_len = args.seq_len
    patch_len = 1 if seq_len == 1 else 4
    stride = 1 if seq_len == 1 else 2
    if patch_len > seq_len:
        raise ValueError(f"patch_len ({patch_len}) cannot exceed seq_len ({seq_len})")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[1] Loading dataset...")
    ds = load_sentinel_hourly(
        sentinel_dir=args.sentinel_dir,
        year=args.year,
        plant_mapping_path=args.plant_mapping,
        energy_coords_path=args.energy_coords,
    )
    ds = merge_with_weather(ds, pvgis_path=args.pvgis_path)
    ds = _normalize_dataset(ds)
    n_plants = int(ds.sizes["plant"])
    print(f"    OK {n_plants} plants x {ds.sizes['time']} timesteps")

    print("[2] Computing QS / m_components...")
    _qs_da, m_components = compute_qs(ds, debug=True)

    kwp = None
    if Path(args.plant_mapping).exists() and Path(args.energy_coords).exists():
        kwp = load_kwp(args.plant_mapping, args.energy_coords, n_plants)

    print(f"[3] Building PVDataset (seq_len={seq_len})...")
    dataset_full = PVDataset(ds, m_components, seq_len=seq_len, kwp=kwp, eta_max=0.98)
    times = pd.DatetimeIndex(ds.coords["time"].values)
    valid_starts = dataset_full.valid_starts
    train_idx, val_idx = split_train_val(times, valid_starts)
    print(f"    Split: {len(train_idx)} train windows, {len(val_idx)} val windows")

    # Build graph (used by STGNN forward + train).
    lats = ds["lat"].values
    lons = ds["lon"].values
    edge_index, edge_weight = build_graph(lats, lons, max_dist_km=10.0)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    # Optional filters on the train sample pool:
    #   --daytime-only restricts to daytime hours (10-15)
    #   --month M restricts to a specific month (1-12)
    train_idx_use = train_idx
    if args.daytime_only or args.month is not None:
        ts_train = times[valid_starts[np.asarray(train_idx)]]
        mask = np.ones(len(train_idx), dtype=bool)
        if args.daytime_only:
            mask &= (ts_train.hour >= 10) & (ts_train.hour <= 15)
        if args.month is not None:
            if not (1 <= args.month <= 12):
                raise ValueError(f"--month must be 1..12, got {args.month}")
            mask &= (ts_train.month == args.month)
        train_idx_use = [train_idx[i] for i in np.where(mask)[0]]
        if not train_idx_use:
            raise RuntimeError("No samples in train_idx after filters.")
        filter_desc = []
        if args.daytime_only:
            filter_desc.append("hours 10-15")
        if args.month is not None:
            filter_desc.append(f"month={args.month}")
        print(f"    Filters: {len(train_idx_use)}/{len(train_idx)} samples ({', '.join(filter_desc)})")

    # Pick a single training sample.
    if args.sample_index is None:
        chosen = train_idx_use[len(train_idx_use) // 2]
    else:
        if not (0 <= args.sample_index < len(train_idx_use)):
            raise IndexError(f"sample-index {args.sample_index} out of range [0, {len(train_idx_use)})")
        chosen = train_idx_use[args.sample_index]
    t_target = int(valid_starts[chosen])
    ts_target = times[t_target]
    print(f"[4] Single training sample: dataset_idx={chosen}, target time={ts_target} (t={t_target})")

    x, y_ghi, y_pv, eta, ghi_cs = dataset_full[chosen]
    x_b      = x.unsqueeze(0).to(device)
    y_ghi_b  = y_ghi.unsqueeze(0).to(device)
    y_pv_b   = y_pv.unsqueeze(0).to(device)
    eta_b    = eta.unsqueeze(0).to(device)
    ghi_cs_b = ghi_cs.unsqueeze(0).to(device)

    print("[5] Initializing model + optimizer...")
    model = STGNN(
        n_nodes=n_plants,
        n_features=N_FEATURES,
        seq_len=seq_len,
        patch_len=patch_len,
        stride=stride,
        d_model=64,
        gat_dim=96,
        gat_heads=4,
        gat_layers=1,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"[6] Training on ONE sample for {args.train_steps} steps...")
    model.train()
    final_loss = float("nan")
    for step in range(args.train_steps):
        pred_ghi, pred_pv = model(x_b, edge_index, edge_weight, ghi_cs_b)
        loss, _parts = physics_loss_full(pred_ghi, pred_pv, y_ghi_b, y_pv_b, eta_b, lam=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if (step + 1) % max(1, args.train_steps // 10) == 0 or step == 0:
            print(f"    step {step+1}/{args.train_steps}  loss={final_loss:.4f}")
    print(f"    Final train loss on the single sample: {final_loss:.4f}")

    print("[7] Evaluating on full validation partition...")
    from torch.utils.data import DataLoader, Subset
    dataset_val = Subset(dataset_full, val_idx)
    loader_val = DataLoader(dataset_val, batch_size=8, shuffle=False, num_workers=0, drop_last=False)

    model.eval()
    pv_pred_all: list[np.ndarray] = []
    pv_true_all: list[np.ndarray] = []
    with torch.no_grad():
        for x_v, _y_ghi_v, y_pv_v, _eta_v, ghi_cs_v in loader_val:
            x_v      = x_v.to(device, non_blocking=True)
            ghi_cs_v = ghi_cs_v.to(device, non_blocking=True)
            pred_ghi_v, pred_pv_v = model(x_v, edge_index, edge_weight, ghi_cs_v)
            pv_pred_all.append(pred_pv_v.cpu().numpy().ravel())
            pv_true_all.append(y_pv_v.numpy().ravel())

    pv_p = np.concatenate(pv_pred_all)
    pv_t = np.concatenate(pv_true_all)
    err = pv_p - pv_t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    n = int(pv_t.size)

    print("\n=== Sanity-check global validation metrics ===")
    print(f"  n_val_samples = {n:,}")
    print(f"  MAE           = {mae:.4f}")
    print(f"  RMSE          = {rmse:.4f}")
    print(f"  bias          = {bias:+.4f}")
    print(f"  final_train_loss_single_sample = {final_loss:.4f}")

    df = band_metrics(pv_t, pv_p)
    print("\n=== Sanity-check val metrics per fasce della campana ===")
    with pd.option_context("display.max_colwidth", 80, "display.width", 200):
        print(df.to_string(index=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "experiment":          "one-hour-training-sanity",
        "purpose":             "anti-leakage sanity check — must underperform real model",
        "seq_len":             seq_len,
        "patch_len":           patch_len,
        "stride":              stride,
        "train_steps":         args.train_steps,
        "train_sample_index":  int(chosen),
        "train_target_time":   str(ts_target),
        "n_val_samples":       n,
        "final_train_loss":    final_loss,
        "global_mae":          mae,
        "global_rmse":         rmse,
        "global_bias":         bias,
        "bands": [
            {"label": b[0], "lo": b[1], "hi": (None if np.isinf(b[2]) else b[2])}
            for b in BANDS
        ],
    }
    config_payload = {
        "seq_len":           seq_len,
        "patch_len":         patch_len,
        "stride":            stride,
        "n_features":        N_FEATURES,
        "default_seq_len_global": DEFAULT_SEQ_LEN,
        "train_steps":       args.train_steps,
        "sample_index":      int(chosen),
        "train_target_time": str(ts_target),
        "train_pv_mean":     float(y_pv.mean().item()),
        "train_pv_max":      float(y_pv.max().item()),
        "train_ghi_cs_mean": float(ghi_cs.mean().item()),
        "device":            device,
        "lr":                1e-3,
        "weight_decay":      1e-4,
        "lam":               0.1,
    }
    with open(out_dir / "metrics_global.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_payload, f, indent=2)
    df.to_csv(out_dir / "metrics_by_production_band.csv", index=False)
    print(f"\n  -> {out_dir/'metrics_global.json'}")
    print(f"  -> {out_dir/'config.json'}")
    print(f"  -> {out_dir/'metrics_by_production_band.csv'}")

    if args.wandb:
        try:
            import wandb
        except ImportError:
            print("  W&B requested but `wandb` not installed -- skipping.")
            return
        month_suffix = f"-m{args.month:02d}" if args.month is not None else ""
        day_suffix   = "-day" if args.daytime_only else ""
        wb_run = wandb.init(
            entity="albertopedalino-politecnico-di-torino",
            project="PhysiQ-PV",
            name=f"one-hour-training-sanity{day_suffix}{month_suffix}",
            job_type="sanity-check",
            tags=["sanity", "one-hour-training", "anti-leakage"]
                 + ([f"month-{args.month:02d}"] if args.month is not None else [])
                 + (["daytime"] if args.daytime_only else []),
            config=config_payload,
        )
        wb_run.summary["sanity/global_mae"]              = mae
        wb_run.summary["sanity/global_rmse"]             = rmse
        wb_run.summary["sanity/global_bias"]             = bias
        wb_run.summary["sanity/n_val_samples"]           = n
        wb_run.summary["sanity/final_train_loss_single"] = final_loss
        wb_table = wandb.Table(dataframe=df)
        wb_run.log({"sanity/by_band": wb_table})
        wb_run.finish()
        print("  W&B run logged.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-hour-training sanity check (anti-leakage).")
    p.add_argument("--seq-len",     type=int, default=1)
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--sample-index", type=int, default=None)
    p.add_argument("--out-dir",     default="checkpoints/one_hour_training_sanity")
    p.add_argument("--wandb",       action="store_true")
    p.add_argument("--daytime-only", action="store_true",
                   help="Restrict train sample pool to hours 10-15 (avoid trivial night samples)")
    p.add_argument("--month", type=int, default=None,
                   help="Restrict train sample pool to a specific calendar month (1-12)")
    p.add_argument("--sentinel-dir",  default="/data/SentinelPV/energy_data/piemonte_energy_data/single_ups")
    p.add_argument("--year",          type=int, default=2019)
    p.add_argument("--plant-mapping", default="data/plant_mapping.csv")
    p.add_argument("--energy-coords", default="data/energy_with_coordinates.csv")
    p.add_argument("--pvgis-path",    default="data/piedmont_pvgis_2019.nc")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
