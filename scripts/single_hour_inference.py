"""
Single-hour inference test for PhysiQ-PV (L=24 model).

Loads a trained ST-GNN checkpoint (default: checkpoints/model.pt) and runs
inference on a single timestamp from the val partition. Reports per-plant
prediction vs actual PV (and GHI) for that single hour.

Useful for: prof's request to feed the model a single hour and inspect
prediction accuracy without aggregate metrics.

Usage:
    python scripts/single_hour_inference.py
    python scripts/single_hour_inference.py --hour-index 3500
    python scripts/single_hour_inference.py --plant-index 42
    python scripts/single_hour_inference.py --checkpoint-dir checkpoints
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physiq_pv.data.sentinel_hourly_loader import (  # noqa: E402
    load_sentinel_hourly,
    merge_with_weather,
)
from physiq_pv.data.dataset import PVDataset  # noqa: E402
from physiq_pv.data.quality_score import compute_qs  # noqa: E402
from physiq_pv.data.load_kwp import load_kwp  # noqa: E402
from physiq_pv.model.st_gnn import STGNN  # noqa: E402
from physiq_pv.model.graph_builder import build_graph  # noqa: E402
from main import _normalize_dataset  # noqa: E402


def load_model(checkpoint_dir: Path, n_plants: int, device: str) -> tuple[STGNN, dict]:
    """Load model + arch config from checkpoint directory."""
    config_path = checkpoint_dir / "model_config.json"
    model_path  = checkpoint_dir / "model.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}")
    with open(config_path) as f:
        cfg = json.load(f)
    model = STGNN(
        n_nodes=cfg.get("n_nodes", n_plants),
        n_features=cfg["n_features"],
        seq_len=cfg["seq_len"],
        patch_len=cfg["patch_len"],
        stride=cfg["stride"],
        d_model=cfg.get("d_model", 64),
        gat_dim=cfg.get("gat_dim", 96),
        gat_heads=cfg.get("gat_heads", 4),
        gat_layers=cfg.get("gat_layers", 1),
        dropout=cfg.get("dropout", 0.0),
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def pick_val_hour(times: pd.DatetimeIndex, valid_starts: np.ndarray, hour_index: int | None) -> int:
    """Pick a (val-partition) sample index. Mirrors monthly 80/20 split from train.py."""
    val_indices: list[int] = []
    for month in range(1, 13):
        month_mask = np.where(times[valid_starts].month == month)[0]
        if len(month_mask) == 0:
            continue
        split = int(len(month_mask) * 0.8)
        val_indices.extend(month_mask[split:].tolist())
    val_indices = sorted(val_indices)
    if not val_indices:
        raise RuntimeError("Empty val partition.")
    if hour_index is None:
        return val_indices[len(val_indices) // 2]
    if not (0 <= hour_index < len(val_indices)):
        raise IndexError(f"hour-index {hour_index} out of range [0, {len(val_indices)})")
    return val_indices[hour_index]


def run(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1] Loading dataset (Sentinel hourly + weather)...")
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
    _qs_da, m_components = compute_qs(ds, debug=False)

    kwp = None
    if Path(args.plant_mapping).exists() and Path(args.energy_coords).exists():
        kwp = load_kwp(args.plant_mapping, args.energy_coords, n_plants)

    print("[3] Loading checkpoint + model...")
    ckpt_dir = Path(args.checkpoint_dir)
    model, cfg = load_model(ckpt_dir, n_plants, device)
    print(f"    Loaded {ckpt_dir/'model.pt'} (seq_len={cfg['seq_len']}, n_features={cfg['n_features']})")

    print("[4] Building PVDataset...")
    dataset = PVDataset(ds, m_components, seq_len=cfg["seq_len"], kwp=kwp, eta_max=0.98)
    lats = ds["lat"].values
    lons = ds["lon"].values
    edge_index, edge_weight = build_graph(lats, lons, max_dist_km=10.0)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)

    times = pd.DatetimeIndex(ds.coords["time"].values)
    idx = pick_val_hour(times, dataset.valid_starts, args.hour_index)
    t = int(dataset.valid_starts[idx])
    ts_target = times[t]
    print(f"[5] Sample: dataset idx={idx}, target time={ts_target} (t={t})")

    x, y_ghi, y_pv, _eta, ghi_cs = dataset[idx]
    x_b      = x.unsqueeze(0).to(device)
    ghi_cs_b = ghi_cs.unsqueeze(0).to(device)

    print("[6] Inference...")
    with torch.no_grad():
        pred_ghi, pred_pv = model(x_b, edge_index, edge_weight, ghi_cs_b)
    pred_pv_np  = pred_pv.squeeze(0).cpu().numpy()
    pred_ghi_np = pred_ghi.squeeze(0).cpu().numpy()
    y_pv_np  = y_pv.numpy()
    y_ghi_np = y_ghi.numpy()
    err_pv = pred_pv_np - y_pv_np
    err_ghi = pred_ghi_np - y_ghi_np

    print("\n=== Single-hour inference summary ===")
    print(f"  timestamp  = {ts_target}")
    print(f"  n_plants   = {n_plants}")
    print(f"  PV  | MAE  = {np.mean(np.abs(err_pv)):.4f}  RMSE = {np.sqrt(np.mean(err_pv ** 2)):.4f}  bias = {np.mean(err_pv):+.4f}")
    print(f"  GHI | MAE  = {np.mean(np.abs(err_ghi)):.4f}  RMSE = {np.sqrt(np.mean(err_ghi ** 2)):.4f}  bias = {np.mean(err_ghi):+.4f}")

    rows = []
    for p in range(n_plants):
        rows.append({
            "plant":       int(p),
            "actual_pv":   float(y_pv_np[p]),
            "pred_pv":     float(pred_pv_np[p]),
            "err_pv":      float(err_pv[p]),
            "actual_ghi":  float(y_ghi_np[p]),
            "pred_ghi":    float(pred_ghi_np[p]),
            "err_ghi":     float(err_ghi[p]),
        })
    df = pd.DataFrame(rows)

    if args.plant_index is not None:
        if not (0 <= args.plant_index < n_plants):
            raise IndexError(f"plant-index {args.plant_index} out of range")
        row = df.iloc[args.plant_index]
        print(f"\n=== Plant {args.plant_index} @ {ts_target} ===")
        print(f"  actual_pv  = {row['actual_pv']:.4f}")
        print(f"  pred_pv    = {row['pred_pv']:.4f}")
        print(f"  err_pv     = {row['err_pv']:+.4f}")
        print(f"  actual_ghi = {row['actual_ghi']:.4f}")
        print(f"  pred_ghi   = {row['pred_ghi']:.4f}")
        print(f"  err_ghi    = {row['err_ghi']:+.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_ts = str(ts_target).replace(" ", "_").replace(":", "-")
    csv_path = out_dir / f"single_hour_{safe_ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  -> {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-hour inference test for PhysiQ-PV.")
    p.add_argument("--checkpoint-dir", default="checkpoints",
                   help="Directory with model.pt + model_config.json (default: L=24 baseline)")
    p.add_argument("--hour-index",   type=int, default=None,
                   help="Index into the val partition (default: midpoint)")
    p.add_argument("--plant-index",  type=int, default=None,
                   help="Optional single plant to inspect")
    p.add_argument("--sentinel-dir", default="/data/SentinelPV/energy_data/piemonte_energy_data/single_ups")
    p.add_argument("--year",         type=int, default=2019)
    p.add_argument("--plant-mapping",default="data/plant_mapping.csv")
    p.add_argument("--energy-coords",default="data/energy_with_coordinates.csv")
    p.add_argument("--pvgis-path",   default="data/piedmont_pvgis_2019.nc")
    p.add_argument("--out-dir",      default="checkpoints/single_hour_inference")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
