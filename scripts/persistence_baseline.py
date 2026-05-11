"""
Persistence baseline for PhysiQ-PV.

Computes y_pred(t) = y_true(t-1) on the real Piedmont 2019 dataset and
reports global + per-band MAE/RMSE/bias against the production-curve
bins used in the W&B training dashboard.

No model, no training. Pure naive forecast for VSTF benchmarking.

Usage:
    python scripts/persistence_baseline.py
    python scripts/persistence_baseline.py --wandb
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Repo root on sys.path so `main` and `physiq_pv` are importable when running from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from physiq_pv.data.sentinel_hourly_loader import (  # noqa: E402
    load_sentinel_hourly,
    merge_with_weather,
)
from main import _normalize_dataset  # noqa: E402


# Production-curve bins keyed on normalized true PV (matches W&B training metrics).
BANDS: list[tuple[str, float, float]] = [
    ("base campana: 0-20%",    0.0, 0.2),
    ("bassa-media: 20-40%",    0.2, 0.4),
    ("media: 40-60%",          0.4, 0.6),
    ("medio-alta: 60-80%",     0.6, 0.8),
    ("picco: 80-100%",         0.8, 1.0),
    ("oltre p99: >100%",       1.0, np.inf),
]


def compute_pv_scale(energia: np.ndarray, daytime: np.ndarray) -> np.ndarray:
    """Per-plant p99 of daytime non-zero ENERGIA. Mirrors PVDataset normalization."""
    n_plants = energia.shape[1]
    pv_scale = np.ones(n_plants, dtype=np.float64)
    for p in range(n_plants):
        e = energia[daytime[:, p], p]
        e = e[e > 0]
        if len(e) > 10:
            pv_scale[p] = float(np.percentile(e, 99)) + 1e-6
    return pv_scale


def band_metrics(true_vals: np.ndarray, pred_vals: np.ndarray) -> pd.DataFrame:
    """Build per-band table on the production-curve bins (uses true_vals for binning)."""
    rows: list[dict] = []
    n_total = int(true_vals.size)
    for label, lo, hi in BANDS:
        band_mask = (true_vals >= lo) & (true_vals < hi)
        n = int(band_mask.sum())
        if n == 0:
            rows.append({
                "fascia":      label,
                "n":           0,
                "share_%":     0.0,
                "MAE":         float("nan"),
                "RMSE":        float("nan"),
                "bias":        float("nan"),
                "actual_mean": float("nan"),
                "pred_mean":   float("nan"),
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
    print("[1] Loading dataset (Sentinel hourly + weather)...")
    ds = load_sentinel_hourly(
        sentinel_dir=args.sentinel_dir,
        year=args.year,
        plant_mapping_path=args.plant_mapping,
        energy_coords_path=args.energy_coords,
    )
    ds = merge_with_weather(ds, pvgis_path=args.pvgis_path)
    ds = _normalize_dataset(ds)
    print(f"    OK {ds.sizes['plant']} plants x {ds.sizes['time']} timesteps")

    # (T, N) layouts to mirror PVDataset internals.
    energia = np.nan_to_num(ds["ENERGIA"].values.T, nan=0.0)
    solar_kwm2 = np.clip(ds["solar_irradiance_poa"].values.T / 1000.0, 0.0, None)
    daytime = solar_kwm2 > 0.03

    print("[2] Computing per-plant pv_scale (p99 daytime ENERGIA)...")
    pv_scale = compute_pv_scale(energia, daytime)
    y_norm = np.clip(energia / pv_scale[None, :], 0.0, 1.5).astype(np.float32)

    print("[3] Persistence: y_pred(t) = y_true(t-1) ...")
    # Predictions for t = 1..T-1 use targets at t-1 = 0..T-2.
    y_pred = y_norm[:-1, :]
    y_true = y_norm[1:, :]
    day_eval = daytime[1:, :]

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    eval_mask = finite & day_eval

    true_flat = y_true[eval_mask].ravel()
    pred_flat = y_pred[eval_mask].ravel()

    err = pred_flat - true_flat
    n = int(true_flat.size)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    print("\n=== Persistence baseline - global (daytime, PV normalizzato) ===")
    print(f"  n          = {n:,}")
    print(f"  MAE        = {mae:.4f}")
    print(f"  RMSE       = {rmse:.4f}")
    print(f"  bias       = {bias:+.4f}")

    df_bands = band_metrics(true_flat, pred_flat)
    print("\n=== Persistence baseline - MAE/RMSE per fasce della campana di produzione ===")
    with pd.option_context("display.max_colwidth", 80, "display.width", 200):
        print(df_bands.to_string(index=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_json_path = out_dir / "metrics_global.json"
    band_csv_path = out_dir / "metrics_by_production_band.csv"

    metrics_payload = {
        "method":         "y_pred(t) = y_true(t-1)",
        "year":           args.year,
        "n_plants":       int(ds.sizes["plant"]),
        "n_timesteps":    int(ds.sizes["time"]),
        "n_samples":      n,
        "daytime_mask":   "solar_irradiance_poa / 1000 > 0.03",
        "target_norm":    "clip(ENERGIA / pv_scale, 0, 1.5); pv_scale = p99 daytime per plant",
        "global_mae":     mae,
        "global_rmse":    rmse,
        "global_bias":    bias,
        "bands": [
            {
                "label": b[0],
                "lo":    b[1],
                "hi":    (None if np.isinf(b[2]) else b[2]),
            }
            for b in BANDS
        ],
    }
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    df_bands.to_csv(band_csv_path, index=False)

    print(f"\n  -> {metrics_json_path}")
    print(f"  -> {band_csv_path}")

    if args.wandb:
        try:
            import wandb
        except ImportError:
            print("  W&B requested but `wandb` not installed -- skipping.")
            return

        wb_run = wandb.init(
            entity="albertopedalino-politecnico-di-torino",
            project="PhysiQ-PV",
            name="persistence-baseline-t-minus-1",
            job_type="baseline",
            tags=["persistence"],
            config={
                "method":       "y_pred(t) = y_true(t-1)",
                "year":         args.year,
                "n_plants":     int(ds.sizes["plant"]),
                "n_timesteps":  int(ds.sizes["time"]),
                "n_samples":    n,
                "daytime_mask": "solar_irradiance_poa/1000 > 0.03",
            },
        )

        wb_run.summary["persistence/global_mae"]  = mae
        wb_run.summary["persistence/global_rmse"] = rmse
        wb_run.summary["persistence/global_bias"] = bias
        wb_run.summary["persistence/n_samples"]   = n

        for _, row in df_bands.iterrows():
            tag = (
                row["fascia"].split(":")[0]
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )
            wb_run.summary[f"persistence/mae_{tag}"]   = float(row["MAE"])
            wb_run.summary[f"persistence/rmse_{tag}"]  = float(row["RMSE"])
            wb_run.summary[f"persistence/bias_{tag}"]  = float(row["bias"])
            wb_run.summary[f"persistence/n_{tag}"]     = int(row["n"])

        wb_table = wandb.Table(dataframe=df_bands)
        wb_run.log({"persistence/by_band": wb_table})
        wb_run.finish()
        print("  W&B run logged.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persistence baseline for PhysiQ-PV.")
    p.add_argument("--sentinel-dir",  default="/data/SentinelPV/energy_data/piemonte_energy_data/single_ups")
    p.add_argument("--year",          type=int, default=2019)
    p.add_argument("--plant-mapping", default="data/plant_mapping.csv")
    p.add_argument("--energy-coords", default="data/energy_with_coordinates.csv")
    p.add_argument("--pvgis-path",    default="data/piedmont_pvgis_2019.nc")
    p.add_argument("--out-dir",       default="checkpoints/persistence_baseline")
    p.add_argument("--wandb",         action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
