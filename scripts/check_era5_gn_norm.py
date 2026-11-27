"""How much G_n changes if G is the L2 norm of the STGAN paper instead of the mean squared error of the authors' code.

Within one cell the 3x3 patch always has the same cells, so the norm is sqrt(G) up to a constant factor that the
standardisation removes. G_n is the climatological z-score of G: the thesis' own function (era5.seasonality.
climatology_zscores: leave one year out, location x day of year +-15 days x UTC hour, values excluded by the normal
masks left out of the climatology) is applied once to G and once to sqrt(G), on the validation years. Then, for both:
the flags (highest `--top-percent` of the finite values) and how many are shared (share of the flags of the thesis'
G_n also flagged with the norm, and Jaccard index), and the mean spatial Spearman correlation of the two G_n over a
sample of timestamps. Only the score changes: the model, trained with the mean squared error, is the same.

    python scripts/check_era5_gn_norm.py --run outputs/stgan_optuna27_validation \
        --masks outputs/era5_sweep_1999/normal_masks --output-dir outputs/check_gn_norm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from physiq_pv.anomaly_detection.stgan.normal_masks import load_normal_masks
from physiq_pv.era5.seasonality import climatology_zscores
from scripts.check_era5_score_correlation import open_component
from scripts.check_era5_yearly_share_and_skill import mask_rows

CHUNK = 256


def write_sqrt(scores, path, chunk=CHUNK):
    """sqrt of the scores (G >= 0) into a float32 .npy memmap of the same shape."""
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=scores.shape)
    for start in range(0, len(scores), chunk):
        block = np.asarray(scores[start:start + chunk], dtype=np.float64)
        out[start:start + chunk] = np.sqrt(np.clip(block, 0, None))
    out.flush()
    return out


def threshold(values, top_percent, chunk=CHUNK, stride=1):
    """The (100 - top_percent) percentile of the finite values, one timestamp every `stride`."""
    rows = np.asarray(values[::stride], dtype=np.float32)
    finite = rows[np.isfinite(rows)]
    return float(np.percentile(finite, 100 - top_percent))


def shared_flags(first, second, limit_first, limit_second, chunk=CHUNK):
    """Counts of flags of `first`, of `second` and of both."""
    n_first = n_second = n_both = 0
    for start in range(0, len(first), chunk):
        a = np.asarray(first[start:start + chunk]) >= limit_first
        b = np.asarray(second[start:start + chunk]) >= limit_second
        n_first += int(a.sum())
        n_second += int(b.sum())
        n_both += int((a & b).sum())
    return n_first, n_second, n_both


def mean_spatial_spearman(first, second, rows):
    out = []
    for row in rows:
        a, b = np.asarray(first[row], dtype=np.float64), np.asarray(second[row], dtype=np.float64)
        keep = np.isfinite(a) & np.isfinite(b)
        if keep.sum() > 2:
            out.append(np.corrcoef(rankdata(a[keep]), rankdata(b[keep]))[0, 1])
    return float(np.mean(out)), float(np.min(out))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="Scored validation run (generator component)")
    parser.add_argument("--masks", type=Path, required=True, help="Normal masks directory of the validation")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the two G_n (about 1.3 GB)")
    parser.add_argument("--top-percent", type=float, default=5.0)
    parser.add_argument("--window-days", type=int, default=15)
    parser.add_argument("--sample-timestamps", type=int, default=500)
    args = parser.parse_args(argv)
    manifest = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
    times = pd.DatetimeIndex(np.load(args.run / "test_timestamps.npy"))
    excluded = mask_rows(np.asarray(load_normal_masks(args.masks).validation_mask, dtype=bool), len(times))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    g, maps = open_component(args.run, manifest, "generator")
    try:
        if np.shape(excluded) != g.shape:
            raise ValueError(f"Masks {np.shape(excluded)}, scores {g.shape}")
        root = write_sqrt(g, args.output_dir / "g_sqrt.npy")
        common = dict(excluded=excluded, period="day", window_days=args.window_days)
        climatology_zscores(g, times, args.output_dir / "gn_mse.npy", **common)
        climatology_zscores(root, times, args.output_dir / "gn_norm.npy", **common)
        del root
    finally:
        for array in maps:
            array._mmap.close()
    gn_mse = np.load(args.output_dir / "gn_mse.npy", mmap_mode="r")
    gn_norm = np.load(args.output_dir / "gn_norm.npy", mmap_mode="r")
    limit_mse = threshold(gn_mse, args.top_percent)
    limit_norm = threshold(gn_norm, args.top_percent)
    n_mse, n_norm, n_both = shared_flags(gn_mse, gn_norm, limit_mse, limit_norm)
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(len(times), size=min(args.sample_timestamps, len(times)), replace=False))
    rho_mean, rho_min = mean_spatial_spearman(gn_mse, gn_norm, rows)
    summary = {"run": str(args.run), "timestamps": len(times), "cells": int(gn_mse.shape[1]),
               "top_percent": args.top_percent, "flags_mse": n_mse, "flags_norm": n_norm, "flags_both": n_both,
               "share_of_mse_flags_kept": n_both / n_mse, "jaccard": n_both / (n_mse + n_norm - n_both),
               "spatial_spearman_mean": rho_mean, "spatial_spearman_min": rho_min,
               "sample_timestamps": len(rows)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[gn-norm] {args.run} | {len(times)} timestamps x {gn_mse.shape[1]} cells | top {args.top_percent}%")
    print(f"[gn-norm] flags: G_n (MSE) {n_mse}, G_n (norm) {n_norm}, both {n_both}")
    print(f"[gn-norm] share of the flags of G_n (MSE) kept with the norm: {100 * n_both / n_mse:.1f}% | "
          f"Jaccard {100 * n_both / (n_mse + n_norm - n_both):.1f}%")
    print(f"[gn-norm] spatial Spearman between the two G_n over {len(rows)} timestamps: mean {rho_mean:.4f}, "
          f"min {rho_min:.4f}")
    print(f"[gn-norm] results in {args.output_dir} (delete gn_*.npy and g_sqrt.npy afterwards)")


if __name__ == "__main__":
    main()
