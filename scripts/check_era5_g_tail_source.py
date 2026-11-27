"""Where the long tail of G comes from: the variable weights or the squared error?

After the min-max of STGAN's combined score, G is squeezed near zero by its few very high values, and the sum
follows D. This check asks what makes those values: the weights of the variables in the reconstruction error or
the square of the error. Nothing is retrained or re-scored: the scored run saves, for every timestamp and cell,
the squared error of each variable at the target cell (scores/feature_scores.npy, mean over the Monte Carlo
draws, never weighted). G is the mean over the 3x3 patch and over the variables of the weighted squared errors
(model.masked_cell_mean); since the mean over the draws is linear, G is rebuilt exactly from those errors, and the
rebuilt G is compared with the saved one as a check. Four versions of G from the same errors:

- squared, weighted: G of the run;
- squared, weight 1 for every variable;
- absolute, weighted: sqrt of the per-variable mean squared error (the RMS over the draws, not the mean absolute
  error: an approximation) with the square roots of the weights, since w e^2 = (sqrt(w) |e|)^2;
- absolute, weight 1.

For each version, on the validation years (one timestamp every `--stride`): the tail (maximum and 99th percentile
over the median, without the excluded values), the interquartile range of D over that of the version after the
min-max of each (minimum and maximum over the same values), and the spatial Spearman correlation of the min-max sum
with D and with the version (mean over the timestamps, all values, as in the thesis). G of the run is also given
with the run's own min-max extremes (r_min, r_max, d_min, d_max of the metadata). Nothing of the run is written.

    python scripts/check_era5_g_tail_source.py --run outputs/stgan_optuna27_validation \
        --masks outputs/era5_sweep_1999/normal_masks --output outputs/check_g_tail_source.csv
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

from scripts.check_era5_minmax_spread import find_ranges, select_ranges
from scripts.check_era5_score_correlation import open_component
from scripts.check_era5_yearly_share_and_skill import mask_rows, timestamps_of

VERSIONS = (("squared, weighted (G)", "square", True), ("squared, weight 1", "square", False),
            ("absolute, weighted", "absolute", True), ("absolute, weight 1", "absolute", False))


def find_weights(node):
    """The dict of reconstruction weights per variable anywhere in the metadata, or None."""
    if isinstance(node, dict):
        value = node.get("reconstruction_feature_weights")
        if isinstance(value, dict) and value:
            return value
        for child in node.values():
            found = find_weights(child)
            if found is not None:
                return found
    elif isinstance(node, list):
        for child in node:
            found = find_weights(child)
            if found is not None:
                return found
    return None


def cell_values(errors, weights, kind):
    """Per cell, the mean over the variables of the weighted error: errors [cell, feature] squared errors."""
    errors = np.asarray(errors, dtype=np.float64)
    if kind == "square":
        return (errors * weights).mean(axis=1)
    return (np.sqrt(np.clip(errors, 0, None)) * np.sqrt(weights)).mean(axis=1)


def patch_mean(values, rows, cols, shape, patch=3):
    """Mean of `values` [cell] over the patch x patch window of each cell, over the cells of the grid only."""
    grid = np.zeros(shape)
    present = np.zeros(shape)
    grid[rows, cols] = values
    present[rows, cols] = 1.0
    half = patch // 2
    padded = np.pad(grid, half)
    counted = np.pad(present, half)
    total = np.zeros(shape)
    count = np.zeros(shape)
    for dr in range(patch):
        for dc in range(patch):
            total += padded[dr:dr + shape[0], dc:dc + shape[1]]
            count += counted[dr:dr + shape[0], dc:dc + shape[1]]
    return (total / np.maximum(count, 1))[rows, cols]


def minmax(values, low, high):
    return (values - low) / (high - low)


def iqr(values):
    q25, q75 = np.percentile(values, (25, 75))
    return q75 - q25


def mean_spatial_spearman(first, second):
    """Mean over the rows [time, cell] of the Spearman correlation over the cells."""
    out = []
    for a, b in zip(first, second):
        keep = np.isfinite(a) & np.isfinite(b)
        out.append(np.corrcoef(rankdata(a[keep]), rankdata(b[keep]))[0, 1])
    return float(np.mean(out))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="Scored validation run with scores/feature_scores.npy")
    parser.add_argument("--masks", type=Path, required=True, help="Normal masks directory (validation_mask.npy)")
    parser.add_argument("--output", type=Path, required=True, help="CSV with one row per version")
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args(argv)
    manifest = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
    weights_by_name = find_weights(manifest)
    if weights_by_name is None:
        raise ValueError("No reconstruction_feature_weights in the metadata: the run has no variable weights")
    weights = np.asarray(list(weights_by_name.values()), dtype=np.float64)
    grid = manifest["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (int(rows.max()) + 1, int(cols.max()) + 1)
    patch = int(manifest.get("patch_size", find_key(manifest, "patch_size") or 3))
    times = timestamps_of(args.run)
    errors = np.load(args.run / "scores" / "feature_scores.npy", mmap_mode="r")
    if errors.shape != (len(times), len(rows), len(weights)):
        raise ValueError(f"feature_scores {errors.shape}, expected {(len(times), len(rows), len(weights))}")
    keep_all = ~mask_rows(np.asarray(np.load(args.masks / "validation_mask.npy", mmap_mode="r"), dtype=bool), len(times))
    picked = np.arange(0, len(times), args.stride)
    saved_g, g_maps = open_component(args.run, manifest, "generator")
    saved_d, d_maps = open_component(args.run, manifest, "discriminator")
    try:
        versions = {name: np.empty((len(picked), len(rows)), np.float64) for name, _, _ in VERSIONS}
        g_run = np.asarray(saved_g[picked], dtype=np.float64)
        d = np.asarray(saved_d[picked], dtype=np.float64)
        for i, row in enumerate(picked):
            e = np.asarray(errors[row])
            for name, kind, weighted in VERSIONS:
                versions[name][i] = patch_mean(cell_values(e, weights if weighted else np.ones_like(weights), kind),
                                               rows, cols, shape, patch)
    finally:
        for array in g_maps + d_maps:
            array._mmap.close()
    keep = keep_all[picked]
    rebuilt = versions[VERSIONS[0][0]]
    relative = np.abs(rebuilt - g_run) / np.maximum(np.abs(g_run), 1e-12)
    print(f"[tail] {args.run} | {len(picked)} timestamps (one every {args.stride}), {len(rows)} cells, patch {patch} | "
          f"weights {weights.min():.3g}-{weights.max():.3g} (mean {weights.mean():.3g})")
    print(f"[tail] check: rebuilt G vs saved G: median relative difference {np.median(relative):.2e}, "
          f"99th percentile {np.percentile(relative, 99):.2e}, spatial Spearman {mean_spatial_spearman(rebuilt, g_run):.4f}")
    d_kept = d[keep]
    d_low, d_high = d_kept.min(), d_kept.max()
    records = []
    for name, _, _ in VERSIONS:
        values = versions[name]
        kept = values[keep]
        median = np.median(kept)
        low, high = kept.min(), kept.max()
        g_scaled, d_scaled = minmax(values, low, high), minmax(d, d_low, d_high)
        total = g_scaled + d_scaled
        records.append({
            "version": name, "max_over_median": high / median,
            "p99_over_median": np.percentile(kept, 99) / median,
            "iqr_d_over_iqr_version": iqr(d_scaled[keep]) / iqr(g_scaled[keep]),
            "spearman_sum_d": mean_spatial_spearman(total, d), "spearman_sum_version": mean_spatial_spearman(total, values)})
    found = find_ranges(manifest)
    if found:
        ranges, _ = select_ranges(found, "backend.score_normalization", "backend.validation_objective.score_ranges")
        g_scaled = minmax(g_run, ranges["r_min"], ranges["r_max"])
        d_scaled = minmax(d, ranges["d_min"], ranges["d_max"])
        total = g_scaled + d_scaled
        kept = g_run[keep]
        records.append({
            "version": "saved G, run's min-max extremes", "max_over_median": kept.max() / np.median(kept),
            "p99_over_median": np.percentile(kept, 99) / np.median(kept),
            "iqr_d_over_iqr_version": iqr(d_scaled[keep]) / iqr(g_scaled[keep]),
            "spearman_sum_d": mean_spatial_spearman(total, d), "spearman_sum_version": mean_spatial_spearman(total, g_run)})
    table = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.round(3).to_string(index=False))
    print(f"[tail] results in {args.output}")


def find_key(node, key):
    """The first value of `key` anywhere in the metadata, or None."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for child in node.values():
            found = find_key(child, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for child in node:
            found = find_key(child, key)
            if found is not None:
                return found
    return None


if __name__ == "__main__":
    main()
