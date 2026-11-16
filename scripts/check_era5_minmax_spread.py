"""How much G and D vary after the min-max of the combined score.

The combined score sums the two components after a min-max with r_min, r_max, d_min, d_max fitted on the validation years
(without the excluded values) and frozen. If after that min-max G spreads over a much narrower range than D, the sum is
ordered by D: this check prints, for the validation years, the median, the quartiles and the 1st and 99th percentiles of
each component after the min-max, over all the values and without the excluded ones, and the ratio of the two
interquartile ranges. The four extremes are read from the metadata of `--ranges-run` (the scored run of the thesis by
default: wherever a dict holds r_min, r_max, d_min and d_max); one timestamp every `--stride` is read.

Usage (on the server, from the checkout):
    python scripts/check_era5_minmax_spread.py --run outputs/stgan_optuna27_validation \
        --ranges-run outputs/stgan_optuna27_scored --masks outputs/era5_sweep_1999/normal_masks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.check_era5_score_correlation import open_component
from scripts.check_era5_yearly_share_and_skill import mask_rows, timestamps_of

KEYS = ("r_min", "r_max", "d_min", "d_max")
QUANTILES = (1, 25, 50, 75, 99)


def find_ranges(node, path="metadata"):
    """Every (path, dict) of the metadata that holds the four extremes of the min-max."""
    found = []
    if isinstance(node, dict):
        if all(key in node for key in KEYS):
            found.append((path, {key: float(node[key]) for key in KEYS}))
        for key, value in node.items():
            found.extend(find_ranges(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(find_ranges(value, f"{path}[{index}]"))
    return found


def select_ranges(found, key, check):
    """The extremes stored under `key` (a path below "metadata"), required to equal those under `check` when that key
    exists too; the other places holding extremes are returned apart, to be printed only."""
    by_path = dict(found)
    chosen = by_path.get(f"metadata.{key}")
    if chosen is None:
        raise ValueError(f"No r_min, r_max, d_min, d_max under metadata.{key}; found under: {[p for p, _ in found]}")
    other = by_path.get(f"metadata.{check}") if check else None
    if other is not None and other != chosen:
        raise ValueError(f"metadata.{key} {chosen} differs from metadata.{check} {other}")
    rest = [(p, r) for p, r in found if p not in (f"metadata.{key}", f"metadata.{check}")]
    return chosen, rest


def scaled_sample(scores, low, high, keep, stride):
    """(all, without the excluded) values of (scores - low) / (high - low), one timestamp every `stride`."""
    rows = np.arange(0, len(keep), stride)
    values = (np.asarray(scores[rows], dtype=np.float64) - low) / (high - low)
    finite = np.isfinite(values)
    return values[finite], values[finite & keep[rows]]


def describe(values):
    q = np.percentile(values, QUANTILES)
    return {f"p{k}": float(v) for k, v in zip(QUANTILES, q)} | {"iqr": float(q[3] - q[1]), "n": int(values.size)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="Validation run with the generator and discriminator scores")
    parser.add_argument("--ranges-run", type=Path, help="Run whose metadata holds r_min, r_max, d_min, d_max (--run if left out)")
    parser.add_argument("--masks", type=Path, required=True, help="Normal masks directory (validation_mask.npy)")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--key", default="backend.score_normalization",
                        help="Path below metadata of the extremes used by the combined score")
    parser.add_argument("--check-key", default="backend.validation_objective.score_ranges",
                        help="Path that must hold the same extremes, when present ('' to skip)")
    args = parser.parse_args(argv)
    source = args.ranges_run or args.run
    found = find_ranges(json.loads((source / "metadata.json").read_text(encoding="utf-8")))
    if not found:
        raise ValueError(f"No r_min, r_max, d_min, d_max in {source / 'metadata.json'}")
    ranges, rest = select_ranges(found, args.key, args.check_key)
    print(f"[minmax] extremes from {source} (metadata.{args.key}, equal to metadata.{args.check_key} where present): "
          + ", ".join(f"{k} {v:.6g}" for k, v in ranges.items()))
    for path, other in rest:
        print(f"[minmax] not used: {path}: " + ", ".join(f"{k} {v:.6g}" for k, v in other.items()))
    manifest = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
    times = timestamps_of(args.run)
    keep = ~mask_rows(np.asarray(np.load(args.masks / "validation_mask.npy", mmap_mode="r"), dtype=bool), len(times))
    print(f"[minmax] {times[0]} to {times[-1]}, one timestamp every {args.stride}; excluded values "
          f"{100 * (1 - keep.mean()):.2f}%")
    spread = {}
    for name, component, (low, high) in (("G", "generator", (ranges["r_min"], ranges["r_max"])),
                                         ("D", "discriminator", (ranges["d_min"], ranges["d_max"]))):
        scores, maps = open_component(args.run, manifest, component)
        try:
            if scores.shape != keep.shape:
                raise ValueError(f"{name}: scores {scores.shape}, mask {keep.shape}")
            everything, kept = scaled_sample(scores, low, high, keep, args.stride)
        finally:
            for array in maps:
                array._mmap.close()
        for label, values in (("all values", everything), ("without excluded", kept)):
            stats = describe(values)
            spread[(name, label)] = stats["iqr"]
            print(f"  {name} after min-max, {label}: " + ", ".join(f"{k} {v:.4g}" for k, v in stats.items()))
    for label in ("all values", "without excluded"):
        print(f"[minmax] {label}: interquartile range of D / that of G = "
              f"{spread[('D', label)] / spread[('G', label)]:.3g}")


if __name__ == "__main__":
    main()
