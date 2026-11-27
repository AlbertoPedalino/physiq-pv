"""A sample of the within-timestamp ranks of G, D and the combined score, for a scatter plot of their correlation.

The spatial Spearman correlation of two scores is, at every timestamp, the correlation of their ranks over the cells.
This script draws `--timestamps` timestamps at random (fixed seed), ranks each score over all the cells of each drawn
timestamp (ranks scaled to 0-1, ties averaged), and keeps `--cells` random cells per timestamp. The output CSV
(timestamp, cell, g, d, combined) can be drawn as clouds of points whose shape matches the spatial Spearman
correlation of the thesis. As a check, the mean over the drawn timestamps of the spatial Spearman correlation of every
pair (computed on all the cells) is printed, next to the correlation of the pooled sampled ranks.

Usage (on the server, from the checkout):
    python scripts/sample_era5_score_ranks.py --run outputs/stgan_optuna27_validation \
        --output outputs/presentazione/ranghi_g_d_combinato.csv
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

from scripts.check_era5_score_correlation import open_component
from scripts.check_era5_yearly_share_and_skill import timestamps_of

SCORES = (("g", "generator"), ("d", "discriminator"), ("combined", "combined"))
PAIRS = (("g", "d"), ("combined", "d"), ("combined", "g"))


def scaled_ranks(values):
    """Ranks of the finite values of a row scaled to 0-1 (ties averaged); NaN where the value is not finite."""
    values = np.asarray(values, dtype=np.float64)
    ranks = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    if finite.sum() > 1:
        ranks[finite] = (rankdata(values[finite]) - 1) / (finite.sum() - 1)
    return ranks


def pearson(a, b):
    keep = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="Scored run with G, D and the combined score")
    parser.add_argument("--output", type=Path, required=True, help="CSV to write")
    parser.add_argument("--timestamps", type=int, default=300)
    parser.add_argument("--cells", type=int, default=10, help="Cells kept per drawn timestamp")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    manifest = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
    times = timestamps_of(args.run)
    opened = {name: open_component(args.run, manifest, component) for name, component in SCORES}
    try:
        shapes = {name: scores.shape for name, (scores, _) in opened.items()}
        if len(set(shapes.values())) != 1 or shapes["g"][0] != len(times):
            raise ValueError(f"Scores of different shapes or not matching the timestamps: {shapes}, {len(times)}")
        n_times, n_cells = shapes["g"]
        generator = np.random.Generator(np.random.PCG64(args.seed))
        rows = np.sort(generator.choice(n_times, size=args.timestamps, replace=False))
        records, per_timestamp = [], {pair: [] for pair in PAIRS}
        for row in rows:
            ranks = {name: scaled_ranks(scores[row]) for name, (scores, _) in opened.items()}
            for pair in PAIRS:
                per_timestamp[pair].append(pearson(ranks[pair[0]], ranks[pair[1]]))
            for cell in generator.choice(n_cells, size=args.cells, replace=False):
                records.append({"timestamp": times[row], "cell": int(cell),
                                **{name: ranks[name][cell] for name, _ in SCORES}})
    finally:
        for _, maps in opened.values():
            for array in maps:
                array._mmap.close()
    sample = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)
    print(f"[ranks] {args.run} | {len(rows)} timestamps from {times[rows[0]]} to {times[rows[-1]]} | "
          f"{n_cells} cells ranked, {args.cells} kept per timestamp | seed {args.seed}")
    for pair in PAIRS:
        print(f"[ranks] {pair[0]} - {pair[1]}: spatial Spearman, mean over the drawn timestamps "
              f"{np.nanmean(per_timestamp[pair]):.3f} (median {np.nanmedian(per_timestamp[pair]):.3f}) | "
              f"pooled sampled ranks {pearson(sample[pair[0]].to_numpy(), sample[pair[1]].to_numpy()):.3f}")
    print(f"[ranks] {len(sample)} points in {args.output}")


if __name__ == "__main__":
    main()
