"""Other ways of combining the two components of the score, as other GAN detectors do: which component do they follow?

A curiosity check, outside the thesis. A scored run holds, for every cell and timestamp, the generator error G and the
discriminator component D (mean over the draws). Their combination is built here in the ways of the literature, on one
timestamp every `--stride` (the whole run otherwise needs tens of GB):
- `stgan`: min-max of each component with the extremes of the run's metadata (score_normalization), then the sum, as the
  combined score of the run (STGAN, lambda = 1);
- `anogan`: (1 - lambda) G + lambda D with lambda = 0.1, on the raw components (Schlegl et al. 2017);
- `tadgan_mean` and `tadgan_product`: both components turned into z-scores (mean and standard deviation over the values
  read), then their mean (alpha = 0.5) or their product (Geiger et al. 2020);
- `rank_mean`: the mean of the two percentile ranks, a combination that no scale can tilt (a control).
For each combination: the Spearman correlation with G and with D, and the Jaccard index of its highest `--top-percent`
with the highest of G and of D.

    python scripts/check_era5_score_combinations.py --run-dir outputs/stgan_optuna27_validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import rankdata, spearmanr

CHUNK = 64


def read_sample(path, stride):
    """The values of one timestamp every `stride` of a [timestamps, cells] score, as one flat array."""
    values = np.load(path, mmap_mode="r")
    try:
        rows = range(0, len(values), stride)
        return np.concatenate([np.asarray(values[i], dtype=np.float64) for i in rows])
    finally:
        values._mmap.close()


def combinations(g, d, ranges):
    """The combined scores of the literature from the two components (same shape, finite values)."""
    r_min, r_max, d_min, d_max = ranges
    zg, zd = (g - g.mean()) / g.std(), (d - d.mean()) / d.std()
    pg, pd_ = rankdata(g) / len(g), rankdata(d) / len(d)
    return {"stgan": (g - r_min) / (r_max - r_min) + (d - d_min) / (d_max - d_min),
            "anogan": 0.9 * g + 0.1 * d,
            "tadgan_mean": 0.5 * zg + 0.5 * zd,
            "tadgan_product": zg * zd,
            "rank_mean": 0.5 * (pg + pd_)}


def top(values, percent):
    return values >= np.percentile(values, 100 - percent)


def jaccard(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def compare(g, d, ranges, percent):
    rows = {}
    g_top, d_top = top(g, percent), top(d, percent)
    for name, score in combinations(g, d, ranges).items():
        mark = top(score, percent)
        rows[name] = {"spearman_G": float(spearmanr(score, g).statistic),
                      "spearman_D": float(spearmanr(score, d).statistic),
                      "jaccard_G": jaccard(mark, g_top), "jaccard_D": jaccard(mark, d_top)}
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ranges-run", type=Path, help="Run whose metadata holds the min-max extremes (--run-dir if left out)")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--top-percent", type=float, default=5)
    args = parser.parse_args(argv)
    run = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    source = run if args.ranges_run is None else json.loads((args.ranges_run / "metadata.json").read_text(encoding="utf-8"))
    norm = source["backend"]["score_normalization"]
    ranges = tuple(float(norm[key]) for key in ("r_min", "r_max", "d_min", "d_max"))
    g = read_sample(args.run_dir / run["generator_mean_file"], args.stride)
    d = read_sample(args.run_dir / run["discriminator_mean_file"], args.stride)
    keep = np.isfinite(g) & np.isfinite(d)
    g, d = g[keep], d[keep]
    print(f"[combine] {args.run_dir}: {len(g)} values (one timestamp every {args.stride}); min-max of the run "
          f"r {ranges[0]:.4g}-{ranges[1]:.4g}, d {ranges[2]:.4g}-{ranges[3]:.4g}; Spearman G-D "
          f"{spearmanr(g, d).statistic:+.3f}; highest {args.top_percent:g}%", flush=True)
    print(f"{'combination':<16}{'Spearman G':>12}{'Spearman D':>12}{'Jaccard G':>11}{'Jaccard D':>11}")
    for name, row in compare(g, d, ranges, args.top_percent).items():
        print(f"{name:<16}{row['spearman_G']:>+12.3f}{row['spearman_D']:>+12.3f}{row['jaccard_G']:>11.3f}"
              f"{row['jaccard_D']:>11.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
