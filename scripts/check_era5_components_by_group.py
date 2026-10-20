"""What do the generator and the discriminator, and their dropout draws, do where only one score flags?

The values are put in groups by the flags of a first score (`--score-dir`, e.g. the generator's error with
`climatology`) and of other scores (`--by-dir`, any of them): first score alone, both, others alone, none;
a value is flagged above `score_limit` of the highest `--top-percent` (the threshold of the validation for
a run that names one). For every group the check reads, in a run scored with Monte Carlo dropout
(`--components-dir`), the mean over the draws and the spread of the draws of the reconstruction error (G)
and of the discriminator component D(real) - D(generated) (D): their medians, the median of the spread
over the mean for G, and for D the share of the values in the highest `--top-percent` of D itself, which
says whether D sees as unusual what the group holds. A spread that grows where the first score flags alone
says the draws of that network disagree there more than elsewhere.

    python scripts/check_era5_components_by_group.py --score-dir outputs/test_gen_clim \
        --by-dir outputs/test_simple_max --components-dir outputs/run_scored
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import numpy as np
import pandas as pd

from scripts.check_era5_dust_lead import score_file, score_limit

GROUPS = ("first alone", "both", "others alone", "none")
FILES = ("generator_mean", "generator_std", "discriminator_mean", "discriminator_std")
NAME, WIDTH = 16, 13
SPREADS = ("G spread", "G spread / error", "D spread")
# G and D together, with nothing fitted: the mean of the ranks of the two spreads among the values read (0 to 1);
# and the mean of the rank of the spread of G and of the rank of the spread of D turned upside down, high where the
# draws of G disagree and those of D agree.
COMBINED = "G and D spreads, mean rank"
OPPOSED = "G up and D down, mean rank"
MOST = 2_000_000  # values read for the medians and the AUC of one group


def open_components(directory):
    """The dropout draws of a run scored with Monte Carlo dropout: {name: [timestamps, cells]}, read lazily."""
    return {name: np.load(directory / "scores" / f"{name}.npy", mmap_mode="r") for name in FILES}


def spreads_at(components, t, normal=None):
    """[cells, 3] of the spread of G, the spread of G over its mean error and the spread of D at timestamp t; with
    `normal` ([cells, 3], spread_normal), each divided by its usual value at the cell for that month and hour."""
    mean, spread, d_spread = (np.asarray(components[name][t], dtype=np.float64)
                              for name in ("generator_mean", "generator_std", "discriminator_std"))
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.stack([spread, spread / mean, d_spread], axis=1)
        return values if normal is None else values / normal


def build_spread_normals(components, timestamps):
    """The usual spreads (spreads_at) of every cell for each calendar month and hour of the day: the mean over the
    timestamps of the run. {"keys": [n, 2] month and hour, "values": [n, cells, 3]}."""
    keys = sorted(set(zip(timestamps.month, timestamps.hour)))
    index = {key: k for k, key in enumerate(keys)}
    cells = components["generator_std"].shape[1]
    total, count = np.zeros((len(keys), cells, 3)), np.zeros((len(keys), cells, 3))
    for t, key in enumerate(zip(timestamps.month, timestamps.hour)):
        values = spreads_at(components, t)
        finite = np.isfinite(values)
        total[index[key]] += np.where(finite, values, 0.0)
        count[index[key]] += finite
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"keys": np.asarray(keys, dtype=np.int64), "values": total / count}


def spread_normal(normals, timestamp):
    """[cells, 3] of the usual spreads for the month and hour of `timestamp`, or None without normals."""
    if normals is None:
        return None
    found = np.nonzero((normals["keys"][:, 0] == timestamp.month) & (normals["keys"][:, 1] == timestamp.hour))[0]
    if not len(found):
        raise ValueError(f"No usual spread for month {timestamp.month} and hour {timestamp.hour}")
    return normals["values"][found[0]]


def load_spread_normals(path):
    """The normals written by build_era5_spread_normals.py, or None for no path."""
    if path is None:
        return None
    data = np.load(path)
    return {"keys": data["keys"], "values": data["values"]}


def compare_spreads(hit, values, seed=0):
    """For every spread (the columns of `values`) and for G and D together (COMBINED): its median where `hit` is true
    and where it is not, and the area under the ROC curve of the spread as a score of the hits (0.5: the spread
    does not tell them apart)."""
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score

    keep = np.isfinite(values).all(axis=1)
    hit, values = np.asarray(hit, dtype=bool)[keep], values[keep]
    if len(hit) > MOST:
        chosen = np.random.default_rng(seed).choice(len(hit), MOST, replace=False)
        hit, values = hit[chosen], values[chosen]
    if len(hit):
        ranks = (rankdata(values[:, 0]) + rankdata(values[:, 2])) / (2 * len(hit))
        opposed = (rankdata(values[:, 0]) + rankdata(-values[:, 2])) / (2 * len(hit))
        values = np.column_stack([values, ranks, opposed])
    rows = []
    for k, name in enumerate((*SPREADS, COMBINED, OPPOSED)[:values.shape[1]]):
        median = lambda mask: float(np.median(values[mask, k])) if mask.any() else np.nan
        auc = roc_auc_score(hit, values[:, k]) if 0 < hit.sum() < len(hit) else np.nan
        rows.append({"spread": name, "hits": int(hit.sum()), "misses": int((~hit).sum()), "median hit": median(hit),
                     "median miss": median(~hit), "AUC": auc})
    return rows


def print_spreads(title, groups, hit_name="hit"):
    """Prints, for every (label, rows of compare_spreads), the medians and the AUC of every spread."""
    print(f"\n{title}")
    print(f"{'group, spread':<{2 * NAME + 8}}" + "".join(f"{name:>{WIDTH}}" for name in (
        hit_name, "not", f"med {hit_name}", "med not", "AUC")))
    for label, rows in groups:
        for row in rows:
            name = f"{label}, {row['spread']}"[:2 * NAME + 7]
            print(f"{name:<{2 * NAME + 8}}{row['hits']:>{WIDTH}d}"
                  f"{row['misses']:>{WIDTH}d}{row['median hit']:>{WIDTH}.4g}{row['median miss']:>{WIDTH}.4g}"
                  f"{row['AUC']:>{WIDTH}.3f}")
    print("spreads of the 20 dropout draws: G the reconstruction error, D the discriminator component; AUC above 0.5: "
          f"a larger spread goes with {hit_name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score")
    parser.add_argument("--by-dir", type=Path, nargs="+", required=True, help="Runs with the other scores")
    parser.add_argument("--components-dir", type=Path, required=True,
                        help="Run scored with dropout draws: scores/generator_mean, _std, discriminator_mean, _std")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--stride", type=int, default=9, help="Timestamps read: one every this many")
    args = parser.parse_args(argv)
    timestamps = pd.DatetimeIndex(np.load(args.score_dir / "test_timestamps.npy"))
    for path in (*args.by_dir, args.components_dir):
        if not pd.DatetimeIndex(np.load(path / "test_timestamps.npy")).equals(timestamps):
            raise ValueError(f"The runs are not over the same timestamps: {path}")
    paths = [args.score_dir, *args.by_dir]
    scores = [np.load(path / score_file(json.loads((path / "metadata.json").read_text(encoding="utf-8")), "combined"),
                      mmap_mode="r") for path in paths]
    components = {name: np.load(args.components_dir / "scores" / f"{name}.npy", mmap_mode="r") for name in FILES}
    try:
        limits = [score_limit(array, args.top_percent) for array in scores]
        d_limit = score_limit(components["discriminator_mean"], args.top_percent)
        rows = np.arange(0, len(timestamps), args.stride)
        first = np.asarray(scores[0][rows]) > limits[0]
        others = np.logical_or.reduce([np.asarray(array[rows]) > limit for array, limit in zip(scores[1:], limits[1:])])
        groups = (first & ~others, first & others, ~first & others, ~first & ~others)
        values = {name: np.asarray(array[rows], dtype=np.float64) for name, array in components.items()}
    finally:
        for array in (*scores, *components.values()):
            array._mmap.close()
    common = len(os.path.commonprefix([path.name for path in paths]))
    short = [path.name[common:] or path.name for path in paths]
    print(f"[components] {timestamps[0].date()} to {timestamps[-1].date()}, one timestamp every {args.stride} | "
          f"flagged: above the threshold of the highest {args.top_percent:g}% of {short[0]} and of "
          f"{' or '.join(short[1:])} | components: {args.components_dir.name}")
    header = ("values", "G mean", "G spread", "G spr/mean", "D mean", "D spread", "D flags %")
    print(f"\nmedians by group (first: {short[0]}; others: {' or '.join(short[1:])})")
    print(f"{'group':<{NAME}}" + "".join(f"{name:>{WIDTH}}" for name in header))
    result = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = values["generator_std"] / values["generator_mean"]
    for name, mask in zip(GROUPS, groups):
        if not mask.any():
            continue
        row = (int(mask.sum()), np.nanmedian(values["generator_mean"][mask]), np.nanmedian(values["generator_std"][mask]),
               np.nanmedian(ratio[mask]), np.nanmedian(values["discriminator_mean"][mask]),
               np.nanmedian(values["discriminator_std"][mask]),
               100 * np.mean(values["discriminator_mean"][mask] > d_limit))
        result[name] = dict(zip(header, row))
        print(f"{name:<{NAME}}{row[0]:>{WIDTH}d}" + "".join(f"{value:>{WIDTH}.4g}" for value in row[1:]))
    print(f"G: reconstruction error, mean and spread of the dropout draws; D: D(real) - D(generated), mean and spread; "
          f"D flags: % above the threshold of the highest {args.top_percent:g}% of D")
    return result


if __name__ == "__main__":
    main()
