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
