"""Where a score flags alone, did the stations measure unusual rain that day?

The rain is the daily precipitation of E-OBS on the cells of the run (`download_eobs.py`): built from
station observations, not from ERA5, so a judge that none of the scores saw. A cell-day is flagged by a
score when the score is in its highest `--top-percent` at one or more timestamps of that UTC day. The
cell-days are in groups: flagged by the first score and by none of the others, by the first and at least
one other, by the others only, by none; and the ring, the cell-days next to (within `--ring` cells) a
cell-day of the first score alone, flagged by no score: the weather around it.

The rain of a cell-day is unusual when it is in its highest `--rain-percent` for the cell and the
calendar month over the days read. For every group the check says how many of its cell-days have
unusual rain on the same day, on the next day (the E-OBS day of many countries ends at 06 UTC of the
next day, so the rain of an evening can fall in it), and on either. About `--rain-percent` if the group
has nothing to do with rain, more if it flags rainy weather. Only cell-days with a measured rain count.

    python scripts/check_era5_score_against_observed_rain.py --rain-dir data/eobs \
        --score-dir outputs/run_clim --by-dirs outputs/run_no_model_clim outputs/run_previous_clim
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import numpy as np
import pandas as pd
from scipy import ndimage

from scripts.check_era5_dust_lead import THRESHOLD_STRIDE, score_file

GROUPS = ("score alone", "score and another", "others only", "none", "ring around score alone", "all cell-days")
NAME, WIDTH = 28, 14
DAYS_AT_A_TIME = 32


def daily_flags(scores, limits, timestamps, days):
    """For every day of `days` and cell: is the first score flagged at some timestamp of that UTC day, and is
    any of the others; and are all the scores finite at every timestamp of the day."""
    day_of = timestamps.normalize()
    first = np.zeros((len(days), scores[0].shape[1]), dtype=bool)
    others, finite = np.zeros_like(first), np.ones_like(first)
    positions = pd.Index(days).get_indexer(day_of)
    for start in range(0, len(days), DAYS_AT_A_TIME):
        rows = np.nonzero((positions >= start) & (positions < start + DAYS_AT_A_TIME))[0]
        if not len(rows):
            continue
        where = positions[rows]
        values = [np.asarray(array[rows[0]:rows[-1] + 1]) for array in scores]
        with np.errstate(invalid="ignore"):
            marks = [block > limit for block, limit in zip(values, limits)]
        ok = np.logical_and.reduce([np.isfinite(block) for block in values])
        np.logical_or.at(first, where, marks[0])
        np.logical_or.at(others, where, np.logical_or.reduce(marks[1:]))
        np.logical_and.at(finite, where, ok)
    covered = np.zeros(len(days), dtype=bool)
    covered[np.unique(positions[positions >= 0])] = True
    finite &= covered[:, None]
    return first, others, finite


def unusual_rain(rain, days, percent):
    """Where the rain [days, cells] is in its highest `percent` for the cell and the calendar month. Dry
    months whose threshold is zero flag nothing."""
    unusual = np.zeros(rain.shape, dtype=bool)
    months = pd.DatetimeIndex(days).month.to_numpy()
    for month in range(1, 13):
        chosen = months == month
        if not chosen.any():
            continue
        block = rain[chosen]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            limit = np.nanpercentile(block, 100 - percent, axis=0)
            unusual[chosen] = (block > limit) & (limit > 0)
    return unusual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rain-dir", type=Path, required=True, help="Output of download_eobs.py")
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score")
    parser.add_argument("--by-dirs", type=Path, nargs="+", required=True, help="Runs with the other scores")
    parser.add_argument("--top-percent", type=float, default=5, help="Flagged: this highest share of each score")
    parser.add_argument("--rain-percent", type=float, default=5, help="Unusual rain: this highest share of the days")
    parser.add_argument("--ring", type=int, default=1, help="Width of the ring around the cell-days of the score alone")
    args = parser.parse_args(argv)
    if args.ring < 1:
        raise ValueError("ring must be at least 1")
    manifest = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))
    grid = manifest["grid"]
    rows, cols = np.asarray(grid["rows"]), np.asarray(grid["cols"])
    lattice = (len(grid["latitudes"]), len(grid["longitudes"]))
    timestamps = pd.DatetimeIndex(np.load(args.score_dir / "test_timestamps.npy"))
    rain = np.load(args.rain_dir / "rain.npy")
    rain_days = pd.DatetimeIndex(np.load(args.rain_dir / "days.npy").astype("datetime64[D]"))
    if rain.shape[1] != len(rows):
        raise ValueError("The rain is not over the cells of the score")
    paths = [args.score_dir, *args.by_dirs]
    scores = []
    try:
        for path in paths:
            if not pd.DatetimeIndex(np.load(path / "test_timestamps.npy")).equals(timestamps):
                raise ValueError(f"The score is not over the same timestamps: {path}")
            run = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            scores.append(np.load(path / score_file(run, "combined"), mmap_mode="r"))
            if scores[-1].shape != (len(timestamps), len(rows)):
                raise ValueError(f"The score is not over the cells: {path}")
        limits = [float(np.nanpercentile(np.asarray(array[::THRESHOLD_STRIDE]), 100 - args.top_percent)) for array in scores]
        days = rain_days[(rain_days >= timestamps[0].normalize()) & (rain_days <= timestamps[-1].normalize())]
        first, others, finite = daily_flags(scores, limits, timestamps, days)
    finally:
        for array in scores:
            array._mmap.close()
    rain = rain[rain_days.get_indexer(days)]
    unusual = unusual_rain(rain, days, args.rain_percent)
    measured = np.isfinite(rain)
    # The next day: the last day has none.
    unusual_next = np.zeros_like(unusual)
    unusual_next[:-1] = unusual[1:]
    measured_next = np.zeros_like(measured)
    measured_next[:-1] = measured[1:]
    alone = first & ~others & finite
    stack = np.zeros((len(days), *lattice), dtype=bool)
    stack[:, rows, cols] = alone
    around = ndimage.binary_dilation(stack, structure=np.ones((1, 3, 3), dtype=bool), iterations=args.ring)[:, rows, cols]
    masks = (alone, first & others & finite, ~first & others & finite, ~first & ~others & finite,
             around & ~first & ~others & finite, finite)
    names = [path.name for path in paths]
    common = len(os.path.commonprefix(names))
    print(f"[observed-rain] {days[0].date()} to {days[-1].date()} | flagged: the highest {args.top_percent:g}% of "
          f"each score, at one timestamp of the UTC day at least | first: {names[0][common:] or names[0]} | others: "
          f"{', '.join(name[common:] or name for name in names[1:])} | unusual rain: highest {args.rain_percent:g}% "
          f"for the cell and the month | cells with stations: {int(measured.any(axis=0).sum())} of {len(rows)}")
    print(f"\n% of the cell-days with unusual rain measured by the stations")
    print(f"{'':<{NAME}}" + "".join(f"{name:>{WIDTH}}" for name in ("cell-days", "same day", "next day", "either")))
    rows_out = {}
    for name, mask in zip(GROUPS, masks):
        same = mask & measured
        both = same & measured_next
        values = [100 * (unusual & same).sum() / max(same.sum(), 1),
                  100 * (unusual_next & mask & measured_next).sum() / max((mask & measured_next).sum(), 1),
                  100 * ((unusual | unusual_next) & both).sum() / max(both.sum(), 1)]
        rows_out[name] = values
        print(f"{name:<{NAME}}{int(same.sum()):>{WIDTH}}" + "".join(f"{value:>{WIDTH}.1f}" for value in values))
    print("same day: the UTC day of the flag; next day: the day after, as the E-OBS day often ends at 06 UTC of it; "
          "either: one of the two (only cell-days with both measured)")
    return rows_out


if __name__ == "__main__":
    main()
