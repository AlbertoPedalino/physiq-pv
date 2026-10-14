"""Where a score flags, is the combination of the variables unusual even when each of them is not?

Two distances of the standardised data from the seasonal normal, at every cell and timestamp:
- the separate one, the sum of the squared values of the variables over their variances: each
  variable counts by itself;
- the joint one (Mahalanobis), with the covariance of the variables in the training years: a
  combination the variables rarely take together counts even when each of them is near its normal.
The same two are read on the change of the data over `--change-hours`, with the covariance of the
changes in the training years. A distance is unusual when it is in its highest `--unusual-percent`
for the cell, the calendar month and the time of day; a value has an unusual combination when its
joint distance is unusual and its separate one is not.

A value is flagged by a score when it is in its highest `--top-percent`. The values are in groups, as
in `check_era5_score_against_unseen.py`: flagged by the first score alone, by it and the others, by the
others alone, by none. For every group the check says in how many values each distance is unusual and
in how many the combination is unusual: if the first score flags combinations the others miss, its
group alone has them more often than the others alone and than all the values.

With `--local-degrees` the covariances are estimated apart in every box of that many degrees and in
every season (December to February, March to May, June to August, September to November), so that a
combination usual in one place and season is not taken as unusual because it is rare elsewhere. The
training timestamps are then read from `--prepared-dir`.

    python scripts/check_era5_combination.py --run-dir outputs/run --score-dir outputs/run_clim \
        --by-dirs outputs/run_simple_max
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

from physiq_pv.era5.data import load_prepared
from scripts.check_era5_dust_lead import THRESHOLD_STRIDE, score_file
from scripts.check_era5_score_against_unseen import GROUPS, cache_of, unusual_values

FIT_STEP = 50  # the covariances are read on one training timestamp every this many
COLUMNS = ("joint state", "separate state", "combination, state", "joint change", "separate change",
           "combination, change")
NAME, WIDTH = 36, 14
SEASON_OF_MONTH = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0])  # month 1..12 -> season, December in winter
RIDGE = 1e-6  # added to the diagonal, relative to its mean, where a box hardly varies (soil water over the sea)


def covariances(train, lag):
    """Covariance of the standardised variables and of their change over `lag` timestamps, in the training years."""
    rows = np.arange(lag, len(train), FIT_STEP)
    now = np.asarray(train[rows], dtype=np.float64).reshape(-1, train.shape[2])
    before = np.asarray(train[rows - lag], dtype=np.float64).reshape(-1, train.shape[2])
    keep = np.isfinite(now).all(axis=1) & np.isfinite(before).all(axis=1)
    return np.cov(now[keep], rowvar=False), np.cov((now - before)[keep], rowvar=False)


def local_covariances(train, times, box, lag):
    """Covariances [season, box, variables, variables] of the state and of the change over `lag` timestamps in the
    training years, the box of every cell given by `box`."""
    variables, boxes = train.shape[2], int(box.max()) + 1
    state, change = (np.zeros((4, boxes, variables, variables)) for _ in range(2))
    rows = np.arange(lag, len(train), FIT_STEP)
    season = SEASON_OF_MONTH[times.month.to_numpy()[rows] - 1]
    for index in range(4):
        chosen = rows[season == index]
        now = np.asarray(train[chosen], dtype=np.float64)
        before = np.asarray(train[chosen - lag], dtype=np.float64)
        for place in range(boxes):
            cells = box == place
            for target, values in ((state, now[:, cells]), (change, (now - before)[:, cells])):
                flat = values.reshape(-1, variables)
                flat = flat[np.isfinite(flat).all(axis=1)]
                if len(flat) > variables:
                    target[index, place] = np.cov(flat, rowvar=False)
    return state, change


def inverse_of(covariance):
    """Inverse and diagonal of a covariance, made invertible where it hardly varies."""
    covariance = np.asarray(covariance, dtype=np.float64)
    diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
    ridge = RIDGE * np.maximum(diagonal.mean(axis=-1), 1e-12)
    regular = covariance + ridge[..., None, None] * np.eye(covariance.shape[-1])
    return np.linalg.inv(regular), np.diagonal(regular, axis1=-2, axis2=-1)


def distances(values, covariance):
    """Joint (Mahalanobis) and separate squared distances of `values` [..., cells, variables] from zero, with one
    covariance, or one per cell [cells, variables, variables]."""
    inverse, variance = inverse_of(covariance)
    if inverse.ndim == 2:
        joint = np.einsum("...i,ij,...j->...", values, inverse, values, optimize=True)
    else:
        joint = np.einsum("...ci,cij,...cj->...c", values, inverse, values, optimize=True)
    separate = (values ** 2 / variance).sum(axis=-1)
    return joint.astype(np.float32), separate.astype(np.float32)


def month_of(standardised, first, rows, lag, state, change):
    """The four distances at the scored timestamps `rows` (in order): [timestamps, cells, 4]. `state` and `change`
    are one covariance each, or one per cell."""
    result = []
    for block in np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1):
        start, stop = first + block[0], first + block[-1] + 1
        now = np.asarray(standardised[start:stop], dtype=np.float64)
        before = np.asarray(standardised[start - lag:stop - lag], dtype=np.float64)
        result.append(np.stack((*distances(now, state), *distances(now - before, change)), axis=-1))
    return np.concatenate(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run with the seasonal normalization")
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the score that is asked about")
    parser.add_argument("--by-dirs", type=Path, nargs="+", required=True, help="Runs with the other scores")
    parser.add_argument("--top-percent", type=float, default=5, help="Flagged: this highest share of each score")
    parser.add_argument("--unusual-percent", type=float, default=5, help="Unusual: this highest share of a distance")
    parser.add_argument("--change-hours", type=int, default=3)
    parser.add_argument("--local-degrees", type=float, default=0,
                        help="Estimate the covariances apart by box of this many degrees and by season")
    parser.add_argument("--prepared-dir", type=Path, help="With --local-degrees: the prepared data, for the training timestamps")
    args = parser.parse_args(argv)
    if args.local_degrees and args.prepared_dir is None:
        raise ValueError("--local-degrees needs --prepared-dir, for the training timestamps")
    if args.change_hours < 3 or args.change_hours % 3:
        raise ValueError("change-hours must be a positive multiple of 3")
    lag = args.change_hours // 3
    paths = [args.score_dir, *args.by_dirs]
    names = [path.name for path in paths]
    common = len(os.path.commonprefix(names))
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    cells = len(grid["rows"])
    timestamps = pd.DatetimeIndex(np.load(args.score_dir / "test_timestamps.npy"))
    standardised, first, _ = cache_of(args.run_dir, timestamps, cells)
    opened = [standardised]
    try:
        if first < lag:
            raise ValueError("The cache holds too few timestamps before the scored period for the change")
        train = np.load(Path(standardised.filename).parent / "train.npy", mmap_mode="r")
        opened.append(train)
        if args.local_degrees:
            cubes = load_prepared(args.prepared_dir)[0]
            try:
                times = pd.DatetimeIndex(cubes.train_timestamps)
                held = getattr(cubes, "validation_timestamps", None)
            finally:
                cubes.close()
            # A run trained on the validation year too has it at the end of its training data.
            if held is not None and len(times) + len(held) == len(train):
                times = times.append(pd.DatetimeIndex(held))
            if len(times) != len(train):
                raise ValueError(f"The training timestamps of the prepared data ({len(times)}, validation "
                                 f"{0 if held is None else len(held)}) are not those of the cache ({len(train)})")
            latitudes = np.asarray(grid["latitudes"], dtype=float)[np.asarray(grid["rows"])]
            longitudes = np.asarray(grid["longitudes"], dtype=float)[np.asarray(grid["cols"])]
            across = int(np.floor((longitudes.max() - longitudes.min()) / args.local_degrees)) + 1
            box = (np.floor((latitudes - latitudes.min()) / args.local_degrees).astype(int) * across
                   + np.floor((longitudes - longitudes.min()) / args.local_degrees).astype(int))
            box = np.unique(box, return_inverse=True)[1]
            local_state, local_change = local_covariances(train, pd.DatetimeIndex(times), box, lag)
        else:
            state, change = covariances(train, lag)
        scores = []
        for path in paths:
            if not pd.DatetimeIndex(np.load(path / "test_timestamps.npy")).equals(timestamps):
                raise ValueError(f"The score is not over the same timestamps: {path}")
            manifest = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            scores.append(np.load(path / score_file(manifest, "combined"), mmap_mode="r"))
            opened.append(scores[-1])
            if scores[-1].shape != (len(timestamps), cells):
                raise ValueError(f"The score is not over the same cells: {path}")
        limits = [float(np.nanpercentile(np.asarray(array[::THRESHOLD_STRIDE]), 100 - args.top_percent)) for array in scores]
        count, hits = np.zeros(len(GROUPS)), np.zeros((len(GROUPS), len(COLUMNS)))
        for month in range(1, 13):
            rows = np.nonzero(timestamps.month == month)[0]
            if not len(rows):
                continue
            if args.local_degrees:
                season = SEASON_OF_MONTH[month - 1]
                state, change = local_state[season][box], local_change[season][box]
            unusual = unusual_values(month_of(standardised, first, rows, lag, state, change),
                                     timestamps.hour[rows].to_numpy(), args.unusual_percent)[0]
            flags = np.stack((unusual[..., 0], unusual[..., 1], unusual[..., 0] & ~unusual[..., 1],
                              unusual[..., 2], unusual[..., 3], unusual[..., 2] & ~unusual[..., 3]), axis=-1)
            del unusual
            values = [np.asarray(array[rows]) for array in scores]
            finite = np.logical_and.reduce([np.isfinite(block) for block in values])
            with np.errstate(invalid="ignore"):
                marks = [finite & (block > limit) for block, limit in zip(values, limits)]
            some = np.logical_or.reduce(marks[1:])
            for index, group in enumerate((marks[0] & ~some, marks[0] & some, ~marks[0] & some,
                                           finite & ~marks[0] & ~some, finite)):
                count[index] += group.sum()
                hits[index] += flags[group].sum(axis=0)
    finally:
        for array in opened:
            array._mmap.close()
    print(f"[combination] {timestamps[0].date()} to {timestamps[-1].date()} | flagged: the highest {args.top_percent:g}% "
          f"of each score | score: {names[0][common:] or names[0]} | others: "
          f"{', '.join(name[common:] or name for name in names[1:])} | unusual: the highest {args.unusual_percent:g}% for the "
          f"cell, the calendar month and the time of day | change over {args.change_hours} h | covariances "
          + (f"by box of {args.local_degrees:g} degrees and by season" if args.local_degrees else "of the whole domain"))
    print("joint: the distance with the covariance of the variables in the training years; separate: each variable by "
          "itself; combination: the joint one unusual and the separate one not")
    print(f"\n% of the values of each group\n{'':<{NAME}}{'values':>{WIDTH}}" + "".join(f"{name:>{WIDTH + 6}}"
                                                                                for name in COLUMNS))
    with np.errstate(invalid="ignore", divide="ignore"):
        for index, name in enumerate(GROUPS):
            print(f"{name:<{NAME}}{int(count[index]):>{WIDTH}}"
                  + "".join(f"{100 * hit / count[index]:>{WIDTH + 6}.1f}" for hit in hits[index]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
