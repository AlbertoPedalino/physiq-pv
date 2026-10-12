"""Which floor of the standard deviation keeps the scored years on the scale of the training ones?

The seasonal normalization divides every value by the standard deviation of its cell, time of year
and time of day in the training years. Where that is almost 0 (rain in a dry season, sun at
night), a floor is used instead: a share of the standard deviation of the variable as a whole.
With a low floor a value of the scored years in such a place is divided by almost nothing and
ends far beyond anything the training years hold, although it is small.

No model is used. For every variable and floor the check gives:

- the share of cells and times at which the floor replaces their own standard deviation: what
  the floor costs, since there the normalization no longer follows the place and the season;
- the range of the standardised training values, which the min-max then maps to the range of the
  model;
- the share of the scored values beyond that range, and their largest size over the largest of
  the training years;
- the mean square of the standardised values in the scored years and in the training years: the
  same if the two are on one scale;
- the share of every variable in the sum of those mean squares after the min-max, a guess of
  which variable makes the plain reconstruction error.

The normal is by calendar month and time of day, as in `check_era5_climate_drift.py`, and every
training timestamp is read, so that a month has about as many values as the window of days of
the real normalization.

    python scripts/check_era5_normalization_floor.py --prepared-dir data/era5_prepared
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from physiq_pv.era5.data import load_prepared
from scripts.check_era5_climate_drift import bins, sample

FEATURES = ("total_precipitation", "surface_solar_radiation_downwards", "total_cloud_cover", "2m_temperature",
            "specific_humidity", "volumetric_soil_water_layer_1")
FLOORS = (0.01, 0.03, 0.1, 0.3, 1.0)


def fit_bins(values, keys):
    """Mean and standard deviation [bins, cells, variables] of every calendar month and time of day, the
    standard deviation of every variable as a whole, and the row of every timestamp among the bins."""
    labels, rows = np.unique(keys, return_inverse=True)
    mean = np.stack([values[rows == row].mean(axis=0, dtype=np.float64) for row in range(len(labels))])
    std = np.stack([values[rows == row].std(axis=0, dtype=np.float64) for row in range(len(labels))])
    overall = values.std(axis=(0, 1), dtype=np.float64)
    overall[overall <= 0] = 1.0
    return labels, mean, std, overall


def standardised(values, keys, labels, mean, spread):
    """Yield the standardised values of one bin at a time [timestamps of the bin, cells, variables]."""
    for row, label in enumerate(labels):
        chosen = keys == label
        if chosen.any():
            yield (values[chosen] - mean[row]) / spread[row]


def measure(train, train_keys, scored, scored_keys, fitted, floor):
    """Per variable: what the docstring lists, for one floor."""
    labels, mean, std, overall = fitted
    spread = np.maximum(std, floor * overall)
    low, high = np.full(len(overall), np.inf), np.full(len(overall), -np.inf)
    squares = count = 0
    for block in standardised(train, train_keys, labels, mean, spread):
        low, high = np.minimum(low, block.min(axis=(0, 1))), np.maximum(high, block.max(axis=(0, 1)))
        squares, count = squares + (block**2).sum(axis=(0, 1)), count + block.shape[0] * block.shape[1]
    largest = np.maximum(np.abs(low), np.abs(high))
    beyond = scored_squares = scored_count = 0
    scored_largest = np.zeros(len(overall))
    for block in standardised(scored, scored_keys, labels, mean, spread):
        beyond = beyond + ((block < low) | (block > high)).sum(axis=(0, 1))
        scored_largest = np.maximum(scored_largest, np.abs(block).max(axis=(0, 1)))
        scored_squares, scored_count = scored_squares + (block**2).sum(axis=(0, 1)), scored_count + block.shape[0] * block.shape[1]
    energy = (scored_squares / scored_count) / (high - low)**2  # after the min-max, up to the same factor for all
    return {"at the floor %": 100 * (std < floor * overall).mean(axis=(0, 1)), "low": low, "high": high,
            "beyond %": 100 * beyond / scored_count, "largest ratio": scored_largest / largest,
            "train square": squares / count, "scored square": scored_squares / scored_count,
            "share %": 100 * energy / energy.sum()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--features", nargs="+", default=list(FEATURES))
    parser.add_argument("--floors", type=float, nargs="+", default=list(FLOORS))
    parser.add_argument("--time-stride", type=int, default=1, help="Training timestamps read: one every this many")
    parser.add_argument("--scored-stride", type=int, default=9, help="Scored timestamps read: one every this many")
    parser.add_argument("--cell-stride", type=int, default=10)
    args = parser.parse_args(argv)
    if min(args.time_stride, args.scored_stride, args.cell_stride) < 1 or args.time_stride % 2 == 0 or args.scored_stride % 2 == 0:
        raise ValueError("The strides must be at least 1 and those in time odd")
    if not args.floors or min(args.floors) <= 0:
        raise ValueError("The floors must be positive")
    cubes = load_prepared(args.prepared_dir)[0]
    try:
        unknown = [name for name in args.features if name not in cubes.feature_names]
        if unknown:
            raise ValueError(f"Unknown variables: {unknown}; the cache has {list(cubes.feature_names)}")
        columns = [cubes.feature_names.index(name) for name in args.features]
        cells = np.arange(0, cubes.train.shape[1], args.cell_stride)
        train, train_times = sample(cubes.train, cubes.train_timestamps, args.time_stride, cells, columns)
        scored, scored_times = sample(cubes.test, cubes.test_timestamps, args.scored_stride, cells, columns)
    finally:
        cubes.close()
    train_keys, scored_keys = bins(train_times), bins(scored_times)
    fitted = fit_bins(train, train_keys)
    results = {floor: measure(train, train_keys, scored, scored_keys, fitted, floor) for floor in args.floors}
    print(f"[floor] normal: {train_times[0].year}-{train_times[-1].year}, {len(train_times)} timestamps | scored: "
          f"{scored_times[0].year}-{scored_times[-1].year}, {len(scored_times)} timestamps | cells {len(cells)}")
    for index, name in enumerate(args.features):
        print(f"\n{name}\n{'floor':>6}{'at the floor %':>16}{'training range':>20}{'scored beyond %':>17}"
              f"{'scored largest / training':>27}{'mean square, training':>23}{'scored':>9}")
        for floor, result in results.items():
            print(f"{floor:>6g}{result['at the floor %'][index]:>16.1f}"
                  f"{result['low'][index]:>+10.1f} ..{result['high'][index]:>+7.1f}{result['beyond %'][index]:>17.4f}"
                  f"{result['largest ratio'][index]:>27.1f}{result['train square'][index]:>23.2f}"
                  f"{result['scored square'][index]:>9.2f}")
    print(f"\nshare of every variable in the mean square of the scored years after the min-max, %\n{'floor':>6}"
          + "".join(f"{name[:13]:>15}" for name in args.features))
    for floor, result in results.items():
        print(f"{floor:>6g}" + "".join(f"{value:>15.1f}" for value in result["share %"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
