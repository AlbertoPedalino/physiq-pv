"""How should the seasonal normalization be set so that unseen years stay on the scale of the training ones?

No model is used. On a sample of the cells, the seasonal normal is fitted on the training years
with the code of the pipeline (`fit_seasonal_climatology`: day of the year with its window, time
of day) and every variable is then standardised in the training years, in the validation year
and in the scored years, with several floors of the standard deviation. The floor is a share of
the standard deviation of the variable as a whole, used where that of a cell, time of year and
time of day is smaller.

For every variable and floor:

- `own spread replaced`: the share of cells and times whose own, non-zero standard deviation is
  replaced by the floor. That is the cost of the floor: there the normalization no longer follows
  the place and the season. `no spread` is the share where the training years never vary, which
  any floor replaces;
- the range of the standardised training values, which the min-max maps to the range of the model;
- for the validation year and for the scored years: the mean square of their standardised values
  over that of the training years (1 if they are on one scale), their largest size over the
  largest of the training years, the share of values beyond the training range, and the same
  mean square once the values are cut at the training range.

The precipitation is also read after a transform applied before the normalization, as a variable
of its own: its square root, its fourth root and log(1 + P / e), with e the median of the
training values above 0. Almost always 0 with rare large values, it is the variable a standard
deviation describes worst.

The floor a variable needs is the smallest that keeps the validation year within RATIO of the
training mean square and within LARGEST of its largest value: the validation year chooses, the
scored years only confirm. Last, for one floor, the drift of the scored years and what a line in
time fitted on the training years leaves of it, for every variable.

    python scripts/check_era5_seasonal_normalization.py --prepared-dir data/era5_prepared
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from physiq_pv.anomaly_detection.stgan.seasonal import _bins, fit_seasonal_climatology
from physiq_pv.era5.data import load_prepared

RAIN = "total_precipitation"
TRANSFORMS = ("square root", "fourth root", "log")
FLOORS = (0.01, 0.03, 0.1, 0.3)
RATIO, LARGEST = 1.5, 3.0  # what a floor must keep the validation year within
SEASONS = {"all months": range(1, 13), "June to August": (6, 7, 8), "December to February": (12, 1, 2)}
LAST_YEARS = 5
CHUNK = 256  # timestamps standardised at a time


def read(array, timestamps, stride, cells, extra):
    """Values [timestamps, cells, variables + extra] of one timestamp every `stride`, the last columns empty."""
    positions = np.arange(0, len(array), stride)
    values = np.empty((len(positions), len(cells), array.shape[2] + extra), dtype=np.float32)
    for index, position in enumerate(positions):
        values[index, :, :array.shape[2]] = np.asarray(array[position:position + 1])[0][cells]
    return values, timestamps[positions]


def add_transforms(values, rain, epsilon):
    """Fill the last columns with the precipitation transformed: square root, fourth root, log(1 + P / epsilon)."""
    amount = np.maximum(values[..., rain], 0)
    values[..., -3] = np.sqrt(amount)
    values[..., -2] = np.sqrt(np.sqrt(amount))
    values[..., -1] = np.log1p(amount / epsilon)


def blocks(values, timestamps, climatology, spread):
    """Yield the standardised values [timestamps of the block, cells, variables] and where the block starts."""
    day, slot = _bins(timestamps, climatology.slots)
    for start in range(0, len(values), CHUNK):
        key = slot[start:start + CHUNK], day[start:start + CHUNK]
        yield start, (values[start:start + CHUNK] - climatology.mean[key]) / spread[key]


def measure(parts, climatology, overall, floor, keep_means=False):
    """What the docstring lists for one floor: a dict of arrays by variable, per period where it applies."""
    std = np.asarray(climatology.std)
    limit = floor * overall
    spread = np.maximum(std, limit).astype(np.float32)
    none = std <= 1e-6 * overall  # the training years never vary there
    result = {"replaced %": 100 * ((std < limit) & ~none).mean(axis=(0, 1, 2)), "none %": 100 * none.mean(axis=(0, 1, 2))}
    low, high = np.full(len(overall), np.inf), np.full(len(overall), -np.inf)
    for name, (values, timestamps) in parts.items():
        squares, cut, beyond, largest, count = (np.zeros(len(overall)) for _ in range(5))
        means = np.empty((len(values), len(overall))) if keep_means else None
        for start, block in blocks(values, timestamps, climatology, spread):
            if name == "training":
                low, high = np.minimum(low, block.min(axis=(0, 1))), np.maximum(high, block.max(axis=(0, 1)))
            else:
                beyond += ((block < low) | (block > high)).sum(axis=(0, 1))
                cut += (np.clip(block, low, high).astype(np.float64)**2).sum(axis=(0, 1))
            squares += (block.astype(np.float64)**2).sum(axis=(0, 1))
            largest = np.maximum(largest, np.abs(block).max(axis=(0, 1)))
            count += block.shape[0] * block.shape[1]
            if keep_means:
                means[start:start + len(block)] = block.mean(axis=1)
        result[name] = {"square": squares / count, "largest": largest, "beyond %": 100 * beyond / count,
                        "cut": cut / count, "means": means}
    result["low"], result["high"] = low, high
    return result


def needed_floor(results, index):
    """The smallest floor that keeps the validation year within RATIO and LARGEST of the training years; None if none."""
    for floor, result in results.items():
        training, validation = result["training"], result["validation"]
        if (validation["square"][index] <= RATIO * training["square"][index]
                and validation["largest"][index] <= LARGEST * training["largest"][index]):
            return floor
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--floors", type=float, nargs="+", default=list(FLOORS))
    parser.add_argument("--trend-floor", type=float, help="Floor at which the drift is read; default: the middle one")
    parser.add_argument("--window-days", type=int, default=15)
    parser.add_argument("--time-stride", type=int, default=1, help="Training timestamps read: one every this many")
    parser.add_argument("--scored-stride", type=int, default=3, help="Scored timestamps read: one every this many")
    parser.add_argument("--cell-stride", type=int, default=10)
    args = parser.parse_args(argv)
    floors = sorted(args.floors)
    if min(args.time_stride, args.scored_stride, args.cell_stride) < 1 or args.time_stride % 2 == 0 or args.scored_stride % 2 == 0:
        raise ValueError("The strides must be at least 1 and those in time odd")
    if not floors or floors[0] <= 0:
        raise ValueError("The floors must be positive")
    trend_floor = floors[len(floors) // 2] if args.trend_floor is None else args.trend_floor
    if trend_floor not in floors:
        raise ValueError("trend-floor must be one of the floors")
    cubes = load_prepared(args.prepared_dir)[0]
    try:
        names = [*cubes.feature_names, *(f"{RAIN}, {transform}" for transform in TRANSFORMS)]
        rain = cubes.feature_names.index(RAIN)
        cells = np.arange(0, cubes.train.shape[1], args.cell_stride)
        parts = {"training": read(cubes.train, cubes.train_timestamps, args.time_stride, cells, len(TRANSFORMS)),
                 "validation": read(cubes.validation, cubes.validation_timestamps, 1, cells, len(TRANSFORMS)),
                 "scored": read(cubes.test, cubes.test_timestamps, args.scored_stride, cells, len(TRANSFORMS))}
    finally:
        cubes.close()
    wet = parts["training"][0][..., rain]
    epsilon = float(np.median(wet[wet > 0]))
    del wet
    for values, _ in parts.values():
        add_transforms(values, rain, epsilon)
    train, train_times = parts["training"]
    overall = np.array([train[..., index].std(dtype=np.float64) for index in range(train.shape[2])])
    overall[overall <= 0] = 1.0
    with tempfile.TemporaryDirectory() as directory:
        # The floor of the fit is as good as none: the floors are applied here, on the same mean and spread.
        climatology = fit_seasonal_climatology(train, train_times, directory, window_days=args.window_days, std_floor=1e-12)
        try:
            results = {floor: measure(parts, climatology, overall, floor, keep_means=floor == trend_floor)
                       for floor in floors}
        finally:
            climatology.close()
    print(f"[normalization] normal: {train_times[0].year}-{train_times[-1].year}, {len(train_times)} timestamps, window of "
          f"{args.window_days} days each side | validation {parts['validation'][1][0].year}, scored "
          f"{parts['scored'][1][0].year}-{parts['scored'][1][-1].year} | cells {len(cells)} | log: e = {epsilon:.3g}, the "
          f"median of the training precipitation above 0")
    header = (f"{'floor':>6}{'own spread':>12}{'no':>9}{'training range':>19} |{'validation':>12}{'':>18} |{'scored':>8}"
              f"{'':>31}\n{'':>6}{'replaced %':>12}{'spread %':>9}{'':>19} |{'square':>8}{'largest':>9}{'beyond %':>10}"
              f"{'if cut':>8} |{'square':>8}{'largest':>9}{'beyond %':>10}{'if cut':>8}")
    for index, name in enumerate(names):
        print(f"\n{name}\n{header}")
        for floor, result in results.items():
            training = result["training"]
            columns = "".join(
                f" |{period['square'][index] / training['square'][index]:>8.2f}"
                f"{period['largest'][index] / training['largest'][index]:>9.1f}{period['beyond %'][index]:>10.4f}"
                f"{period['cut'][index] / training['square'][index]:>8.2f}"
                for period in (result["validation"], result["scored"]))
            print(f"{floor:>6g}{result['replaced %'][index]:>12.1f}{result['none %'][index]:>9.1f}"
                  f"{result['low'][index]:>+9.1f} ..{result['high'][index]:>+7.1f}{columns}")
    print(f"\nthe smallest floor that keeps the validation year within {RATIO:g} of the training mean square and "
          f"{LARGEST:g} of its largest value, and its cost")
    for index, name in enumerate(names):
        floor = needed_floor(results, index)
        print(f"{name:<40}" + ("none of these floors" if floor is None else
                               f"{floor:>6g}   own spread replaced {results[floor]['replaced %'][index]:.1f}%"))
    # The drift and a line in time fitted on the training years, at one floor.
    result = results[trend_floor]
    tables = {name: pd.DataFrame(result[name]["means"], index=parts[name][1]) for name in parts}
    yearly = tables["training"].groupby(tables["training"].index.year).mean()
    time = yearly.index.to_numpy() + 0.5
    middle = time.mean()
    slope = ((time - middle)[:, None] * (yearly.to_numpy() - yearly.to_numpy().mean(axis=0))).sum(axis=0) / ((time - middle)**2).sum()
    scored = tables["scored"]
    late = scored[scored.index.year > scored.index.year.max() - LAST_YEARS]
    years = late.index.year.to_numpy() + (late.index.dayofyear.to_numpy() - 1) / 365.25
    left = late - np.outer(years - middle, slope)
    print(f"\ndrift at floor {trend_floor:g}: mean distance of the last {LAST_YEARS} scored years from the normal, in its "
          f"standard deviations, as it is and with a line fitted on the training years taken away\n{'':<40}"
          f"{'slope by year':>14}" + "".join(f"{season[:12]:>14}{'line away':>11}" for season in SEASONS))
    for index, name in enumerate(names):
        print(f"{name:<40}{slope[index]:>+14.4f}" + "".join(
            f"{late[late.index.month.isin(list(months))][index].mean():>+14.2f}"
            f"{left[left.index.month.isin(list(months))][index].mean():>+11.2f}" for months in SEASONS.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
