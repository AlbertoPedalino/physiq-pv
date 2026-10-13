"""Do the floors of the seasonal standard deviation follow from the training years alone?

No model is used and the scored years are not read. The training years are those of the thesis
protocol: the year the prepared data keep apart as validation is one of them, unless
`--without-validation-year` is given. They are split in two: the seasonal normal is fitted on the first ones with the code of the
pipeline, and the last `--held-out-years` are standardised with it, as years it has not seen. For
every variable the floor needed is the smallest that keeps the held-out years within RATIO of the
mean square of the fitted years and within LARGEST of their largest value: the rule of
`check_era5_seasonal_normalization.py`, with held-out training years in the place of the
validation year. The precipitation is read as its fourth root, as the run uses it.

    python scripts/check_era5_floor_on_training_years.py --prepared-dir data/era5_prepared
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import numpy as np

from physiq_pv.anomaly_detection.stgan.seasonal import fit_seasonal_climatology
from physiq_pv.era5.data import load_prepared
from scripts.check_era5_seasonal_normalization import FLOORS, LARGEST, RAIN, RATIO, measure, needed_floor, read


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--held-out-years", type=int, default=10, help="Last training years kept out of the normal")
    parser.add_argument("--without-validation-year", action="store_true",
                        help="Leave out the year the prepared data keep apart as validation")
    parser.add_argument("--floors", type=float, nargs="+", default=list(FLOORS))
    parser.add_argument("--window-days", type=int, default=15)
    parser.add_argument("--time-stride", type=int, default=1, help="Training timestamps read: one every this many")
    parser.add_argument("--cell-stride", type=int, default=10)
    args = parser.parse_args(argv)
    floors = sorted(args.floors)
    if min(args.time_stride, args.cell_stride) < 1 or args.time_stride % 2 == 0:
        raise ValueError("The strides must be at least 1 and that in time odd")
    if not floors or floors[0] <= 0:
        raise ValueError("The floors must be positive")
    cubes = load_prepared(args.prepared_dir)[0]
    try:
        names = list(cubes.feature_names)
        cells = np.arange(0, cubes.train.shape[1], args.cell_stride)
        values, times = read(cubes.train, cubes.train_timestamps, args.time_stride, cells, 0)
        last = (None if args.without_validation_year else
                read(cubes.validation, cubes.validation_timestamps, args.time_stride, cells, 0))
    finally:
        cubes.close()
    rain = names.index(RAIN)
    for block in (values,) if last is None else (values, last[0]):
        block[..., rain] = np.sqrt(np.sqrt(np.maximum(block[..., rain], 0)))
    names[rain] = f"{RAIN}, fourth root"
    last_year = int(times.year.max() if last is None else last[1].year.max())
    split = int(np.searchsorted(np.asarray(times.year), last_year - args.held_out_years + 1))
    years = np.asarray(times.year)
    if args.held_out_years < 1 or len(np.unique(years[:split])) < 2:
        raise ValueError("held-out-years must leave at least two years to fit the normal")
    # The timestamps are in order: the fitted years are a view, only the held-out ones may be copied.
    held = ((values[split:], times[split:]) if last is None else
            (np.concatenate((values[split:], last[0])), times[split:].append(last[1])))
    parts = {"training": (values[:split], times[:split]), "validation": held}
    fitted = parts["training"][0]
    overall = np.array([fitted[..., index].std(dtype=np.float64) for index in range(fitted.shape[2])])
    overall[overall <= 0] = 1.0
    with tempfile.TemporaryDirectory() as directory:
        # The floor of the fit is as good as none: the floors are applied here, on the same mean and spread.
        climatology = fit_seasonal_climatology(fitted, parts["training"][1], directory, window_days=args.window_days,
                                               std_floor=1e-12)
        try:
            results = {floor: measure(parts, climatology, overall, floor) for floor in floors}
        finally:
            climatology.close()
    print(f"[floors] normal fitted on {years[0]}-{years[split - 1]}, {split} timestamps | held out "
          f"{held[1][0].year}-{held[1][-1].year}, {len(held[1])} timestamps | cells {len(cells)} | window of "
          f"{args.window_days} days each side | scored years not read")
    print(f"needed: the smallest floor that keeps the held-out years within {RATIO:g} of the mean square of the fitted "
          f"years and {LARGEST:g} of their largest value\nat each floor: mean square / largest value of the held-out "
          f"years over those of the fitted years")
    print(f"{'variable':<40}{'needed':>8}" + "".join(f"{f'floor {floor:g}':>16}" for floor in floors))
    for index, name in enumerate(names):
        needed = needed_floor(results, index)
        columns = "".join(
            f"{result['validation']['square'][index] / result['training']['square'][index]:>9.2f} /"
            f"{result['validation']['largest'][index] / result['training']['largest'][index]:>5.1f}"
            for result in results.values())
        print(f"{name:<40}{'none' if needed is None else format(needed, 'g'):>8}{columns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
