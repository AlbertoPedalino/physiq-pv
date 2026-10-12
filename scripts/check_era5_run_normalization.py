"""The normalization a run will make, read on a sample of the cells before the run is started.

It takes the normalization options of `run_era5_stgan.py train` and makes, with the code of the
pipeline, what the run makes before training: the fourth root of the named variables, the
seasonal normal of the training years with its floors, the line in time of the named variables,
the standardised training, validation and scored years, the min-max of the training years to the
range of the model [-1, 1], and the daily-change weights. Nothing is trained.

For every variable it then says:

- what was applied to it, and the range of its standardised training values;
- whether the validation year and the scored years are on the scale of the training years: the
  mean square of their standardised values over that of the training years, their largest size
  over the largest of the training years, the share of values beyond the range of the model and
  the largest size that reaches the model (1 in the training years);
- the drift that is left: the mean distance of the last scored years from the normal;
- how much of the reconstruction error it is expected to make. The daily-change weights rest on
  the mean square of the change of a variable over 24 hours, in the units of the model: its share
  of the sum of those mean squares is read in the training years, in the scored years, and in the
  scored years with the weights of the training years, where every variable has the same share
  if the scored years behave as the training ones.

A variable is marked `!!` where the validation year or the scored years leave the scale: more
than RATIO of the training mean square or LARGEST of its largest value.

    python scripts/check_era5_run_normalization.py --prepared-dir data/era5_prepared \
        --seasonal-std-floors total_precipitation=0.1 --seasonal-fourth-root-features total_precipitation \
        --seasonal-trend linear --seasonal-trend-features 2m_temperature temperature
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from physiq_pv.anomaly_detection.stgan.pipeline import _daily_change_weights, _feature_minmax
from physiq_pv.anomaly_detection.stgan.seasonal import FourthRoot, fit_seasonal_climatology, fit_seasonal_trend, seasonal_memmap
from physiq_pv.era5.data import load_prepared
from scripts.run_era5_stgan import floors_by_variable

RATIO, LARGEST = 1.5, 3.0  # beyond these a period has left the scale of the training years
SEASONS = {"all months": range(1, 13), "June to August": (6, 7, 8), "December to February": (12, 1, 2)}
LAST_YEARS = 5
STEPS_PER_DAY = 8
CHUNK = 512  # timestamps read at a time
NAME = 34


def read(array, timestamps, cells):
    """Values [timestamps, cells, variables] of a sample of the cells, in memory."""
    values = np.empty((len(array), len(cells), array.shape[2]), dtype=np.float32)
    for start in range(0, len(array), CHUNK):
        values[start:start + CHUNK] = np.asarray(array[start:start + CHUNK])[:, cells]
    return values, timestamps


def measure(standardised, minimum, scale):
    """Per variable: mean square and largest size of the standardised values, share beyond the range of the
    model and largest size in it; and the mean over the cells at every timestamp [timestamps, variables]."""
    squares, largest, beyond, reaching = (np.zeros(len(scale)) for _ in range(4))
    means = np.empty((len(standardised), len(scale)))
    for start in range(0, len(standardised), CHUNK):
        block = np.asarray(standardised[start:start + CHUNK], dtype=np.float64)
        model = (block - minimum) / scale * 2.0 - 1.0
        squares += (block**2).sum(axis=(0, 1))
        largest = np.maximum(largest, np.abs(block).max(axis=(0, 1)))
        beyond += (np.abs(model) > 1 + 1e-6).sum(axis=(0, 1))
        reaching = np.maximum(reaching, np.abs(model).max(axis=(0, 1)))
        means[start:start + len(block)] = block.mean(axis=1)
    count = standardised.shape[0] * standardised.shape[1]
    return {"square": squares / count, "largest": largest, "beyond %": 100 * beyond / count, "reaching": reaching,
            "means": means}


def parser():
    root = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    root.add_argument("--prepared-dir", type=Path, required=True)
    root.add_argument("--seasonal-window-days", type=int, default=15)
    root.add_argument("--seasonal-std-floor", type=float, default=0.01)
    root.add_argument("--seasonal-std-floors", nargs="+", metavar="VARIABLE=FLOOR")
    root.add_argument("--seasonal-fourth-root-features", nargs="+")
    root.add_argument("--seasonal-trend", choices=("none", "linear"), default="none")
    root.add_argument("--seasonal-trend-features", nargs="+")
    root.add_argument("--cell-stride", type=int, default=10, help="One cell every this many is read")
    root.add_argument("--work-dir", type=Path, help="Where the sample is written while it is read; default: next to the cache")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.cell_stride < 1:
        raise ValueError("cell-stride must be at least 1")
    cubes = load_prepared(args.prepared_dir)[0]
    try:
        names = list(cubes.feature_names)
        floors = dict(floors_by_variable(args.seasonal_std_floors) or ())
        roots, lines = args.seasonal_fourth_root_features or [], args.seasonal_trend_features
        unknown = [name for name in (*floors, *roots, *(lines or [])) if name not in names]
        if unknown:
            raise ValueError(f"Unknown variables: {unknown}; the cache has {names}")
        if lines is not None and args.seasonal_trend != "linear":
            raise ValueError("--seasonal-trend-features needs --seasonal-trend linear")
        cells = np.arange(0, cubes.train.shape[1], args.cell_stride)
        parts = {"training": read(cubes.train, cubes.train_timestamps, cells),
                 "validation": read(cubes.validation, cubes.validation_timestamps, cells),
                 "scored": read(cubes.test, cubes.test_timestamps, cells)}
    finally:
        cubes.close()
    std_floor = [float(floors.get(name, args.seasonal_std_floor)) for name in names]
    columns = [names.index(name) for name in roots]
    rooted = lambda array: FourthRoot(array, columns) if columns else array
    train, train_times = parts["training"]
    work = args.work_dir or args.prepared_dir.resolve().parent
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as directory:
        directory = Path(directory)
        # As the pipeline does: the normal, the line, the three periods on them, the min-max, the weights.
        climatology = fit_seasonal_climatology(rooted(train), train_times, directory, window_days=args.seasonal_window_days,
                                               std_floor=std_floor)
        written = {}
        try:
            trend = fit_seasonal_trend(rooted(train), train_times, climatology) if args.seasonal_trend == "linear" else None
            if trend is not None and lines is not None:
                trend = {**trend, "slope_per_year": [slope if name in lines else 0.0
                                                     for name, slope in zip(names, trend["slope_per_year"])]}
            for name, (values, timestamps) in parts.items():
                written[name] = seasonal_memmap(rooted(values), timestamps, climatology, directory / f"{name}.npy", trend=trend)
            minimum, scale = _feature_minmax(written["training"])
            weights = _daily_change_weights(written["training"], scale, STEPS_PER_DAY)
            scored_weights = _daily_change_weights(written["scored"], scale, STEPS_PER_DAY)
            measured = {name: measure(array, minimum, scale) for name, array in written.items()}
        finally:
            climatology.close()
            for array in written.values():
                array._mmap.close()
    slopes = [0.0] * len(names) if trend is None else trend["slope_per_year"]
    times = {name: timestamps for name, (_, timestamps) in parts.items()}
    print(f"[run-normalization] normal: {train_times[0].year}-{train_times[-1].year}, window of {args.seasonal_window_days} "
          f"days each side | validation {times['validation'][0].year} | scored {times['scored'][0].year}-"
          f"{times['scored'][-1].year} | cells {len(cells)} of {len(cells) * args.cell_stride}")
    training = measured["training"]
    print(f"\nwhat is applied, and the scale of the validation year and of the scored years against the training ones\n"
          f"{'':<{NAME}}{'root':>5}{'floor':>7}{'line/year':>11}{'training range':>18} |{'validation':>12}{'':>24}"
          f" |{'scored':>8}\n{'':<{NAME}}{'':>5}{'':>7}{'':>11}{'':>18} |{'square':>8}{'largest':>9}{'beyond %':>10}"
          f"{'in model':>9} |{'square':>8}{'largest':>9}{'beyond %':>10}{'in model':>9}")
    for index, name in enumerate(names):
        row, marked = "", False
        for period in (measured["validation"], measured["scored"]):
            square = period["square"][index] / training["square"][index]
            largest = period["largest"][index] / training["largest"][index]
            marked = marked or square > RATIO or largest > LARGEST
            row += f" |{square:>8.2f}{largest:>9.1f}{period['beyond %'][index]:>10.4f}{period['reaching'][index]:>9.1f}"
        print(f"{name:<{NAME}}{'yes' if index in columns else '':>5}{std_floor[index]:>7g}{slopes[index]:>+11.4f}"
              f"{minimum[index]:>+9.1f} ..{minimum[index] + scale[index]:>+6.1f}{row}{'  !!' if marked else ''}")
    scored = pd.DataFrame(measured["scored"]["means"], index=times["scored"])
    late = scored[scored.index.year > scored.index.year.max() - LAST_YEARS]
    print(f"\ndrift left: mean distance of the last {LAST_YEARS} scored years from the normal, in its standard deviations\n"
          f"{'':<{NAME}}" + "".join(f"{season:>22}" for season in SEASONS))
    for index, name in enumerate(names):
        print(f"{name:<{NAME}}" + "".join(f"{late[late.index.month.isin(list(months))][index].mean():>+22.2f}"
                                          for months in SEASONS.values()))
    change, scored_change = 1 / weights, 1 / scored_weights
    shares = [100 * values / values.sum() for values in (change, scored_change, weights * scored_change)]
    print(f"\nexpected share of the reconstruction error: the mean square of the change over 24 hours in the units of the "
          f"model, % of the sum over the variables ({100 / len(names):.1f} each if equal)\n{'':<{NAME}}{'weight':>9}"
          f"{'training':>11}{'scored':>9}{'scored, weighted':>18}")
    for index, name in enumerate(names):
        print(f"{name:<{NAME}}{weights[index]:>9.3g}" + "".join(f"{share[index]:>{width}.1f}"
                                                               for share, width in zip(shares, (11, 9, 18))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
