"""Is what a score flags unusual in a variable the score never saw?

A score built on some variables flags a value; whether the weather there was unusual is asked to
variables that are in none of the scores, downloaded apart (`download_era5.py
--independent-variables`: the 10 m wind gust and the convective available potential energy). Such
a variable is unusual at a cell and timestamp when it is in its highest `--unusual-percent` among
the scored years for that cell, calendar month and time of day: the same reading of "unusual for
the place and the season" the scores have.

A value is flagged by a score when it is in its highest `--top-percent`. The values are in groups:
flagged by the first score and by none of the others, by the first score and at least one other, by
the others and not by the first, by none. For every group the check says in how many of its values
each unseen variable is unusual: about `--unusual-percent` if what the group flags has nothing to
do with that variable, more if the group flags weather that is unusual there too. The other way
round, it says how many of the values where an unseen variable is unusual each score flags.

The same question can be asked to a variable the scores did see, with `--run-dir` and `--variables`
in the place of `--unseen-dir`: the values are then the standardised ones of the shared cache of that
run. With `--tail low` the unusual values are the lowest ones: a low solar radiation, for instance.
Where a variable does not change at all for a cell, month and time of day, as the solar radiation at
night, it cannot be unusual, and those values are left out of its shares.

With `--near-normal-run-dir` only the values are counted where no variable of that run is more than
`--far` standard deviations of the training years from its seasonal normal: the weather that a
distance from the normal does not tell from any other. The last column of the second table is then
the share of those values each score flags, to compare the others with.

With `--split-by-spread` the flags of the first score are also split by the spread of its dropout
draws, which the run of that score keeps: solid when the score stays above the threshold one spread
lower, at the limit otherwise. Then the same shares, for the flags of the first score alone and for
those it shares with another score: if the spread tells how much a flag can be trusted, the solid
ones are unusual more often.

The whole scored period is read, one calendar month at a time. Nothing is written.

    python scripts/check_era5_score_against_unseen.py --unseen-dir data/era5_independent/independent \
        --score-dir outputs/run_clim --by-dirs outputs/run_no_model_clim outputs/run_previous_clim
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

from physiq_pv.anomaly_detection.stgan.shared_cache import MANIFEST
from scripts.check_era5_dust_lead import COMPONENTS, THRESHOLD_STRIDE, score_file

GROUPS = ("score alone, none of the others", "score and at least one other", "others, not the score", "flagged by none",
          "all values")
ALL = GROUPS.index("all values")
SPLIT = ("score alone, solid", "score alone, at the limit", "with another, solid", "with another, at the limit")
NAME, WIDTH = 36, 14


def read_month(root, year, month, latitudes, longitudes):
    """Timestamps and values [timestamps, cells, variables] of the unseen variables of one month, with their names.
    `latitudes` and `longitudes` are those of the cells."""
    import xarray as xr

    paths = sorted((root / str(year)).glob(f"*_{year}_{month:02d}*.nc"))
    if not paths:
        raise ValueError(f"No file of the unseen variables for {year}-{month:02d} under {root}")
    times, fields = None, {}
    for path in paths:
        with xr.open_dataset(path) as dataset:
            time = "valid_time" if "valid_time" in dataset.coords else "time"
            found = pd.DatetimeIndex(dataset[time].values)
            if times is not None and not found.equals(times):
                raise ValueError(f"The files of {year}-{month:02d} are not over the same timestamps")
            times = found
            axes = []
            for name, wanted in (("latitude", latitudes), ("longitude", (longitudes + 180) % 360 - 180)):
                axis = dataset[name].values.astype(float)
                axis = (axis + 180) % 360 - 180 if name == "longitude" else axis
                nearest = np.abs(axis[None, :] - wanted[:, None]).argmin(axis=1)
                if np.abs(axis[nearest] - wanted).max() > 1e-3:
                    raise ValueError(f"{path.name}: its {name} axis does not hold the cells of the score")
                axes.append(nearest)
            for name, field in dataset.data_vars.items():
                if {time, "latitude", "longitude"} <= set(field.dims):
                    values = field.transpose(time, "latitude", "longitude", ...).values
                    fields[name] = values.reshape(values.shape[:3])[:, axes[0], axes[1]].astype(np.float32)
    if not fields:
        raise ValueError(f"The files of {year}-{month:02d} hold no variable over time, latitude and longitude")
    return times, np.stack([fields[name] for name in sorted(fields)], axis=2), sorted(fields)


def read_month_of_the_run(standardised, first, rows, columns):
    """Values [timestamps, cells, variables] of the variables `columns` of the cache at the scored timestamps
    `rows`, which are in order. `first` is the cache row of the first scored timestamp."""
    blocks = np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)  # a calendar month is one block a year
    return np.concatenate([np.asarray(standardised[first + block[0]:first + block[-1] + 1][:, :, columns], dtype=np.float32)
                           for block in blocks])


def cache_of(run_dir, timestamps, cells):
    """The standardised scored period of the shared cache of `run_dir`, the cache row of the first scored
    timestamp and the names of the variables."""
    manifest = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    seasonal = manifest.get("backend", {}).get("seasonal_normalization")
    if not seasonal:
        raise ValueError("The run has no seasonal normalization: there is no cache of standardised data")
    cache = Path(seasonal["mean_file"]).parent
    if not (cache / MANIFEST).is_file() or not (cache / "test.npy").is_file():
        raise ValueError(f"The shared cache of the run is not there: {cache}")
    standardised = np.load(cache / "test.npy", mmap_mode="r")
    first = len(standardised) - len(timestamps)  # the cache holds a few timestamps of context before
    if first < 0 or standardised.shape[1] != cells:
        standardised._mmap.close()
        raise ValueError("The cache and the score are not over the same period and cells")
    return standardised, first, list(manifest.get("preparation", {}).get("features") or [])


def near_normal(standardised, first, rows, far):
    """Where no variable of the cache is more than `far` from its seasonal normal, at the scored timestamps
    `rows`, which are in order: [timestamps, cells]."""
    blocks = np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)
    return np.concatenate([(np.abs(np.asarray(standardised[first + block[0]:first + block[-1] + 1])) <= far).all(axis=2)
                           for block in blocks])


def unusual_values(values, hours, percent, low=False):
    """Where each variable [timestamps, cells, variables] is in its highest `percent`, or with `low` its lowest,
    among the timestamps of the same time of day, cell by cell; and where it changes at all there, so that it
    can be unusual."""
    unusual, defined = np.zeros(values.shape, dtype=bool), np.zeros(values.shape, dtype=bool)
    for hour in np.unique(hours):
        chosen = hours == hour
        block = values[chosen]
        unusual[chosen] = (block < np.percentile(block, percent, axis=0) if low else
                           block > np.percentile(block, 100 - percent, axis=0))
        defined[chosen] = block.max(axis=0) > block.min(axis=0)
    return unusual, defined


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--unseen-dir", type=Path, help="Directory with a folder of monthly NetCDF files per year")
    parser.add_argument("--run-dir", type=Path, help="Training run with the seasonal normalization: its cache is read instead")
    parser.add_argument("--variables", nargs="+", help="With --run-dir: the variables of the run to ask")
    parser.add_argument("--tail", choices=("high", "low"), default="high", help="Unusual: the highest or the lowest values")
    parser.add_argument("--near-normal-run-dir", type=Path,
                        help="Count only the values where no variable of this run is more than --far from its seasonal normal")
    parser.add_argument("--far", type=float, default=2.0, help="In standard deviations of the training years")
    parser.add_argument("--split-by-spread", action="store_true",
                        help="Split the flags of the first score by the spread of its dropout draws")
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the score that is asked about")
    parser.add_argument("--by-dirs", type=Path, nargs="+", required=True, help="Runs with the other scores, over the same period")
    parser.add_argument("--score-component", choices=COMPONENTS, default="combined",
                        help="Part of the first score to read, for a training run: the reconstruction error (generator), the discriminator part, or their sum")
    parser.add_argument("--top-percent", type=float, default=5, help="Flagged: this highest share of each score")
    parser.add_argument("--unusual-percent", type=float, help="Unusual: this highest share of an unseen variable; default: top-percent")
    parser.add_argument("--years", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="Read only these scored years, for instance those downloaded so far; the unusual values "
                             "are then those of these years, the thresholds of the scores stay those of the whole period")
    args = parser.parse_args(argv)
    unusual_percent = args.top_percent if args.unusual_percent is None else args.unusual_percent
    if (args.unseen_dir is None) == (args.run_dir is None) or (args.run_dir is not None) != bool(args.variables):
        raise ValueError("Give --unseen-dir, or --run-dir with --variables")
    kind = "an unseen variable" if args.run_dir is None else "a variable of the run"
    paths = [args.score_dir, *args.by_dirs]
    common = os.path.commonprefix([path.name for path in paths])
    labels = [path.name[len(common):] or path.name for path in paths]
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    latitudes = np.asarray(grid["latitudes"], dtype=float)[np.asarray(grid["rows"])]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[np.asarray(grid["cols"])]
    timestamps = pd.DatetimeIndex(np.load(args.score_dir / "test_timestamps.npy"))
    first_year, last_year = args.years or (timestamps.year.min(), timestamps.year.max())
    wanted = (timestamps.year >= first_year) & (timestamps.year <= last_year)
    if not wanted.any():
        raise ValueError(f"The score has no timestamp in {first_year}-{last_year}")
    opened, names = [], None
    names_of_groups = list(GROUPS) + (list(SPLIT) if args.split_by_spread else [])
    count, hits, spread_of_first = np.zeros(len(names_of_groups)), None, None
    standardised, near = None, None
    flag_count, before = np.zeros(len(paths)), 0
    try:
        if args.run_dir is not None:
            standardised, first, features = cache_of(args.run_dir, timestamps, len(latitudes))
            unknown = [name for name in args.variables if name not in features]
            if unknown:
                raise ValueError(f"Unknown variables: {unknown}; the run has {features}")
            columns, names = [features.index(name) for name in args.variables], list(args.variables)
        if args.near_normal_run_dir is not None:
            near, near_first, _ = cache_of(args.near_normal_run_dir, timestamps, len(latitudes))
        for path in paths:
            if not pd.DatetimeIndex(np.load(path / "test_timestamps.npy")).equals(timestamps):
                raise ValueError(f"The score is not over the same timestamps: {path}")
            manifest = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            component = args.score_component if not opened else "combined"
            opened.append(np.load(path / score_file(manifest, component), mmap_mode="r"))
            if opened[-1].shape != (len(timestamps), len(latitudes)):
                raise ValueError(f"The score is not over the same cells: {path}")
            if args.split_by_spread and spread_of_first is None:
                kept_spread = (manifest.get("anomaly_std_file") if component == "combined"
                               else manifest.get(f"{component}_std_file"))
                if not kept_spread:
                    raise ValueError(f"The run of the first score kept no spread of the dropout draws: {path}")
                spread_of_first = np.load(path / kept_spread, mmap_mode="r")
                if spread_of_first.shape != opened[-1].shape:
                    raise ValueError("The first score and its spread are not over the same timestamps and cells")
        limits = [float(np.nanpercentile(np.asarray(array[::THRESHOLD_STRIDE]), 100 - args.top_percent)) for array in opened]
        read = 0
        for month in range(1, 13):
            years = sorted(set(timestamps.year[(timestamps.month == month) & wanted]))
            if not years:
                continue
            if standardised is not None:
                rows = np.nonzero((timestamps.month == month) & wanted)[0]
                times, values = timestamps[rows], read_month_of_the_run(standardised, first, rows, columns)
            else:
                parts = [read_month(args.unseen_dir, year, month, latitudes, longitudes) for year in years]
                if any(part[2] != parts[0][2] for part in parts):
                    raise ValueError(f"The files of month {month} do not hold the same variables in every year")
                names = parts[0][2] if names is None else names
                if parts[0][2] != names:
                    raise ValueError(f"The files of month {month} do not hold the variables of the other months")
                times = parts[0][0].append([part[0] for part in parts[1:]])
                rows = timestamps.get_indexer(times)
                kept = rows >= 0  # a file may hold timestamps the score does not have
                values = np.concatenate([part[1] for part in parts])[kept]
                del parts
                times, rows = times[kept], rows[kept]
            unusual, defined = unusual_values(values, times.hour.to_numpy(), unusual_percent, low=args.tail == "low")
            del values
            scores = [np.asarray(array[rows]) for array in opened]
            finite = np.logical_and.reduce([np.isfinite(block) for block in scores])
            before += finite.sum()
            if near is not None:
                finite &= near_normal(near, near_first, rows, args.far)
            with np.errstate(invalid="ignore"):
                marks = [finite & (block > limit) for block, limit in zip(scores, limits)]
                solid = (None if spread_of_first is None else
                         marks[0] & (scores[0] - np.asarray(spread_of_first[rows]) > limits[0]))
            del scores
            flag_count += [mark.sum() for mark in marks]
            some = np.logical_or.reduce(marks[1:])
            groups = (marks[0] & ~some, marks[0] & some, ~marks[0] & some, finite & ~marks[0] & ~some, finite)
            if solid is not None:
                groups += (groups[0] & solid, groups[0] & ~solid, groups[1] & solid, groups[1] & ~solid)
            if hits is None:
                hits, sizes = (np.zeros((len(names_of_groups), len(names))) for _ in range(2))
                by_score, of_unusual = np.zeros((len(paths), len(names))), np.zeros(len(names))
            for index, group in enumerate(groups):
                count[index] += group.sum()
                hits[index] += unusual[group].sum(axis=0)
                sizes[index] += defined[group].sum(axis=0)  # the values where the variable can be unusual
            for variable in range(len(names)):
                where = unusual[..., variable] & finite
                of_unusual[variable] += where.sum()
                by_score[:, variable] += [mark[where].sum() for mark in marks]
            read += len(rows)
    finally:
        for array in [*opened, *(array for array in (standardised, near, spread_of_first) if array is not None)]:
            array._mmap.close()
    if read != wanted.sum():
        raise ValueError(f"The unseen variables cover {read} of the {wanted.sum()} timestamps of the score in "
                         f"{first_year}-{last_year}")

    def row(label, values, form="{:.1f}"):
        print(f"{label:<{NAME}}" + "".join(
            f"{'-' if isinstance(value, float) and np.isnan(value) else form.format(value):>{WIDTH}}" for value in values))

    print(f"[score-against-unseen] years {first_year}-{last_year}, every timestamp | flagged: the highest "
          f"{args.top_percent:g}% of each score | unusual: the {'lowest' if args.tail == 'low' else 'highest'} "
          f"{unusual_percent:g}% of {kind} for the cell, the calendar month and the time of day")
    print(f"score: {labels[0]}, {args.score_component} | others: {', '.join(labels[1:])}")
    if near is not None:
        print(f"only the values where no variable of the run is more than {args.far:g} standard deviations of the training "
              f"years from its seasonal normal: {100 * count[ALL] / before:.1f}% of the values")
    print("\n% of the values of each group where the variable is unusual, among those where it can be")
    row("", ["values", *names], "{}")
    with np.errstate(invalid="ignore", divide="ignore"):
        for index, name in enumerate(names_of_groups):
            print(f"{name:<{NAME}}{int(count[index]):>{WIDTH}}"
                  + "".join(f"{'-' if not size else format(100 * hit / size, '.1f'):>{WIDTH}}"
                            for hit, size in zip(hits[index], sizes[index])))
        print("\n% of the values where the variable is unusual that each score flags; last column: % of all the "
              "values counted that it flags, what it would be if the score had nothing to do with the variable")
        row("", [*names, "all counted"], "{}")
        for label, flagged, of_all in zip(labels, by_score, flag_count):
            row(label, [*(100 * flagged / of_unusual), 100 * of_all / count[ALL]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
