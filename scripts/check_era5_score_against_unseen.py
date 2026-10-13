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

from scripts.check_era5_dust_lead import THRESHOLD_STRIDE, score_file

GROUPS = ("score alone, none of the others", "score and at least one other", "others, not the score", "flagged by none",
          "all values")
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


def unusual_values(values, hours, percent):
    """Where each variable [timestamps, cells, variables] is in its highest `percent` among the timestamps of
    the same time of day, cell by cell."""
    result = np.zeros(values.shape, dtype=bool)
    for hour in np.unique(hours):
        chosen = hours == hour
        result[chosen] = values[chosen] > np.percentile(values[chosen], 100 - percent, axis=0)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--unseen-dir", type=Path, required=True, help="Directory with a folder of monthly NetCDF files per year")
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the score that is asked about")
    parser.add_argument("--by-dirs", type=Path, nargs="+", required=True, help="Runs with the other scores, over the same period")
    parser.add_argument("--top-percent", type=float, default=5, help="Flagged: this highest share of each score")
    parser.add_argument("--unusual-percent", type=float, help="Unusual: this highest share of an unseen variable; default: top-percent")
    args = parser.parse_args(argv)
    unusual_percent = args.top_percent if args.unusual_percent is None else args.unusual_percent
    paths = [args.score_dir, *args.by_dirs]
    common = os.path.commonprefix([path.name for path in paths])
    labels = [path.name[len(common):] or path.name for path in paths]
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    latitudes = np.asarray(grid["latitudes"], dtype=float)[np.asarray(grid["rows"])]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[np.asarray(grid["cols"])]
    timestamps = pd.DatetimeIndex(np.load(args.score_dir / "test_timestamps.npy"))
    opened, names = [], None
    count, hits = np.zeros(len(GROUPS)), None
    try:
        for path in paths:
            if not pd.DatetimeIndex(np.load(path / "test_timestamps.npy")).equals(timestamps):
                raise ValueError(f"The score is not over the same timestamps: {path}")
            manifest = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            opened.append(np.load(path / score_file(manifest, "combined"), mmap_mode="r"))
            if opened[-1].shape != (len(timestamps), len(latitudes)):
                raise ValueError(f"The score is not over the same cells: {path}")
        limits = [float(np.nanpercentile(np.asarray(array[::THRESHOLD_STRIDE]), 100 - args.top_percent)) for array in opened]
        read = 0
        for month in range(1, 13):
            years = sorted(set(timestamps.year[timestamps.month == month]))
            if not years:
                continue
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
            unusual = unusual_values(values, times.hour.to_numpy(), unusual_percent)
            del values
            scores = [np.asarray(array[rows]) for array in opened]
            finite = np.logical_and.reduce([np.isfinite(block) for block in scores])
            with np.errstate(invalid="ignore"):
                marks = [finite & (block > limit) for block, limit in zip(scores, limits)]
            del scores
            some = np.logical_or.reduce(marks[1:])
            groups = (marks[0] & ~some, marks[0] & some, ~marks[0] & some, finite & ~marks[0] & ~some, finite)
            if hits is None:
                hits = np.zeros((len(GROUPS), len(names)))
                by_score, of_unusual = np.zeros((len(paths), len(names))), np.zeros(len(names))
            for index, group in enumerate(groups):
                count[index] += group.sum()
                hits[index] += unusual[group].sum(axis=0)
            for variable in range(len(names)):
                where = unusual[..., variable] & finite
                of_unusual[variable] += where.sum()
                by_score[:, variable] += [mark[where].sum() for mark in marks]
            read += len(rows)
    finally:
        for array in opened:
            array._mmap.close()
    if read != len(timestamps):
        raise ValueError(f"The unseen variables cover {read} of the {len(timestamps)} timestamps of the score")

    def row(label, values, form="{:.1f}"):
        print(f"{label:<{NAME}}" + "".join(
            f"{'-' if isinstance(value, float) and np.isnan(value) else form.format(value):>{WIDTH}}" for value in values))

    print(f"[score-against-unseen] {timestamps[0].date()} to {timestamps[-1].date()}, every timestamp | flagged: the highest "
          f"{args.top_percent:g}% of each score | unusual: the highest {unusual_percent:g}% of an unseen variable for the "
          f"cell, the calendar month and the time of day")
    print(f"score: {labels[0]} | others: {', '.join(labels[1:])}")
    print("\n% of the values of each group where the unseen variable is unusual")
    row("", ["values", *names], "{}")
    with np.errstate(invalid="ignore", divide="ignore"):
        for index, name in enumerate(GROUPS):
            print(f"{name:<{NAME}}{int(count[index]):>{WIDTH}}"
                  + "".join(f"{'-' if not count[index] else format(100 * hit / count[index], '.1f'):>{WIDTH}}"
                            for hit in hits[index]))
        print(f"\n% of the values where the unseen variable is unusual that each score flags ({args.top_percent:g} if the "
              f"score had nothing to do with it)")
        row("", names, "{}")
        for label, flagged in zip(labels, by_score):
            row(label, list(100 * flagged / of_unusual))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
