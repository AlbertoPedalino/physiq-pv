"""Does the score flag the documented days of 2019?

The days of 2019 with a documented event over Piedmont and over France and northern Italy (the cold
of January and February, the days of April, the heat of June and July, the floods of October and
November) are read against the score of a run. The reading is the one of
`check_era5_dust_events.py`: per place, the share of its cells above the score threshold on that
day, as a percentile among the days of the same calendar month.

With `--days` other days are read instead, and with `--zone` other places (name, south, north,
west, east in degrees; repeatable): any documented day anywhere in the domain.

With `--feature-errors-dir`, the run whose scoring saved the squared error by variable, the same
days are also read one variable at a time: per place and variable, the mean error over the place
on that day as a percentile among the days of the same calendar month. An event that only one or
two variables carry shows there even when the score, a mean over all of them, does not move.

With `--prepared-dir`, the prepared ERA5 data the run was scored on, the same days are read in the
data themselves, with no model: per place and variable, the mean of the variable over the place on
that day as a percentile among the days of the same calendar month of the scored years, 0 for the
lowest and 100 for the highest. It says what the weather was; the other readings say what the
model made of it.

    python scripts/check_era5_known_days.py --run-dir outputs/run_bal_rarity
    python scripts/check_era5_known_days.py --run-dir outputs/run_bal_rarity \
        --days 2014-04-03 2014-04-04 2014-04-05 --zone Germany 47 55 6 15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts.check_era5_dust_events import ALARM, month_percentile
from scripts.check_era5_dust_lead import COMPONENTS, THRESHOLD_STRIDE, score_file

# name: south, north, west, east
KNOWN_ZONES = {"Piedmont": (44, 46.5, 6.5, 9.5), "France and northern Italy": (43, 49, 0, 12)}
FEATURE_CHUNK = 256  # timestamps of the errors by variable read at a time
NAME = 36
LOW, HIGH = 5, 95  # percentiles beyond which a variable is far from the usual of its month
DAYS_SHOWN = 12  # up to this many days the data are printed day by day
KNOWN_DAYS = ("2019-01-30", "2019-01-31", "2019-02-01", "2019-02-02", "2019-02-03", "2019-04-22", "2019-04-23",
              "2019-04-24", "2019-04-25", "2019-06-25", "2019-06-26", "2019-06-27", "2019-06-28", "2019-06-29",
              "2019-06-30", "2019-07-15", "2019-07-25", "2019-10-19", "2019-10-20", "2019-10-21", "2019-10-22",
              "2019-10-23", "2019-11-15", "2019-11-22", "2019-11-23", "2019-11-24")


def zone_percentiles(scores, days, latitudes, longitudes, threshold, zones=None):
    """Per zone: the daily share of its cells above `threshold`, as a percentile within the calendar month."""
    result = {}
    for name, (south, north, west, east) in (KNOWN_ZONES if zones is None else zones).items():
        cells = np.nonzero((latitudes >= south) & (latitudes <= north) & (longitudes >= west) & (longitudes <= east))[0]
        if not len(cells):
            raise ValueError(f"Zone {name} has no cell in the score grid")
        share = pd.Series(np.nanmean(np.asarray(scores[:, cells], dtype=float) > threshold, axis=1), index=days)
        result[name] = month_percentile(share.groupby(level=0).mean())
    return result


def variable_percentiles(errors, days, places):
    """Per place: the daily mean error of each variable over its cells, as a percentile within the calendar month
    ([days, variables]). The errors are read once for all the places."""
    means = {name: np.empty((len(errors), errors.shape[2])) for name in places}
    for start in range(0, len(errors), FEATURE_CHUNK):
        block = np.asarray(errors[start:start + FEATURE_CHUNK])
        for name, cells in places.items():
            means[name][start:start + FEATURE_CHUNK] = block[:, cells].mean(axis=1, dtype=np.float64)
    result = {}
    for name, values in means.items():
        daily = pd.DataFrame(values, index=days).groupby(level=0).mean()
        result[name] = daily.groupby(daily.index.month).rank(pct=True) * 100
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Run with scores, e.g. a rebalanced run")
    parser.add_argument("--score-component", choices=COMPONENTS, default="combined")
    parser.add_argument("--top-percent", type=float, default=5, help="Score threshold: this highest share of the values")
    parser.add_argument("--days", nargs="+", help="Days to read instead of the documented days of 2019")
    parser.add_argument("--zone", nargs=5, action="append", metavar=("NAME", "SOUTH", "NORTH", "WEST", "EAST"),
                        help="A place to read instead of the two of the documented days; repeatable")
    parser.add_argument("--feature-errors-dir", type=Path,
                        help="Run with scores/feature_scores.npy over the same period: the days one variable at a time")
    parser.add_argument("--prepared-dir", type=Path,
                        help="Prepared ERA5 data the run was scored on: the same days in the data themselves, no model")
    args = parser.parse_args(argv)
    zones = KNOWN_ZONES if args.zone is None else {name: tuple(map(float, box)) for name, *box in args.zone}
    listed = pd.DatetimeIndex(KNOWN_DAYS if args.days is None else args.days)
    manifest = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    grid = manifest["grid"]
    latitudes = np.asarray(grid["latitudes"], dtype=float)[np.asarray(grid["rows"])]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[np.asarray(grid["cols"])]
    days = pd.DatetimeIndex(np.load(args.run_dir / "test_timestamps.npy")).normalize()
    scores = np.load(args.run_dir / score_file(manifest, args.score_component), mmap_mode="r")
    try:
        threshold = float(np.nanpercentile(np.asarray(scores[::THRESHOLD_STRIDE]), 100 - args.top_percent))
        percentile = zone_percentiles(scores, days, latitudes, longitudes, threshold, zones)
    finally:
        scores._mmap.close()
    known = [day for day in listed if day in days]
    print(f"[known-days] run read from {days[0].date()} to {days[-1].date()} | documented days in it: {len(known)} of "
          f"{len(listed)} | score: percentile of the day within its calendar month, alarm at {ALARM}")
    if not known:
        return 0
    print(f"{'day':<12}" + "".join(f"{name:>28}" for name in zones))
    for day in known:
        print(f"{str(day.date()):<12}" + "".join(f"{percentile[name][day]:>28.0f}" for name in zones))
    readings = len(known) * len(zones)
    print(f"\ndays and places at or above {ALARM}, of {readings}: "
          + " | ".join(f"{name} {int(sum(percentile[name][day] >= ALARM for day in known))}" for name in zones)
          + f" | both {int(sum(percentile[name][day] >= ALARM for name in zones for day in known))} "
          f"({(1 - ALARM / 100) * readings:.1f} expected by chance)")
    places = {name: np.nonzero((latitudes >= south) & (latitudes <= north) & (longitudes >= west) & (longitudes <= east))[0]
              for name, (south, north, west, east) in zones.items()}
    if args.prepared_dir is not None:
        prepared = json.loads((args.prepared_dir / "metadata.json").read_text(encoding="utf-8"))
        values = np.load(args.prepared_dir / "test.npy", mmap_mode="r")
        try:
            stamps = pd.DatetimeIndex(np.load(args.prepared_dir / "test_timestamps.npy")).normalize()
            if values.shape[:2] != (len(days), len(latitudes)) or not stamps.equals(days):
                raise ValueError("The prepared data and the score are not over the same timestamps and cells")
            in_data = variable_percentiles(values, days, places)
        finally:
            values._mmap.close()
        names = list(prepared.get("features") or [f"feature_{index}" for index in range(values.shape[2])])
        print(f"\nthe data themselves, no model: mean of the variable over the place, percentile of the day within its "
              f"calendar month (0 the lowest of the month, 100 the highest; about {0.1 * len(names):.1f} variables outside "
              f"{LOW}-{HIGH} on any day)")
        for place in zones:
            table = in_data[place].loc[known]
            if len(known) <= DAYS_SHOWN:
                print(f"\n{place}\n{'':<{NAME}}" + "".join(f"{str(day.date())[5:]:>7}" for day in known))
                for index, name in enumerate(names):
                    print(f"{name:<{NAME}}" + "".join(f"{value:>7.0f}" for value in table[index]))
                outside = ((table <= LOW) | (table >= HIGH)).sum(axis=1)
                print(f"{'variables outside ' + str(LOW) + '-' + str(HIGH):<{NAME}}" + "".join(f"{int(count):>7}" for count in outside))
            else:
                print(f"\n{place}\n{'':<{NAME}}{'low days':>9}{'high days':>10}{'mean':>7}")
                for index, name in enumerate(names):
                    column = table[index]
                    print(f"{name:<{NAME}}{int((column <= LOW).sum()):>9}{int((column >= HIGH).sum()):>10}{column.mean():>7.0f}")
    if args.feature_errors_dir is None:
        return 0
    source = json.loads((args.feature_errors_dir / "metadata.json").read_text(encoding="utf-8"))
    errors = np.load(args.feature_errors_dir / "scores" / "feature_scores.npy", mmap_mode="r")
    try:
        if errors.shape[:2] != (len(days), len(latitudes)):
            raise ValueError("The errors by variable and the score are not over the same timestamps and cells")
        names = list(source.get("preparation", {}).get("features") or
                     [f"feature_{index}" for index in range(errors.shape[2])])
        by_variable = variable_percentiles(errors, days, places)
    finally:
        errors._mmap.close()
    print(f"\none variable at a time: mean error over the place, percentile of the day within its calendar month | "
          f"days at or above {ALARM} of {len(known)} ({(1 - ALARM / 100) * len(known):.1f} expected by chance), "
          f"mean and highest percentile")
    for place in zones:
        print(f"\n{place}\n{'':<{NAME}}{'days':>6}{'mean':>7}{'highest':>9}")
        table = by_variable[place].loc[known]
        for index, name in enumerate(names):
            values = table[index]
            print(f"{name:<{NAME}}{int((values >= ALARM).sum()):>6}{values.mean():>7.0f}{values.max():>9.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
