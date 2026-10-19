"""When one score flags a cell before another starts an episode there, is the weather already unusual?

A value is flagged by a score when it is in its highest `--top-percent`. An episode of a score starts
at a cell when the score flags it after at least `--quiet-hours` without a flag there, as in
`check_era5_lead.py`. Every start of an episode of the first score is put in a group by what the
second score does at the same cell:
- already on: it flags the cell at the timestamp before the start, so it saw the episode first;
- at the start: it flags the cell at the start and not before;
- not at the start: it flags the cell neither before nor at the start.
For every group the check reads two variables that are in none of the scores (`download_era5.py
--independent-variables`: the 10 m wind gust and the convective available potential energy), at the
start and OFFSETS_HOURS before it. A variable is unusual when it is in its highest
`--unusual-percent` for the cell, the calendar month and the time of day, as in
`check_era5_score_against_unseen.py`: about `--unusual-percent` of the values are unusual by chance.

If the second score flags real weather before the first, the variables are unusual more often in
the group "already on", at the timestamps before the start where only the second score flags.

What happens by chance: the same groups at cells and timestamps where the first score has been
quiet as long as before a start and does not start (one timestamp every CHANCE_STEP). The group
"already on" by chance is a flag of the second score that no episode of the first follows: if the
variables are unusual there as often as before a start, a flag that comes first is not special.
The same is done the other way round, with the starts of the second score.

The whole scored period is read, one calendar month at a time. With `--csv` the shares go to a
file, one row per direction, group, variable and offset.

    python scripts/check_era5_lead_unseen.py --unseen-dir data/era5_independent/independent \
        --score-dir outputs/run_gen_clim --by-dir outputs/run_simple_max --csv outputs/lead_unseen_5.csv
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

from scripts.check_era5_lead import flags
from scripts.check_era5_score_against_unseen import read_month, unusual_values

CHANCE_STEP = 29  # the chance rows read one timestamp every this many
GROUPS = ("already on", "at the start", "not at the start")
OFFSETS_HOURS = (6, 3, 0)  # hours before the start at which the variables are read
NAME, WIDTH = 40, 11


def starts_by_group(one, other, quiet, chance=False):
    """Steps, cells and groups (index in GROUPS) of the starts of an episode of `one` [time, cell], by what
    `other` does at the same cell. With `chance`, the quiet cells where `one` does not start, one timestamp
    every CHANCE_STEP."""
    times, cells = one.shape
    silent = np.full(cells, np.int32(quiet))
    steps, places, groups = [], [], []
    for step in range(times):
        start = (~one[step] if chance else one[step]) & (silent >= quiet)
        if step >= quiet and not (chance and step % CHANCE_STEP):  # before: what came first is unknown
            where = np.nonzero(start)[0]
            steps.append(np.full(len(where), step, dtype=np.int32))
            places.append(where.astype(np.int32))
            groups.append(np.where(other[step - 1, where], 0, np.where(other[step, where], 1, 2)).astype(np.int8))
        silent = np.where(one[step], 0, np.minimum(silent + 1, quiet)).astype(np.int32)
    return np.concatenate(steps), np.concatenate(places), np.concatenate(groups)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--unseen-dir", type=Path, required=True, help="Directory with a folder of monthly NetCDF files per year")
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score (the model)")
    parser.add_argument("--by-dir", type=Path, required=True, help="Run with the second score")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--by-top-percent", type=float, help="Level of the second score; default --top-percent")
    parser.add_argument("--quiet-hours", type=int, default=24, help="Hours without a flag before a start")
    parser.add_argument("--unusual-percent", type=float, default=5, help="Unusual: this highest share of an unseen variable")
    parser.add_argument("--csv", type=Path, help="Also write the shares here, one row per direction, group, variable and offset")
    args = parser.parse_args(argv)
    first, timestamps, _ = flags(args.score_dir, args.top_percent)
    by_top = args.top_percent if args.by_top_percent is None else args.by_top_percent
    second, other_timestamps, _ = flags(args.by_dir, by_top)
    if not timestamps.equals(other_timestamps) or first.shape != second.shape:
        raise ValueError("The two scores are not over the same timestamps and cells")
    step_hours = int(round((timestamps[1] - timestamps[0]) / pd.Timedelta(hours=1)))
    if args.quiet_hours % step_hours or any(offset % step_hours for offset in OFFSETS_HOURS):
        raise ValueError(f"quiet-hours and the offsets must be multiples of {step_hours}")
    quiet, offsets = args.quiet_hours // step_hours, [offset // step_hours for offset in OFFSETS_HOURS]
    if max(offsets) > quiet:
        raise ValueError("The offsets must not reach before the quiet hours")
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    latitudes = np.asarray(grid["latitudes"], dtype=float)[np.asarray(grid["rows"])]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[np.asarray(grid["cols"])]
    if len(latitudes) != first.shape[1]:
        raise ValueError("The grid of the run does not match the cells of the scores")
    names = [path.name for path in (args.score_dir, args.by_dir)]
    common = len(os.path.commonprefix(names))
    short = [name[common:] or name for name in names]
    # [direction, chance]: steps, cells and groups of the starts
    found = {(index, chance): starts_by_group(one, other, quiet, chance=chance)
             for index, (one, other) in enumerate(((first, second), (second, first))) for chance in (False, True)}
    del first, second
    variables, hits, sizes, read = None, {}, {}, 0
    position = np.full(len(timestamps), -1, dtype=np.int64)  # row of each timestamp in the month read
    for month in range(1, 13):
        years = sorted(set(timestamps.year[timestamps.month == month]))
        if not years:
            continue
        parts = [read_month(args.unseen_dir, year, month, latitudes, longitudes) for year in years]
        variables = parts[0][2] if variables is None else variables
        if any(part[2] != variables for part in parts):
            raise ValueError(f"The files of month {month} do not hold the variables of the other months")
        times = parts[0][0].append([part[0] for part in parts[1:]])
        rows = timestamps.get_indexer(times)
        kept = rows >= 0  # a file may hold timestamps the score does not have
        values = np.concatenate([part[1] for part in parts])[kept]
        del parts
        times, rows = times[kept], rows[kept]
        unusual, defined = unusual_values(values, times.hour.to_numpy(), args.unusual_percent)
        del values
        position[:] = -1
        position[rows] = np.arange(len(rows))
        for key, (steps, places, groups) in found.items():
            if key not in hits:
                hits[key], sizes[key] = (np.zeros((len(GROUPS), len(offsets), len(variables))) for _ in range(2))
            for o, offset in enumerate(offsets):
                row = position[steps - offset]
                inside = row >= 0
                for v in range(len(variables)):
                    hits[key][:, o, v] += np.bincount(groups[inside], weights=unusual[row[inside], places[inside], v],
                                                      minlength=len(GROUPS))
                    sizes[key][:, o, v] += np.bincount(groups[inside], weights=defined[row[inside], places[inside], v],
                                                       minlength=len(GROUPS))
        read += len(rows)
        del unusual, defined
    if read != len(timestamps):
        raise ValueError(f"The unseen variables cover {read} of the {len(timestamps)} timestamps of the scores")

    print(f"[lead-unseen] {timestamps[0].date()} to {timestamps[-1].date()} | flagged: the highest "
          f"{args.top_percent:g}% of {short[0]}, {by_top:g}% of {short[1]} | start of an episode: a flag after "
          f"{args.quiet_hours} h without one at the cell | unusual: the highest {args.unusual_percent:g}% of the "
          f"variable for the cell, the calendar month and the time of day")
    columns = [f"{name} {offset}h" for name in variables for offset in OFFSETS_HOURS]
    rows_out = []
    for index in (0, 1):
        print(f"\nstarts of {short[index]}, by what {short[1 - index]} does at the cell | % of the values where the "
              f"variable is unusual, hours before the start")
        print(f"{'':<{NAME}}{'starts':>{WIDTH + 2}}" + "".join(f"{column:>{WIDTH}}" for column in columns))
        for chance in (False, True):
            steps, _, groups = found[index, chance]
            counts = np.bincount(groups, minlength=len(GROUPS))
            with np.errstate(invalid="ignore", divide="ignore"):
                shares = 100 * hits[index, chance] / sizes[index, chance]
            for g, group in enumerate(GROUPS):
                label = f"{'chance, ' if chance else ''}{short[1 - index]} {group}"
                print(f"{label[:NAME - 1]:<{NAME}}{counts[g]:>{WIDTH + 2}d}"
                      + "".join(f"{'-' if np.isnan(shares[g, o, v]) else format(shares[g, o, v], '.1f'):>{WIDTH}}"
                                for v in range(len(variables)) for o in range(len(offsets))))
                for o, offset in enumerate(OFFSETS_HOURS):
                    for v, name in enumerate(variables):
                        rows_out.append({"starts_of": short[index], "other": short[1 - index], "chance": chance,
                                         "group": group, "variable": name, "hours_before": offset,
                                         "share": shares[g, o, v], "starts": int(counts[g]),
                                         "top_percent": args.top_percent if index == 0 else by_top,
                                         "by_top_percent": by_top if index == 0 else args.top_percent,
                                         "unusual_percent": args.unusual_percent})
    print(f"already on: the other score flags the cell {step_hours} h before the start; at the start: it flags it at the start "
          f"and not before; not at the start: neither; chance: the same groups where the score of the episodes was "
          f"quiet as long and did not start. By chance about {args.unusual_percent:g}% of the values are unusual.")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(args.csv, index=False)
        print(f"written {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
