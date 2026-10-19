"""What the model flags where no simple score comes within a day: how many events, where, when, and is the weather unusual after?

A value is flagged by a score when it is in its highest `--top-percent` (`--by-top-percent` for the
others). A value the first score flags is unreached when none of the other scores (`--by-dir`, one
or more) flags the same cell within `--window-hours` before or after it. The unreached values are
grouped in events: touching cells at a timestamp (diagonals included) and the same cell at the
timestamps before and after, as in `check_era5_event_first.py`. The check says:
- how many values and events there are, how large and how long the events are;
- in which season, time of day (UTC) and bands of latitude and longitude the values fall, beside the
  share of all the values the first score flags there: where they are over-represented;
- with `--unseen-dir`, how often a variable no score saw (the 10 m gust, the CAPE) is in its highest
  `--unusual-percent` for the cell, the calendar month and the time of day, at the same cell at
  OFFSETS_HOURS from the value: for all the unreached values, for the first timestamp of every event
  (its start) and, as a reference, for calm values (flagged by no score, and no other score within the
  window), one timestamp every CHANCE_STEP. Unusual weather that grows after the start is a lead of
  the first score on the weather itself, which the simple scores never flag.

    python scripts/check_era5_unreached.py --score-dir outputs/run_gen_clim \
        --by-dir outputs/run_simple_max outputs/run_simple_convective_clim \
        --unseen-dir data/era5_independent/independent --csv outputs/unreached_5.csv
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
from scipy import ndimage

from scripts.check_era5_edge_lead import since_and_until
from scripts.check_era5_lead import flags
from scripts.check_era5_score_against_unseen import read_month, unusual_values

CHUNK = 1024  # timestamps written at a time
OFFSETS_HOURS = (-6, -3, 0, 3, 6, 12)
CHANCE_STEP = 29  # the calm reference reads one timestamp every this many
DURATIONS_HOURS = ((3, 3), (6, 6), (9, 12), (15, 24), (27, None))
SIZES = ((1, 1), (2, 4), (5, 19), (20, None))
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
LATITUDE_BAND, LONGITUDE_BAND = 5.0, 10.0  # degrees
GROUPS = ("unreached values", "event starts", "calm values")
NAME, WIDTH = 36, 10


def band_label(low, high, unit=""):
    return f"{low}{unit}" if low == high else (f"{low}-{high}{unit}" if high else f">{low - 1}{unit}" if unit != "h"
                                                else f">{low - 3}{unit}")


def shares(values, reference, keys):
    """% of `values` and of `reference` in every key."""
    count = lambda array, key: 100 * np.count_nonzero(array == key) / max(len(array), 1)
    return [(key, count(values, key), count(reference, key)) for key in keys]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score (the model)")
    parser.add_argument("--by-dir", type=Path, nargs="+", required=True, help="Runs with the other scores")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--by-top-percent", type=float, help="Level of the other scores; default --top-percent")
    parser.add_argument("--window-hours", type=int, default=24, help="Hours before and after with no flag of the others")
    parser.add_argument("--unseen-dir", type=Path, help="Monthly NetCDF files of the unseen variables")
    parser.add_argument("--unusual-percent", type=float, default=5, help="Unusual: this highest share of an unseen variable")
    parser.add_argument("--csv", type=Path, help="Also write the events here, one row each")
    args = parser.parse_args(argv)
    first, timestamps, _ = flags(args.score_dir, args.top_percent)
    by_top = args.top_percent if args.by_top_percent is None else args.by_top_percent
    others = None
    for path in args.by_dir:
        flagged, other_timestamps, _ = flags(path, by_top)
        if not timestamps.equals(other_timestamps) or first.shape != flagged.shape:
            raise ValueError(f"The scores are not over the same timestamps and cells: {path}")
        others = flagged if others is None else others | flagged
        del flagged
    step_hours = int(round((timestamps[1] - timestamps[0]) / pd.Timedelta(hours=1)))
    if args.window_hours % step_hours or any(hours % step_hours for hours in OFFSETS_HOURS):
        raise ValueError(f"The hours must be multiples of {step_hours}")
    window, offsets = args.window_hours // step_hours, [hours // step_hours for hours in OFFSETS_HOURS]
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    if len(rows) != first.shape[1]:
        raise ValueError("The grid of the run does not match the cells of the scores")
    latitudes = np.asarray(grid["latitudes"], dtype=float)[rows]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[cols]
    times = len(timestamps)

    since, until = since_and_until(others, window + 1)
    away = (since > window) & (until > window)  # no flag of the others at the cell within the window
    del since, until
    unreached = first & ~others & away
    calm = ~first & ~others & away
    del away, others
    lattice = np.zeros((times, *shape), dtype=bool)
    for start in range(0, times, CHUNK):
        lattice[start:start + CHUNK, rows, cols] = unreached[start:start + CHUNK]
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1] = True  # touching cells at the same timestamp, diagonals included
    structure[0, 1, 1] = structure[2, 1, 1] = True  # the same cell at the timestamps before and after
    labels, count = ndimage.label(lattice, structure=structure, output=np.int32)
    del lattice
    steps, cells = np.nonzero(unreached)  # in order of time
    events = labels[steps, rows[cells], cols[cells]].astype(np.int64)
    del labels, unreached
    size = count + 1
    values = np.bincount(events, minlength=size)
    found, position = np.unique(events, return_index=True)
    begin = np.zeros(size, dtype=np.int64)
    begin[found] = steps[position]
    found, position = np.unique(events[::-1], return_index=True)
    end = np.zeros(size, dtype=np.int64)
    end[found] = steps[::-1][position]
    duration = (end - begin + 1) * step_hours
    starting = steps == begin[events]
    calm_steps, calm_cells = np.nonzero(calm[::CHANCE_STEP])
    calm_steps = calm_steps * CHANCE_STEP
    flagged_steps, flagged_cells = np.nonzero(first)
    del calm, first

    names = [path.name for path in (args.score_dir, *args.by_dir)]
    common = len(os.path.commonprefix(names))
    short = [name[common:] or name for name in names]
    years = (timestamps[-1] - timestamps[0]) / pd.Timedelta(days=365.25)
    total_flagged = len(flagged_steps)
    print(f"[unreached] {timestamps[0].date()} to {timestamps[-1].date()} | flagged: the highest {args.top_percent:g}% of "
          f"{short[0]}, {by_top:g}% of {', '.join(short[1:])} | unreached: {short[0]} flags it and none of the others "
          f"flags the cell within {args.window_hours} h before or after")
    print(f"unreached values: {len(steps)} ({100 * len(steps) / max(total_flagged, 1):.1f}% of the {total_flagged} values "
          f"{short[0]} flags), {len(steps) / years:.0f} a year | events: {count}, {count / years:.0f} a year")
    real = np.arange(1, size)
    print(f"\nevents by size (values) and by duration (% of the events)")
    print(f"{'size':<{NAME}}" + "".join(f"{band_label(low, high):>{WIDTH}}" for low, high in SIZES)
          + f"{'median':>{WIDTH}}{'largest':>{WIDTH}}")
    by_size = [100 * np.count_nonzero((values[real] >= low) & ((values[real] <= high) if high else True)) / max(count, 1)
               for low, high in SIZES]
    print(f"{'':<{NAME}}" + "".join(f"{share:>{WIDTH}.1f}" for share in by_size)
          + f"{np.median(values[real]) if count else 0:>{WIDTH}.0f}{values[real].max() if count else 0:>{WIDTH}d}")
    print(f"{'duration':<{NAME}}" + "".join(f"{band_label(low, high, 'h'):>{WIDTH}}" for low, high in DURATIONS_HOURS)
          + f"{'median h':>{WIDTH}}{'longest h':>{WIDTH}}")
    by_duration = [100 * np.count_nonzero((duration[real] >= low) & ((duration[real] <= high) if high else True))
                   / max(count, 1) for low, high in DURATIONS_HOURS]
    print(f"{'':<{NAME}}" + "".join(f"{share:>{WIDTH}.1f}" for share in by_duration)
          + f"{np.median(duration[real]) if count else 0:>{WIDTH}.0f}{duration[real].max() if count else 0:>{WIDTH}d}")

    season_of = {month: name for name, months in SEASONS.items() for month in months}
    month_of = timestamps.month.to_numpy()
    tables = [
        ("season", [season_of[month] for month in month_of], list(SEASONS)),
        ("time of day, UTC", timestamps.hour.to_numpy(), sorted(set(timestamps.hour))),
    ]
    print(f"\n% of the unreached values, and of all the values {short[0]} flags, by season, time of day and band")
    print(f"{'':<{NAME}}{'unreached':>{WIDTH}}{'all flags':>{WIDTH}}{'ratio':>{WIDTH}}")
    for title, of_step, keys in tables:
        of_step = np.asarray(of_step)
        for key, mine, everyone in shares(of_step[steps], of_step[flagged_steps], keys):
            print(f"{f'{title} {key}'[:NAME - 1]:<{NAME}}{mine:>{WIDTH}.1f}{everyone:>{WIDTH}.1f}"
                  f"{mine / everyone if everyone else np.nan:>{WIDTH}.2f}")
    for title, of_cell, band in (("latitude", latitudes, LATITUDE_BAND), ("longitude", longitudes, LONGITUDE_BAND)):
        low = np.floor(of_cell / band) * band
        for key, mine, everyone in shares(low[cells], low[flagged_cells], sorted(set(low))):
            print(f"{f'{title} {key:g} to {key + band:g}'[:NAME - 1]:<{NAME}}{mine:>{WIDTH}.1f}{everyone:>{WIDTH}.1f}"
                  f"{mine / everyone if everyone else np.nan:>{WIDTH}.2f}")
    del flagged_steps, flagged_cells

    if args.unseen_dir is not None:
        groups = [(steps, cells), (steps[starting], cells[starting]), (calm_steps, calm_cells)]
        hits = defined = None
        variables = None
        position = np.full(times, -1, dtype=np.int64)
        for month in range(1, 13):
            positions = np.nonzero(month_of == month)[0]
            if not len(positions):
                continue
            parts = [read_month(args.unseen_dir, year, month, latitudes, longitudes)
                     for year in sorted(set(timestamps.year[positions]))]
            variables = parts[0][2] if variables is None else variables
            if any(part[2] != variables for part in parts):
                raise ValueError(f"The files of month {month} do not hold the variables of the other months")
            read = parts[0][0].append([part[0] for part in parts[1:]])
            index = timestamps.get_indexer(read)
            kept = index >= 0
            block = np.concatenate([part[1] for part in parts])[kept]
            del parts
            unusual, known = unusual_values(block, read[kept].hour.to_numpy(), args.unusual_percent)
            del block
            if hits is None:
                hits, defined = (np.zeros((len(GROUPS), len(offsets), len(variables))) for _ in range(2))
            position[:] = -1
            position[index[kept]] = np.arange(np.count_nonzero(kept))
            for g, (group_steps, group_cells) in enumerate(groups):
                for o, offset in enumerate(offsets):
                    target = group_steps + offset
                    inside = (target >= 0) & (target < times)
                    row = np.full(len(target), -1, dtype=np.int64)
                    row[inside] = position[target[inside]]
                    use = row >= 0
                    hits[g, o] += unusual[row[use], group_cells[use]].sum(axis=0)
                    defined[g, o] += known[row[use], group_cells[use]].sum(axis=0)
            del unusual, known
        print(f"\n% of the values where an unseen variable is in its highest {args.unusual_percent:g}% for the cell, the "
              f"calendar month and the time of day, at the same cell, hours from the value (by chance "
              f"{args.unusual_percent:g}%)")
        print(f"{'group, variable':<{NAME}}" + "".join(f"{f'{hours:+d}h':>{WIDTH}}" for hours in OFFSETS_HOURS))
        with np.errstate(invalid="ignore", divide="ignore"):
            share = 100 * hits / defined
        for g, group in enumerate(GROUPS):
            for v, variable in enumerate(variables):
                print(f"{f'{group}, {variable}'[:NAME - 1]:<{NAME}}"
                      + "".join(f"{share[g, o, v]:>{WIDTH}.1f}" for o in range(len(offsets))))
        print(f"event starts: the cells of the first timestamp of every event; calm values: flagged by no score and no "
              f"other score within {args.window_hours} h, one timestamp every {CHANCE_STEP}")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        start_cells = cells[starting]
        centre_latitude = np.bincount(events[starting], weights=latitudes[start_cells], minlength=size)
        centre_longitude = np.bincount(events[starting], weights=longitudes[start_cells], minlength=size)
        start_count = np.bincount(events[starting], minlength=size)
        pd.DataFrame({"event": real, "start": timestamps[begin[real]], "duration_hours": duration[real],
                      "values": values[real], "start_latitude": centre_latitude[real] / start_count[real],
                      "start_longitude": centre_longitude[real] / start_count[real]}).to_csv(args.csv, index=False)
        print(f"written {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
