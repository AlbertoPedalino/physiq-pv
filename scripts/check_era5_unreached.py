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
  the first score on the weather itself, which the simple scores never flag;
- with `--feature-errors-dir`, the run whose scoring saved the squared error by variable, what the
  error of the unreached values is made of, beside all the values the first score flags: the share of
  each variable in the sum of the weighted errors, and how often it is the variable with the largest
  weighted error (the main variable). In the events of two values or more, the share whose values all
  have the same main variable, beside the same share with the main variables shuffled among the
  unreached values: one kind of weather keeps its variable, noise does not.

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
ERROR_CHUNK = 64  # timestamps of the errors by variable read at a time
SHUFFLE_SEED = 0
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


def error_makeup(errors, weights, masks):
    """For every (T, N) mask: the number of finite values, the share of each variable in the sum of their weighted
    errors and the share of the values whose largest weighted error is that variable; and the main variable of
    every value of the last mask, in the order of np.nonzero."""
    variables = errors.shape[2]
    counts, sums, main = np.zeros(len(masks)), np.zeros((len(masks), variables)), np.zeros((len(masks), variables))
    last = []
    for start in range(0, len(errors), ERROR_CHUNK):
        block = np.asarray(errors[start:start + ERROR_CHUNK], dtype=np.float64) * weights
        finite = np.isfinite(block).all(axis=2)
        largest = np.where(finite, np.nan_to_num(block, nan=-np.inf).argmax(axis=2), -1)
        for m, mask in enumerate(masks):
            chosen = np.asarray(mask[start:start + ERROR_CHUNK])
            if m == len(masks) - 1:
                last.append(largest[chosen])
            chosen = chosen & finite
            counts[m] += chosen.sum()
            sums[m] += block[chosen].sum(axis=0)
            main[m] += np.bincount(largest[chosen], minlength=variables)
    with np.errstate(invalid="ignore", divide="ignore"):
        return counts, sums / sums.sum(axis=1, keepdims=True), main / counts[:, None], np.concatenate(last)


def same_main_share(events, main, variables):
    """% of the events of two values or more whose values all have the same main variable (values with no main
    variable, -1, leave their event out)."""
    size = int(events.max()) + 1 if len(events) else 1
    values = np.bincount(events, minlength=size)
    unknown = np.bincount(events, weights=main < 0, minlength=size) > 0
    known = main >= 0
    pairs = np.bincount(events[known] * variables + main[known], minlength=size * variables).reshape(size, variables)
    kept = (values >= 2) & ~unknown
    kept[0] = False  # label 0 is no event
    return 100 * np.count_nonzero(pairs[kept].max(axis=1) == values[kept]) / max(np.count_nonzero(kept), 1), \
        int(np.count_nonzero(kept))


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
    parser.add_argument("--feature-errors-dir", type=Path,
                        help="Run with scores/feature_scores.npy over the same timestamps and cells: what the error of "
                             "the unreached values is made of")
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
    del labels
    makeup = None
    if args.feature_errors_dir is not None:
        source = json.loads((args.feature_errors_dir / "metadata.json").read_text(encoding="utf-8"))
        errors = np.load(args.feature_errors_dir / "scores" / "feature_scores.npy", mmap_mode="r")
        try:
            if errors.shape[:2] != first.shape:
                raise ValueError("The errors by variable and the scores are not over the same timestamps and cells")
            variable_names = list(source.get("preparation", {}).get("features") or
                                  [f"feature_{index}" for index in range(errors.shape[2])])
            saved = source.get("backend", {}).get("reconstruction_feature_weights")
            weights = np.array([saved[name] for name in variable_names]) if saved else np.ones(len(variable_names))
            counts, made_of, main_share, main_of_value = error_makeup(errors, weights, (first, unreached))
        finally:
            errors._mmap.close()
        shuffled = np.random.default_rng(SHUFFLE_SEED).permutation(main_of_value)
        makeup = (variable_names, saved is not None, counts, made_of, main_share,
                  same_main_share(events, main_of_value, len(variable_names)),
                  same_main_share(events, shuffled, len(variable_names)))
    del unreached
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
    if makeup is not None:
        variable_names, weighted, counts, made_of, main_share, (same, kept), (same_shuffled, _) = makeup
        print(f"\nwhat the error is made of, % by variable ({100 / len(variable_names):.1f} each if equal), "
              f"{'with the weights of the run' if weighted else 'unweighted'}: share of the sum of the errors, and "
              f"share of the values where it is the largest (main variable)")
        print(f"{'':<{NAME}}{'unreached':>{WIDTH}}{'all flags':>{WIDTH}}{'ratio':>{WIDTH}}"
              f"{'main unr.':>{WIDTH}}{'main all':>{WIDTH}}{'ratio':>{WIDTH}}")
        print(f"{'values read':<{NAME}}{counts[1]:>{WIDTH}.0f}{counts[0]:>{WIDTH}.0f}")
        with np.errstate(invalid="ignore", divide="ignore"):
            for index, name in enumerate(variable_names):
                print(f"{name[:NAME - 1]:<{NAME}}{100 * made_of[1, index]:>{WIDTH}.1f}{100 * made_of[0, index]:>{WIDTH}.1f}"
                      f"{made_of[1, index] / made_of[0, index]:>{WIDTH}.2f}{100 * main_share[1, index]:>{WIDTH}.1f}"
                      f"{100 * main_share[0, index]:>{WIDTH}.1f}{main_share[1, index] / main_share[0, index]:>{WIDTH}.2f}")
        print(f"events of two values or more ({kept}): all values with the same main variable {same:.1f}%, with the "
              f"main variables shuffled among the unreached values {same_shuffled:.1f}%")
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
