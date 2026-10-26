"""How long the weather of the unreached values lasts, hour by hour, between the 3-hourly timestamps.

The unreached values are those of `check_era5_unreached.py`: the first score flags them and no other score
flags the cell within `--window-hours` before or after. The hourly ERA5 fields of `--hourly-dir` (written by
`download_era5.py --hourly-check-variables`: 10 m gust, CAPE, boundary layer height, solar radiation, cloud
cover) are read at the same cell from 3 hours before to 3 hours after each value. A variable is unusual when
it is in its highest (or lowest) `--unusual-percent` for the cell, the calendar month and the hour of the day,
over the years of the period. For three groups, the unreached values, all the values the first score flags
and the calm values (flagged by no score, no other score within the window), the last two read one timestamp
every SAMPLE_STEP, the check says:
- the share of unusual values at every hour from -3 h to +3 h, highest and lowest apart;
- among the values unusual at the hour of the value, how many hours in a row the variable stays unusual
  on the same side (1 to 7): an anomaly shorter than the 3 hours between the model's timestamps lasts 1 or 2;
- among the same values, how abrupt the change is: the largest of the six hourly changes from -3 h to +3 h
  over their sum (1/6 if the variable changes evenly, 1 if it changes in one hour only), and how often the
  largest change is the hour just before or just after the value.
The hourly fields come from the same reanalysis as the model's data: the check describes the kind of anomaly
(short or long, gradual or abrupt), it does not confirm it with an independent observation.

    python scripts/check_era5_unreached_hourly.py --score-dir outputs/good2v/val_gen_clim \
        --by-dir outputs/good2v/val_simple_max outputs/good2v/val_simple_cloud_clim \
        outputs/good2v/val_simple_convective_clim --hourly-dir data/era5_hourly_check/hourly_check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import numpy as np
import pandas as pd

from scripts.check_era5_score_against_unseen import read_month, unusual_values
from scripts.check_era5_unreached import select

OFFSETS = tuple(range(-3, 4))  # hours from the value
CENTRE = OFFSETS.index(0)
SAMPLE_STEP = 29  # the reference groups read one timestamp every this many
GROUPS = ("unreached values", "all flags of the first", "calm values")
TAILS = ("high", "low")
NAME, WIDTH = 34, 8


def run_length(unusual):
    """Hours in a row unusual through the centre column of [values, offsets], the centre included."""
    after = np.cumprod(unusual[:, CENTRE:], axis=1).sum(axis=1)
    before = np.cumprod(unusual[:, CENTRE::-1], axis=1).sum(axis=1)
    return after + before - 1


def abruptness(series):
    """For [values, offsets]: the largest hourly change over the sum of the changes, and whether the largest
    change is the hour just before or just after the centre. Rows with no change give NaN."""
    changes = np.abs(np.diff(series, axis=1))
    total = changes.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(total > 0, changes.max(axis=1) / total, np.nan)
    largest = changes.argmax(axis=1)
    return ratio, np.isin(largest, (CENTRE - 1, CENTRE)) & (total > 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score (the model)")
    parser.add_argument("--by-dir", type=Path, nargs="+", required=True, help="Runs with the other scores")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--by-top-percent", type=float, help="Level of the other scores; default --top-percent")
    parser.add_argument("--window-hours", type=int, default=24, help="Hours before and after with no flag of the others")
    parser.add_argument("--hourly-dir", type=Path, required=True,
                        help="Directory with a folder of monthly NetCDF files of hourly fields per year")
    parser.add_argument("--unusual-percent", type=float, default=5, help="Unusual: this highest (lowest) share")
    args = parser.parse_args(argv)
    by_top = args.top_percent if args.by_top_percent is None else args.by_top_percent
    timestamps, first, unreached, calm, _ = select(args.score_dir, args.by_dir, args.top_percent, by_top,
                                                   args.window_hours)
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    latitudes = np.asarray(grid["latitudes"], dtype=float)[rows]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[cols]
    sampled = np.zeros(len(timestamps), dtype=bool)
    sampled[::SAMPLE_STEP] = True
    groups = [np.nonzero(unreached), np.nonzero(first & sampled[:, None]), np.nonzero(calm & sampled[:, None])]
    del first, unreached, calm
    month_of = timestamps.month.to_numpy()

    variables = None
    hits = defined = lengths = ratio_sum = ratio_count = near_centre = None
    for month in range(1, 13):
        if not (month_of == month).any():
            continue
        parts = [read_month(args.hourly_dir, year, month, latitudes, longitudes)
                 for year in sorted(set(timestamps.year[month_of == month]))]
        variables = parts[0][2] if variables is None else variables
        if any(part[2] != variables for part in parts):
            raise ValueError(f"The files of month {month} do not hold the variables of the other months")
        hourly_times = parts[0][0].append([part[0] for part in parts[1:]])
        values = np.concatenate([part[1] for part in parts])
        del parts
        hours = hourly_times.hour.to_numpy()
        high, known = unusual_values(values, hours, args.unusual_percent)
        low, _ = unusual_values(values, hours, args.unusual_percent, low=True)
        if hits is None:
            shape = (len(GROUPS), len(TAILS), len(OFFSETS), len(variables))
            hits, defined = np.zeros(shape), np.zeros(shape)
            lengths = np.zeros((len(GROUPS), len(variables), len(OFFSETS)))  # run lengths 1 to 7
            ratio_sum, ratio_count, near_centre = (np.zeros((len(GROUPS), len(variables))) for _ in range(3))
        for g, (steps, cells) in enumerate(groups):
            here = month_of[steps] == month
            when, where = timestamps[steps[here]], cells[here]
            index = np.stack([hourly_times.get_indexer(when + pd.Timedelta(hours=offset)) for offset in OFFSETS], axis=1)
            present = index >= 0
            safe = np.where(present, index, 0)
            for t, unusual in enumerate((high, low)):
                for o in range(len(OFFSETS)):
                    use = present[:, o]
                    hits[g, t, o] += unusual[safe[use, o], where[use]].sum(axis=0)
                    defined[g, t, o] += known[safe[use, o], where[use]].sum(axis=0)
            whole = present.all(axis=1)
            for v in range(len(variables)):
                rows_v = (safe[whole], where[whole][:, None])
                series = values[rows_v[0], rows_v[1], v]
                up, down = high[rows_v[0], rows_v[1], v], low[rows_v[0], rows_v[1], v]
                for unusual in (up, down):
                    chosen = unusual[:, CENTRE]
                    lengths[g, v] += np.bincount(run_length(unusual[chosen]) - 1, minlength=len(OFFSETS))
                chosen = up[:, CENTRE] | down[:, CENTRE]
                ratio, centre = abruptness(series[chosen])
                ratio_sum[g, v] += np.nansum(ratio)
                ratio_count[g, v] += np.count_nonzero(np.isfinite(ratio))
                near_centre[g, v] += np.count_nonzero(centre)
        del values, high, low, known

    print(f"[unreached-hourly] {timestamps[0].date()} to {timestamps[-1].date()} | flagged: the highest "
          f"{args.top_percent:g}% of the first score, {by_top:g}% of the others | unreached: no other score within "
          f"{args.window_hours} h | unusual: highest or lowest {args.unusual_percent:g}% for the cell, month and hour")
    print("values: " + ", ".join(f"{name} {len(steps)}" for name, (steps, _) in zip(GROUPS, groups))
          + f" (the last two: one timestamp every {SAMPLE_STEP})")
    with np.errstate(invalid="ignore", divide="ignore"):
        share = 100 * hits / defined
        for t, tail in enumerate(TAILS):
            print(f"\n% of the values with the variable in its {tail}est {args.unusual_percent:g}%, hours from the value "
                  f"(by chance {args.unusual_percent:g}%)")
            print(f"{'group, variable':<{NAME}}" + "".join(f"{f'{offset:+d}h':>{WIDTH}}" for offset in OFFSETS))
            for v, variable in enumerate(variables):
                for g, group in enumerate(GROUPS):
                    print(f"{f'{group[:18]}, {variable}'[:NAME - 1]:<{NAME}}"
                          + "".join(f"{share[g, t, o, v]:>{WIDTH}.1f}" for o in range(len(OFFSETS))))
        print("\nhours in a row unusual on the same side, among the values unusual at their hour: % with each length")
        print(f"{'group, variable':<{NAME}}" + "".join(f"{f'{hours}h':>{WIDTH}}" for hours in range(1, len(OFFSETS) + 1))
              + f"{'mean h':>{WIDTH}}{'values':>{WIDTH + 2}}")
        for v, variable in enumerate(variables):
            for g, group in enumerate(GROUPS):
                count = lengths[g, v].sum()
                mean = (lengths[g, v] * np.arange(1, len(OFFSETS) + 1)).sum() / count
                print(f"{f'{group[:18]}, {variable}'[:NAME - 1]:<{NAME}}"
                      + "".join(f"{100 * value / count:>{WIDTH}.1f}" for value in lengths[g, v])
                      + f"{mean:>{WIDTH}.2f}{count:>{WIDTH + 2}.0f}")
        print(f"\nabrupt change, among the values unusual at their hour: largest hourly change over the sum of the six "
              f"(1/6 = {1 / 6:.2f} if even, 1 if in one hour), and % with the largest change next to the value")
        print(f"{'group, variable':<{NAME}}{'ratio':>{WIDTH}}{'% next':>{WIDTH}}{'values':>{WIDTH + 2}}")
        for v, variable in enumerate(variables):
            for g, group in enumerate(GROUPS):
                print(f"{f'{group[:18]}, {variable}'[:NAME - 1]:<{NAME}}{ratio_sum[g, v] / ratio_count[g, v]:>{WIDTH}.3f}"
                      f"{100 * near_centre[g, v] / ratio_count[g, v]:>{WIDTH}.1f}{ratio_count[g, v]:>{WIDTH + 2}.0f}")
    print("\nthe hourly fields are from the same reanalysis: the check describes the anomaly, it does not confirm it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
