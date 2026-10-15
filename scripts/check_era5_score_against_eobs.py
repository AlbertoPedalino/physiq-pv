"""Where a score flags alone, did the stations measure unusual weather that day?

The weather is a daily variable of E-OBS on the cells of the run (`download_eobs.py`): the precipitation,
the sea level pressure, the solar radiation or the wind, built from station observations and not from
ERA5, so a judge that none of the scores saw. A cell-day is flagged by a
score when the score is in its highest `--top-percent` at one or more timestamps of that UTC day. The
cell-days are in groups: flagged by the first score and by none of the others, by the first and at least
one other, by the others only, by none; and the ring, the cell-days next to (within `--ring` cells) a
cell-day of the first score alone, flagged by no score: the weather around it.

The variable of a cell-day is unusual when it is in its highest `--unusual-percent` (with `--tail low`
its lowest) for the cell and the calendar month over the days read; with `--change-days` the value is
its change from that many days before, so that `--variable pressure --change-days 1 --tail low` asks for
the strongest drops of the pressure in a day. For every group the check says how many of its cell-days
are unusual on the same day, on the next day (the E-OBS rain of many countries ends at 06 UTC of the next
day, so the rain of an evening can fall in it), and on either. About `--unusual-percent` if the group has
nothing to do with that weather, more if it flags it. Only cell-days with a measured value count.

Cell-days next to each other are not independent, so the count alone overstates how sure a ratio is.
Below the table, the ratios of the first score alone to all cell-days, to its ring and to the others
only, with the 95% interval of `--bootstrap` draws of whole years (with replacement): years are far
enough apart to be nearly independent. A ratio whose interval stays above 1 is above chance for sure.

    python scripts/check_era5_score_against_eobs.py --eobs-dir data/eobs --variable rain \
        --score-dir outputs/run_clim --by-dirs outputs/run_no_model_clim outputs/run_previous_clim
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import numpy as np
import pandas as pd
from scipy import ndimage

from scripts.check_era5_dust_lead import THRESHOLD_STRIDE, score_file

GROUPS = ("score alone", "score and another", "others only", "none", "ring around score alone", "all cell-days")
MEASURES = ("same day", "next day", "either")
RATIOS = (("score alone", "all cell-days"), ("score alone", "ring around score alone"), ("score alone", "others only"),
          ("score and another", "all cell-days"))
NAME, WIDTH = 28, 14
DAYS_AT_A_TIME = 32


def daily_flags(scores, limits, timestamps, days):
    """For every day of `days` and cell: is the first score flagged at some timestamp of that UTC day, and is
    any of the others; and are all the scores finite at every timestamp of the day."""
    day_of = timestamps.normalize()
    first = np.zeros((len(days), scores[0].shape[1]), dtype=bool)
    others, finite = np.zeros_like(first), np.ones_like(first)
    positions = pd.Index(days).get_indexer(day_of)
    for start in range(0, len(days), DAYS_AT_A_TIME):
        rows = np.nonzero((positions >= start) & (positions < start + DAYS_AT_A_TIME))[0]
        if not len(rows):
            continue
        where = positions[rows]
        values = [np.asarray(array[rows[0]:rows[-1] + 1]) for array in scores]
        with np.errstate(invalid="ignore"):
            marks = [block > limit for block, limit in zip(values, limits)]
        ok = np.logical_and.reduce([np.isfinite(block) for block in values])
        np.logical_or.at(first, where, marks[0])
        np.logical_or.at(others, where, np.logical_or.reduce(marks[1:]))
        np.logical_and.at(finite, where, ok)
    covered = np.zeros(len(days), dtype=bool)
    covered[np.unique(positions[positions >= 0])] = True
    finite &= covered[:, None]
    return first, others, finite


def unusual_values(values, days, percent, low=False, positive_only=False):
    """Where the values [days, cells] are in their highest `percent` (or with `low` their lowest) for the cell
    and the calendar month. With `positive_only`, as for the rain, a month whose threshold is zero flags
    nothing."""
    unusual = np.zeros(values.shape, dtype=bool)
    months = pd.DatetimeIndex(days).month.to_numpy()
    for month in range(1, 13):
        chosen = months == month
        if not chosen.any():
            continue
        block = values[chosen]
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)  # cells never measured: NaN thresholds, nothing flagged
            limit = np.nanpercentile(block, percent if low else 100 - percent, axis=0)
            unusual[chosen] = (block < limit) if low else (block > limit)
            if positive_only:
                unusual[chosen] &= limit > 0
    return unusual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--eobs-dir", "--rain-dir", dest="eobs_dir", type=Path, required=True,
                        help="Output of download_eobs.py")
    parser.add_argument("--variable", choices=("rain", "pressure", "radiation", "wind"), default="rain")
    parser.add_argument("--tail", choices=("high", "low"), default="high", help="Unusual: the highest or the lowest values")
    parser.add_argument("--change-days", type=int, default=0, help="Ask the change from this many days before")
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score")
    parser.add_argument("--by-dirs", type=Path, nargs="+", required=True, help="Runs with the other scores")
    parser.add_argument("--top-percent", type=float, default=5, help="Flagged: this highest share of each score")
    parser.add_argument("--unusual-percent", "--rain-percent", dest="unusual_percent", type=float, default=5,
                        help="Unusual: this highest (or lowest) share of the days")
    parser.add_argument("--ring", type=int, default=1, help="Width of the ring around the cell-days of the score alone")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Draws of whole years for the intervals of the ratios")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.ring < 1 or args.change_days < 0:
        raise ValueError("ring must be at least 1 and change-days not negative")
    manifest = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))
    grid = manifest["grid"]
    rows, cols = np.asarray(grid["rows"]), np.asarray(grid["cols"])
    lattice = (len(grid["latitudes"]), len(grid["longitudes"]))
    timestamps = pd.DatetimeIndex(np.load(args.score_dir / "test_timestamps.npy"))
    observed = np.load(args.eobs_dir / f"{args.variable}.npy")
    observed_days = pd.DatetimeIndex(np.load(args.eobs_dir / "days.npy").astype("datetime64[D]"))
    if observed.shape[1] != len(rows) or len(observed) != len(observed_days):
        raise ValueError(f"The {args.variable} is not over the cells of the score or its days")
    if args.change_days:
        if not (np.diff(observed_days.values).astype("timedelta64[D]").astype(int) == 1).all():
            raise ValueError("The days of the stations are not consecutive: no change can be read")
        change = np.full_like(observed, np.nan)
        change[args.change_days:] = observed[args.change_days:] - observed[:-args.change_days]
        observed = change
    paths = [args.score_dir, *args.by_dirs]
    scores = []
    try:
        for path in paths:
            if not pd.DatetimeIndex(np.load(path / "test_timestamps.npy")).equals(timestamps):
                raise ValueError(f"The score is not over the same timestamps: {path}")
            run = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            scores.append(np.load(path / score_file(run, "combined"), mmap_mode="r"))
            if scores[-1].shape != (len(timestamps), len(rows)):
                raise ValueError(f"The score is not over the cells: {path}")
        limits = [float(np.nanpercentile(np.asarray(array[::THRESHOLD_STRIDE]), 100 - args.top_percent)) for array in scores]
        days = observed_days[(observed_days >= timestamps[0].normalize()) & (observed_days <= timestamps[-1].normalize())]
        first, others, finite = daily_flags(scores, limits, timestamps, days)
    finally:
        for array in scores:
            array._mmap.close()
    observed = observed[observed_days.get_indexer(days)]
    unusual = unusual_values(observed, days, args.unusual_percent, low=args.tail == "low",
                             positive_only=args.variable == "rain" and not args.change_days)
    measured = np.isfinite(observed)
    # The next day: the last day has none.
    unusual_next = np.zeros_like(unusual)
    unusual_next[:-1] = unusual[1:]
    measured_next = np.zeros_like(measured)
    measured_next[:-1] = measured[1:]
    alone = first & ~others & finite
    stack = np.zeros((len(days), *lattice), dtype=bool)
    stack[:, rows, cols] = alone
    around = ndimage.binary_dilation(stack, structure=np.ones((1, 3, 3), dtype=bool), iterations=args.ring)[:, rows, cols]
    masks = (alone, first & others & finite, ~first & others & finite, ~first & ~others & finite,
             around & ~first & ~others & finite, finite)
    names = [path.name for path in paths]
    common = len(os.path.commonprefix(names))
    asked = (f"change of the {args.variable} over {args.change_days} day(s)" if args.change_days else args.variable)
    print(f"[eobs] {days[0].date()} to {days[-1].date()} | flagged: the highest {args.top_percent:g}% of "
          f"each score, at one timestamp of the UTC day at least | first: {names[0][common:] or names[0]} | others: "
          f"{', '.join(name[common:] or name for name in names[1:])} | unusual {asked}: "
          f"{'lowest' if args.tail == 'low' else 'highest'} {args.unusual_percent:g}% for the cell and the month | "
          f"cells with stations: {int(measured.any(axis=0).sum())} of {len(rows)}")
    print(f"\n% of the cell-days with unusual {asked} measured by the stations")
    print(f"{'':<{NAME}}" + "".join(f"{name:>{WIDTH}}" for name in ("cell-days", "same day", "next day", "either")))
    years = pd.DatetimeIndex(days).year.to_numpy()
    year_list = np.unique(years)
    # Hits and cell-days [group, measure, year].
    hits, counts = (np.zeros((len(GROUPS), len(MEASURES), len(year_list))) for _ in range(2))
    rows_out = {}
    for index, (name, mask) in enumerate(zip(GROUPS, masks)):
        same = mask & measured
        both = same & measured_next
        pairs = ((unusual & same, same), (unusual_next & mask & measured_next, mask & measured_next),
                 ((unusual | unusual_next) & both, both))
        for measure, (hit, count) in enumerate(pairs):
            for position, year in enumerate(year_list):
                chosen = years == year
                hits[index, measure, position] = hit[chosen].sum()
                counts[index, measure, position] = count[chosen].sum()
        values = [100 * hits[index, measure].sum() / max(counts[index, measure].sum(), 1) for measure in range(3)]
        rows_out[name] = values
        print(f"{name:<{NAME}}{int(same.sum()):>{WIDTH}}" + "".join(f"{value:>{WIDTH}.1f}" for value in values))
    print("same day: the UTC day of the flag; next day: the day after (the E-OBS rain often ends at 06 UTC of it); "
          "either: one of the two (only cell-days with both measured)")
    draws = np.random.default_rng(args.seed).integers(0, len(year_list), size=(args.bootstrap, len(year_list)))
    print(f"\nRatios of the shares, with the 95% interval of {args.bootstrap} draws of whole years "
          f"({len(year_list)} years)")
    print(f"{'':<{2 * NAME}}" + "".join(f"{name:>{2 * WIDTH}}" for name in MEASURES))
    for top, bottom in RATIOS:
        a, b = GROUPS.index(top), GROUPS.index(bottom)
        cells = []
        for measure in range(3):
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = ((hits[a, measure].sum() / counts[a, measure].sum())
                         / (hits[b, measure].sum() / counts[b, measure].sum()))
                sampled = ((hits[a, measure][draws].sum(axis=1) / counts[a, measure][draws].sum(axis=1))
                           / (hits[b, measure][draws].sum(axis=1) / counts[b, measure][draws].sum(axis=1)))
            low, high = np.nanpercentile(sampled, [2.5, 97.5]) if np.isfinite(sampled).any() else (np.nan, np.nan)
            cells.append(f"{ratio:.2f} ({low:.2f}-{high:.2f})")
            rows_out[f"{top} / {bottom}"] = rows_out.get(f"{top} / {bottom}", []) + [(ratio, low, high)]
        print(f"{f'{top} / {bottom}':<{2 * NAME}}" + "".join(f"{cell:>{2 * WIDTH}}" for cell in cells))
    return rows_out


if __name__ == "__main__":
    main()
