"""Before an episode of a score starts at a cell, does another score already flag the cells around it?

A value is flagged by a score when it is in its highest `--top-percent`. An episode of a score starts
at a cell when the score flags it after at least `--quiet-hours` without a flag there, as in
`check_era5_lead.py`. That check looks at the same cell; this one looks around it. For every start
of an episode of the first score, and for every ring of cells around the starting cell (RINGS, in
cells of the grid, counted as the larger of the row and column offsets) and every lag before the
start (LAGS_HOURS), the check asks whether a precursor flagged at least one cell of the ring at that
lag. Three precursors:
- the other score;
- the score of the episodes itself: the episode moving in from the cells around, the reference;
- the other score where the score of the episodes does not flag: what only the other sees.

What happens by chance: the same reading at cells and timestamps where the score of the episodes has
been quiet as long as before a start and does not start (one timestamp every CHANCE_STEP). A ring
holds more cells the farther it is, so its chance is higher too: a share means something only
against its chance. The same is done the other way round, with the starts of the second score.

The table gives, for every precursor and ring, the share of the starts with a flag in the ring at
each lag, and the same share by chance. With `--csv` the same numbers go to a file, one row per
precursor, ring and lag.

    python scripts/check_era5_spatial_lead.py --score-dir outputs/run_gen_clim --by-dir outputs/run_simple_max \
        --csv outputs/spatial_lead_5.csv
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

CHUNK = 256  # timestamps whose rings are computed at a time
CHANCE_STEP = 29  # the chance reading takes one timestamp every this many
RINGS = ((0, 0), (1, 2), (3, 5), (6, 10))  # cells of 0.5 degrees: the same cell, then about 55-110, 165-280, 330-560 km
LAGS_HOURS = (0, 3, 6, 12, 24)  # hours before the start; 0 is the timestamp of the start
NAME, WIDTH = 46, 9


def ring_flags(maps, rings):
    """For maps [time, row, col] of flags: [ring, time, row, col], whether a flag lies in each ring around a cell.

    A ring (low, high) holds the offsets whose larger component is from low to high; counted with box sums
    of the flags, a box minus the box inside it."""
    reach = max(high for _, high in rings)
    times, height, width = maps.shape
    padded = np.zeros((times, height + 2 * reach + 1, width + 2 * reach + 1), dtype=np.int32)
    padded[:, reach + 1:reach + 1 + height, reach + 1:reach + 1 + width] = maps
    total = padded.cumsum(axis=1).cumsum(axis=2)  # total[:, i, j]: flags with padded row <= i and column <= j

    def box(distance):
        # Flags within `distance` of every cell: the cell sits at padded (row + reach + 1, col + reach + 1).
        if distance < 0:
            return np.zeros((times, height, width), dtype=np.int32)
        inner, outer = reach - distance, reach + 1 + distance  # last padded index before the box, last inside it
        return (total[:, outer:outer + height, outer:outer + width]
                - total[:, inner:inner + height, outer:outer + width]
                - total[:, outer:outer + height, inner:inner + width]
                + total[:, inner:inner + height, inner:inner + width])

    return np.stack([(box(high) - box(low - 1)) > 0 for low, high in rings])


def starts(first, quiet, chance=False):
    """Cells where an episode of `first` [time, cell] starts at each timestamp, after `quiet` steps without a
    flag. With `chance`, the quiet cells where it does not start, one timestamp every CHANCE_STEP."""
    times, cells = first.shape
    silent = np.full(cells, np.int32(quiet))
    positions = []
    for step in range(times):
        start = (~first[step] if chance else first[step]) & (silent >= quiet)
        if step < quiet or (chance and step % CHANCE_STEP):
            start[:] = False  # what came before the first timestamps is unknown; chance: not a sampled step
        positions.append(np.nonzero(start)[0])
        silent = np.where(first[step], 0, np.minimum(silent + 1, quiet)).astype(np.int32)
    return positions


def precursor_counts(precursors, positions, rows, cols, shape, lags):
    """Number of positions with a flag of each precursor in each ring at each lag: [precursor, ring, lag]."""
    longest = max(lags)
    counts = np.zeros((len(precursors), len(RINGS), len(lags)), dtype=np.int64)
    times = len(positions)
    for begin in range(0, times, CHUNK):
        end = min(begin + CHUNK, times)
        first = max(begin - longest, 0)
        rings = []
        for flagged in precursors:
            maps = np.zeros((end - first, *shape), dtype=bool)
            maps[:, rows, cols] = flagged[first:end]
            rings.append(ring_flags(maps, RINGS)[:, :, rows, cols])  # [ring, time, cell]
        rings = np.stack(rings)  # [precursor, ring, time, cell]
        for step in range(begin, end):
            where = positions[step]
            if not len(where):
                continue
            for index, lag in enumerate(lags):
                if step - lag >= first:
                    counts[:, :, index] += rings[:, :, step - lag - first][:, :, where].sum(axis=-1)
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score (the model)")
    parser.add_argument("--by-dir", type=Path, required=True, help="Run with the second score")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--by-top-percent", type=float, help="Level of the second score; default --top-percent")
    parser.add_argument("--quiet-hours", type=int, default=24, help="Hours without a flag before a start")
    parser.add_argument("--csv", type=Path, help="Also write the shares here, one row per precursor, ring and lag")
    args = parser.parse_args(argv)
    first, timestamps, _ = flags(args.score_dir, args.top_percent)
    by_top = args.top_percent if args.by_top_percent is None else args.by_top_percent
    second, other_timestamps, _ = flags(args.by_dir, by_top)
    if not timestamps.equals(other_timestamps) or first.shape != second.shape:
        raise ValueError("The two scores are not over the same timestamps and cells")
    step_hours = int(round((timestamps[1] - timestamps[0]) / pd.Timedelta(hours=1)))
    if args.quiet_hours % step_hours or any(lag % step_hours for lag in LAGS_HOURS):
        raise ValueError(f"quiet-hours and the lags must be multiples of {step_hours}")
    quiet, lags = args.quiet_hours // step_hours, [lag // step_hours for lag in LAGS_HOURS]
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    if len(rows) != first.shape[1]:
        raise ValueError("The grid of the run does not match the cells of the scores")
    names = [path.name for path in (args.score_dir, args.by_dir)]
    common = len(os.path.commonprefix(names))
    short = [name[common:] or name for name in names]
    print(f"[spatial-lead] {timestamps[0].date()} to {timestamps[-1].date()} | flagged: the highest "
          f"{args.top_percent:g}% of {short[0]}, {by_top:g}% of {short[1]} | start of an episode: a flag after "
          f"{args.quiet_hours} h without one at the cell | rings in cells of the grid, the larger of the row and "
          f"column offsets")
    rows_out = []
    for index, (one, other) in enumerate(((first, second), (second, first))):
        labels = [short[1 - index], f"{short[index]} (itself)", f"{short[1 - index]} not {short[index]}"]
        precursors = [other, one, other & ~one]
        found = {chance: starts(one, quiet, chance=chance) for chance in (False, True)}
        totals = {chance: sum(len(where) for where in found[chance]) for chance in (False, True)}
        counts = {chance: precursor_counts(precursors, found[chance], rows, cols, shape, lags)
                  for chance in (False, True)}
        print(f"\nstarts of {short[index]}: {totals[False]} | chance: {totals[True]} quiet cell-timestamps | % with a "
              f"flag of the precursor in the ring, hours before the start")
        print(f"{'precursor, ring (cells)':<{NAME}}" + "".join(f"{f'{lag}h':>{WIDTH}}" for lag in LAGS_HOURS)
              + "".join(f"{f'ch {lag}h':>{WIDTH}}" for lag in LAGS_HOURS))
        for p, label in enumerate(labels):
            for r, (low, high) in enumerate(RINGS):
                shares = {chance: 100 * counts[chance][p, r] / max(totals[chance], 1) for chance in (False, True)}
                ring = f"{low}" if low == high else f"{low}-{high}"
                print(f"{f'{label}, {ring}'[:NAME - 1]:<{NAME}}" + "".join(f"{value:>{WIDTH}.1f}" for value in shares[False])
                      + "".join(f"{value:>{WIDTH}.1f}" for value in shares[True]))
                for lag, share, chance in zip(LAGS_HOURS, shares[False], shares[True]):
                    rows_out.append({"starts_of": short[index], "precursor": label, "ring_low": low, "ring_high": high,
                                     "lag_hours": lag, "share": share, "chance": chance, "starts": totals[False],
                                     "chance_points": totals[True], "top_percent": args.top_percent if index == 0 else by_top,
                                     "by_top_percent": by_top if index == 0 else args.top_percent})
    print("ch: the same share where the score of the episodes was quiet as long and did not start; the ring of the "
          "score itself at 0 cells is empty before a start by definition")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(args.csv, index=False)
        print(f"written {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
