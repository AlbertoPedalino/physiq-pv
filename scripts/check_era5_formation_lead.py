"""Where the other score flags weather after a calm spell around a cell, did the model flag it there before?

A value is flagged by a score when it is in its highest share (`--top-percent` for the first score,
`--by-top-percent` for the second). A birth is a cell the second score flags after `--quiet-hours`
without any flag of its own within `--quiet-radius` cells (the larger of the row and column offsets):
new weather where the area was calm, not weather that moves in from a neighbour. A birth lasts when
the second score still flags a cell within `--quiet-radius` LASTING_HOURS later, so the weather
grows into an episode; it ends otherwise.

For every birth and every radius in RADII, the check looks whether a watcher flags a cell within that
radius at the birth and within each window of WINDOWS_HOURS before it. Before a birth the second score
flags nothing within the quiet radius, so every flag of a watcher there comes before the second.
Watchers:
- the first score (the model);
- with `--early-percent`, the second score at that lower level (e.g. 5% for births at 1%): does the
  weather show in the simple score first, before it becomes extreme;
- the first score where the second at `--early-percent` does not flag: what only the model adds.

What happens by chance: the same shares at cells where the second score was as quiet and did not give
a birth (one timestamp every CHANCE_STEP). A share above chance before the birth is a lead on the
formation of the weather, not on its arrival.

The whole scored period is read. With `--csv` the shares go to a file.

    python scripts/check_era5_formation_lead.py --score-dir outputs/run_gen_clim --by-dir outputs/run_simple_max \
        --top-percent 5 --by-top-percent 1 --early-percent 5 --csv outputs/formation_1.csv
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

from scripts.check_era5_lead import flags

CHUNK = 256  # timestamps read at a time
RADII = (0, 2, 5)  # cells around the birth where a watcher is looked for
WINDOWS_HOURS = (3, 6, 12, 24)  # hours before the birth
LASTING_HOURS = 12
CHANCE_STEP = 29  # the chance row reads one timestamp every this many
GROUPS = ("births that last", "births that end", "chance")
NAME, WIDTH = 34, 10


def near(frames, radius, rows, cols, shape):
    """[time, cell]: some cell within `radius` cells of each cell is flagged in `frames` [time, cell]."""
    if radius == 0:
        return np.asarray(frames, dtype=bool)
    lattice = np.zeros((len(frames), *shape), dtype=np.uint8)
    lattice[:, rows, cols] = frames
    lattice = ndimage.maximum_filter1d(lattice, 2 * radius + 1, axis=1, mode="constant")
    lattice = ndimage.maximum_filter1d(lattice, 2 * radius + 1, axis=2, mode="constant")
    return lattice[:, rows, cols].astype(bool)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score (the model)")
    parser.add_argument("--by-dir", type=Path, required=True, help="Run with the second score")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--by-top-percent", type=float, help="Level of the second score; default --top-percent")
    parser.add_argument("--early-percent", type=float, help="Also watch the second score at this lower level")
    parser.add_argument("--quiet-hours", type=int, default=24, help="Hours without a flag of the second score around a birth")
    parser.add_argument("--quiet-radius", type=int, default=5, help="Cells around a birth without a flag of the second score")
    parser.add_argument("--csv", type=Path, help="Also write the shares here")
    args = parser.parse_args(argv)
    first, timestamps, _ = flags(args.score_dir, args.top_percent)
    by_top = args.top_percent if args.by_top_percent is None else args.by_top_percent
    second, other_timestamps, _ = flags(args.by_dir, by_top)
    if not timestamps.equals(other_timestamps) or first.shape != second.shape:
        raise ValueError("The two scores are not over the same timestamps and cells")
    early = None
    if args.early_percent is not None:
        if args.early_percent <= by_top:
            raise ValueError("--early-percent must be a lower level (a larger share) than --by-top-percent")
        early = flags(args.by_dir, args.early_percent)[0]
    step_hours = int(round((timestamps[1] - timestamps[0]) / pd.Timedelta(hours=1)))
    if any(hours % step_hours for hours in (*WINDOWS_HOURS, LASTING_HOURS, args.quiet_hours)):
        raise ValueError(f"The hours must be multiples of {step_hours}")
    if max(WINDOWS_HOURS) > args.quiet_hours:
        raise ValueError("The windows must not reach before the quiet hours")
    radii = [radius for radius in RADII if radius <= args.quiet_radius]
    quiet, lasting = args.quiet_hours // step_hours, LASTING_HOURS // step_hours
    windows = [hours // step_hours for hours in WINDOWS_HOURS]
    cap = max(windows) + 1
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    if len(rows) != first.shape[1]:
        raise ValueError("The grid of the run does not match the cells of the scores")
    names = [path.name for path in (args.score_dir, args.by_dir)]
    common = len(os.path.commonprefix(names))
    short = [name[common:] or name for name in names]
    watchers = [(short[0], lambda part: first[part])]
    if early is not None:
        watchers += [(f"{short[1]} at {args.early_percent:g}%", lambda part: early[part]),
                     (f"{short[0]} without {short[1]} at {args.early_percent:g}%", lambda part: first[part] & ~early[part])]

    times, cells = second.shape
    calm = np.zeros(cells, dtype=np.int32)  # steps before this one without a flag of the second within the quiet radius
    since = np.full((len(watchers), len(radii), cells), cap, dtype=np.int32)  # steps since a watcher flagged nearby
    before = np.zeros((len(GROUPS), len(watchers), len(radii), cap + 1))  # [group, watcher, radius, steps since]
    at_birth = np.zeros((len(GROUPS), len(watchers), len(radii)))
    counts, clusters = np.zeros(len(GROUPS)), 0
    for start in range(0, times, CHUNK):
        part = slice(start, min(start + CHUNK, times))
        flagged = second[part]
        around = near(flagged, args.quiet_radius, rows, cols, shape)
        ahead = second[start + lasting:part.stop + lasting]
        later = near(ahead, args.quiet_radius, rows, cols, shape) if len(ahead) else ahead
        watched = [[near(watch(part), radius, rows, cols, shape) for radius in radii] for _, watch in watchers]
        for index in range(part.stop - start):
            step = start + index
            if step >= quiet:
                chosen, groups = [], []
                birth = np.nonzero(flagged[index] & (calm >= quiet))[0]
                if step + lasting < times and len(birth):
                    chosen.append(birth)
                    groups.append(np.where(later[index][birth], 0, 1))
                    lattice = np.zeros(shape, dtype=bool)
                    lattice[rows[birth], cols[birth]] = True
                    clusters += ndimage.label(lattice, structure=np.ones((3, 3)))[1]
                if not step % CHANCE_STEP:
                    others = np.nonzero(~flagged[index] & (calm >= quiet))[0]
                    chosen.append(others)
                    groups.append(np.full(len(others), 2))
                if chosen:
                    where, group = np.concatenate(chosen), np.concatenate(groups).astype(np.int64)
                    counts += np.bincount(group, minlength=len(GROUPS))
                    for w in range(len(watchers)):
                        for r in range(len(radii)):
                            before[:, w, r] += np.bincount(group * (cap + 1) + since[w, r, where],
                                                           minlength=len(GROUPS) * (cap + 1)).reshape(len(GROUPS), cap + 1)
                            at_birth[:, w, r] += np.bincount(group, weights=watched[w][r][index][where].astype(float),
                                                             minlength=len(GROUPS))
            calm = np.where(around[index], 0, np.minimum(calm + 1, quiet)).astype(np.int32)
            for w in range(len(watchers)):
                for r in range(len(radii)):
                    since[w, r] = np.where(watched[w][r][index], 1, np.minimum(since[w, r] + 1, cap))

    print(f"[formation] {timestamps[0].date()} to {timestamps[-1].date()} | birth: {short[1]} at its highest {by_top:g}% "
          f"flags a cell after {args.quiet_hours} h without a flag within {args.quiet_radius} cells | it lasts: a flag "
          f"within {args.quiet_radius} cells {LASTING_HOURS} h later | watchers flagged at: {short[0]} "
          f"{args.top_percent:g}%" + (f", {short[1]} {args.early_percent:g}%" if early is not None else ""))
    print(f"births: {int(counts[0] + counts[1])} cells in {clusters} groups of touching cells at one timestamp; that "
          f"last {int(counts[0])}, that end {int(counts[1])} | chance: {int(counts[2])} quiet cells without a birth")
    labels = ["at birth", *[f"{hours}h" for hours in WINDOWS_HOURS]]
    rows_out = []
    # All births: the first two groups together.
    shares = np.concatenate([before[:2].sum(axis=0, keepdims=True), before], axis=0)
    births_at = np.concatenate([at_birth[:2].sum(axis=0, keepdims=True), at_birth], axis=0)
    sizes = np.concatenate([counts[:2].sum(keepdims=True), counts])
    group_names = ("all births", *GROUPS)
    with np.errstate(invalid="ignore", divide="ignore"):
        for w, (watcher, _) in enumerate(watchers):
            print(f"\n{watcher}: % of the cells where it flags a cell within the radius, at the birth and at least once "
                  f"in the hours before it")
            print(f"{'radius, group':<{NAME}}" + "".join(f"{label:>{WIDTH}}" for label in labels)
                  + f"{'3h x':>{WIDTH}}{'24h x':>{WIDTH}}")
            for r, radius in enumerate(radii):
                table = np.empty((len(group_names), len(labels)))
                for g in range(len(group_names)):
                    cumulative = np.cumsum(shares[g, w, r])
                    table[g] = 100 * np.array([births_at[g, w, r], *[cumulative[window] for window in windows]]) / sizes[g]
                for g, group in enumerate(group_names):
                    ratios = table[g, [1, -1]] / table[-1, [1, -1]] if g < len(group_names) - 1 else [np.nan, np.nan]
                    print(f"{f'{radius}, {group}'[:NAME - 1]:<{NAME}}"
                          + "".join(f"{'-' if np.isnan(value) else format(value, '.1f'):>{WIDTH}}" for value in table[g])
                          + "".join(f"{'' if np.isnan(value) else format(value, '.2f'):>{WIDTH}}" for value in ratios))
                    for label, value in zip(labels, table[g]):
                        rows_out.append({"watcher": watcher, "radius": radius, "group": group, "when": label,
                                         "share": value, "cells": int(sizes[g]), "top_percent": args.top_percent,
                                         "by_top_percent": by_top, "early_percent": args.early_percent})
    print(f"at birth: the watcher flags a cell within the radius at the timestamp of the birth; 3h ... 24h: at least once "
          f"within those hours before it (the birth itself not counted); 3h x, 24h x: over chance")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(args.csv, index=False)
        print(f"written {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
