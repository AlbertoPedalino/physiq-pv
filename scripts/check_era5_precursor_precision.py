"""When the model and a simple score flag a cell where all is calm, how often does an event form there next?

The formation check reads back from the births; this one reads forward, from the flags, as an alarm is
used. A cell is calm at a timestamp when the second score at its event level (`--by-top-percent`, e.g. the
highest 1%) flagged no cell within `--quiet-radius` cells in the `--quiet-hours` before. At a calm cell the
watchers are the first score at `--top-percent` and the second score at its lower level `--early-percent`;
the cell is in one of four cases by which of the two flag it: both, first only, second only, neither. The
outcome is a flag of the second score at its event level within `--radius` cells in the next
`--lead-hours`: an event forms there, since nothing was flagged around before.

For every case the check says how many calm cells it holds and how often the event follows (the precision of
the alarm), and that over the cells where neither flags (the lift). Cells of the same event are not
independent: the shares are those of cell-timestamps, as an alarm map would give them.

With `--all-cells` no cell needs to be calm: every cell counts where the outcome is not there now (the
second score at its event level not flagging the cell, or the weather of the target not unusual at it),
inside and around events too.

    python scripts/check_era5_precursor_precision.py --score-dir outputs/test_gen_clim \
        --by-dir outputs/test_simple_max --top-percent 5 --by-top-percent 1 --early-percent 5
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

CASES = ("both", "first only", "second early only", "neither")
NAME, WIDTH = 22, 13


def spread(frame, rows, cols, shape, radius):
    """[cells] of the cells within `radius` cells (the larger of the row and column offsets) of a flagged one."""
    if radius == 0:
        return frame.copy()
    lattice = np.zeros(shape, dtype=bool)
    lattice[rows[frame], cols[frame]] = True
    return ndimage.binary_dilation(lattice, structure=np.ones((2 * radius + 1,) * 2, dtype=bool))[rows, cols]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score (the model)")
    parser.add_argument("--by-dir", type=Path, required=True, help="Run with the second score")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--by-top-percent", type=float, default=1, help="Event level of the second score")
    parser.add_argument("--early-percent", type=float, default=5, help="Lower level of the second score as a watcher")
    parser.add_argument("--quiet-hours", type=int, default=24)
    parser.add_argument("--quiet-radius", type=int, default=5)
    parser.add_argument("--radius", type=int, nargs="+", default=[0, 2], help="Cells around the alarm where the event counts")
    parser.add_argument("--lead-hours", type=int, nargs="+", default=[3, 6], help="Hours after the alarm")
    parser.add_argument("--target", choices=("simple", "cape", "gust", "cape_or_gust"), default="simple",
                        help="What the event after is: a flag of the second score at its event level, or the CAPE, the "
                             "gust, either of them unusual within the radius where it is not unusual at the cell now")
    parser.add_argument("--unseen-dir", type=Path, help="Monthly files of the CAPE and the gust, for those targets")
    parser.add_argument("--unusual-percent", type=float, default=5, help="Unusual: this highest share of the CAPE or gust")
    parser.add_argument("--all-cells", action="store_true", help="Every cell, not only the calm ones")
    args = parser.parse_args(argv)
    if args.early_percent <= args.by_top_percent:
        raise ValueError("--early-percent must be a lower level (a larger share) than --by-top-percent")
    if args.target != "simple" and args.unseen_dir is None:
        raise ValueError(f"--target {args.target} needs --unseen-dir")
    first, timestamps, _ = flags(args.score_dir, args.top_percent)
    event, other_times, _ = flags(args.by_dir, args.by_top_percent)
    early = flags(args.by_dir, args.early_percent)[0]
    if not timestamps.equals(other_times) or first.shape != event.shape:
        raise ValueError("The two scores are not over the same timestamps and cells")
    step_hours = int(round((timestamps[1] - timestamps[0]) / pd.Timedelta(hours=1)))
    if any(hours % step_hours for hours in (args.quiet_hours, *args.lead_hours)):
        raise ValueError(f"The hours must be multiples of {step_hours}")
    quiet_steps, leads = args.quiet_hours // step_hours, [hours // step_hours for hours in args.lead_hours]
    if args.all_cells:
        quiet_steps = 0
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    # The event flags near every cell: within the quiet radius (for the calm) and within each outcome radius.
    near_quiet = None if args.all_cells else np.stack([spread(frame, rows, cols, shape, args.quiet_radius)
                                                       for frame in event])
    unusual = None
    if args.target == "simple":
        outcome = event
    else:  # weather no score reads, unusual for the cell, month and time of day
        from scripts.nowcast_era5_anomalies import TARGETS, unseen_flags

        latitudes = np.asarray(grid["latitudes"], dtype=float)[rows]
        longitudes = np.asarray(grid["longitudes"], dtype=float)[cols]
        unusual = unseen_flags(args.unseen_dir, timestamps, latitudes, longitudes, args.unusual_percent)[
            :, :, list(TARGETS[args.target])].any(axis=2)
        outcome = unusual
    near_outcome = {radius: outcome if radius == 0 else np.stack([spread(frame, rows, cols, shape, radius)
                                                                  for frame in outcome]) for radius in args.radius}
    counts = np.zeros((len(args.radius), len(leads), len(CASES), 2))  # calm cells, followed by an event
    for t in range(quiet_steps, len(timestamps) - max(leads)):
        # Calm at t: no event flag near the cell in the quiet_steps before t, nor at the event level now; with weather
        # as the target, nor that weather unusual at the cell now.
        if near_quiet is not None:
            calm = ~near_quiet[t - quiet_steps:t].any(axis=0) & ~event[t]
        else:  # every cell where the outcome is not there now
            calm = ~event[t] if unusual is None else np.ones(event.shape[1], dtype=bool)
        if unusual is not None:
            calm &= ~unusual[t]
        if not calm.any():
            continue
        case = np.where(first[t] & early[t], 0, np.where(first[t], 1, np.where(early[t], 2, 3)))[calm]
        for r, radius in enumerate(args.radius):
            for l, lead in enumerate(leads):
                followed = near_outcome[radius][t + 1:t + lead + 1].any(axis=0)[calm]
                counts[r, l, :, 0] += np.bincount(case, minlength=len(CASES))
                counts[r, l, :, 1] += np.bincount(case, weights=followed, minlength=len(CASES))
    names = [path.name for path in (args.score_dir, args.by_dir)]
    common = len(os.path.commonprefix(names))
    short = [name[common:] or name for name in names]
    print(f"[precursor] {timestamps[0].date()} to {timestamps[-1].date()} | "
          + ("cells: all where the event is not there now" if args.all_cells else
             f"calm: no flag of {short[1]} at its highest {args.by_top_percent:g}% within {args.quiet_radius} cells in "
             f"the {args.quiet_hours} h before") + " | watchers: "
          f"{short[0]} at {args.top_percent:g}%, {short[1]} at {args.early_percent:g}% | event: "
          + (f"a flag of {short[1]} at {args.by_top_percent:g}%" if args.target == "simple" else
             f"{args.target.replace('_', ' ')} in its highest {args.unusual_percent:g}%, not unusual at the cell now")
          + " within the radius in the hours after")
    print(f"\n{'radius, lead, case':<{NAME + 12}}{'calm cells':>{WIDTH}}{'% of calm':>{WIDTH}}{'event after %':>{WIDTH}}"
          f"{'lift':>{WIDTH}}")
    result = {}
    for r, radius in enumerate(args.radius):
        for l, hours in enumerate(args.lead_hours):
            total = counts[r, l, :, 0].sum()
            base = counts[r, l, 3, 1] / counts[r, l, 3, 0] if counts[r, l, 3, 0] else np.nan
            for c, name in enumerate(CASES):
                cells, followed = counts[r, l, c]
                precision = followed / cells if cells else np.nan
                lift = precision / base if base else np.nan
                result[(radius, hours, name)] = (int(cells), 100 * precision, lift)
                label = f"{radius} cell(s), {hours}h, {name}"
                print(f"{label:<{NAME + 12}}{int(cells):>{WIDTH}d}{100 * cells / total:>{WIDTH}.2f}"
                      f"{100 * precision:>{WIDTH}.2f}{lift:>{WIDTH}.1f}")
            print()
    print(f"event after: % of the calm cells of the case followed by an event; lift: over the cells where neither flags")
    return result


if __name__ == "__main__":
    main()
