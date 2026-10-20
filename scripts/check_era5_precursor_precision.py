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

With `--carry wind` (and `--wind-dir`) or `--carry motion` the second watcher is not the second score at its
lower level but where its flags go in one step: moved by the wind at 850 hPa, or as their event moved in the
step before, widened by one cell, less the cells it flags now (where its anomaly arrives).

With `--components-dir` (the run scored with Monte Carlo dropout) the check also reads, in every case, the
spread of the dropout draws of G and of D at the cells followed by an event (at the first radius and lead) and
at those not followed: their medians and how well each spread tells the two apart (the area under the ROC
curve), on one timestamp every `--components-stride` and a share `--neither-share` of the cells where neither
flags.

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


def carried(early, timestamps, grid, mode, wind_dir):
    """[timestamps, cells] of where the flags `early` go in one step, by the wind or by the motion of their event in
    the step before, less the cells flagged now."""
    from scripts.check_era5_edge_direction import wind_of_month
    from scripts.nowcast_era5_anomalies import METERS_PER_DEGREE, motion_shift, shifted

    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    latitudes, longitudes = np.asarray(grid["latitudes"], dtype=float), np.asarray(grid["longitudes"], dtype=float)
    shape = (len(latitudes), len(longitudes))
    seconds = (timestamps[1] - timestamps[0]) / pd.Timedelta(seconds=1)
    per_metre_north = 1 / (METERS_PER_DEGREE * abs(latitudes[1] - latitudes[0]))
    per_metre_east = 1 / (METERS_PER_DEGREE * np.cos(np.radians(latitudes)) * abs(longitudes[1] - longitudes[0]))

    def lattice(frame, dtype=bool):
        result = np.zeros(shape, dtype=dtype)
        result[rows, cols] = frame
        return result

    moved = np.zeros(early.shape, dtype=bool)
    for month in range(1, 13):
        positions = np.nonzero(timestamps.month == month)[0]
        if not len(positions):
            continue
        wind = wind_of_month(wind_dir, timestamps, positions, latitudes[rows], longitudes[cols]) if mode == "wind" else None
        for index, t in enumerate(positions):
            now = lattice(early[t])
            if not now.any() or (mode == "motion" and t == 0):
                continue
            if mode == "wind":
                u, v = lattice(wind[index, :, 0], float), lattice(wind[index, :, 1], float)
                row_shift, col_shift = -v * seconds * per_metre_north, u * seconds * per_metre_east[:, None]
            else:
                row_shift, col_shift = motion_shift(now, lattice(early[t - 1]), shape)
            moved[t] = shifted(now, row_shift, col_shift, shape)[rows, cols] & ~early[t]
    return moved


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
    parser.add_argument("--carry", choices=("none", "wind", "motion"), default="none",
                        help="The second watcher: where the flags of the second score go by the wind or by their motion")
    parser.add_argument("--wind-dir", type=Path, help="Monthly ERA5 files at 850 hPa with u and v, for --carry wind")
    parser.add_argument("--components-dir", type=Path, help="Run scored with dropout draws, for their spreads")
    parser.add_argument("--components-stride", type=int, default=3, help="Timestamps read for the spreads: one every this many")
    parser.add_argument("--spread-normals", type=Path,
                        help="Usual spreads (build_era5_spread_normals.py): read each spread over its usual value")
    parser.add_argument("--neither-share", type=float, default=0.02, help="Share of the cells where neither flags read")
    args = parser.parse_args(argv)
    # With weather as the outcome the event level of the second score only says where all is calm: its watcher may
    # be at the same level.
    if args.early_percent <= args.by_top_percent and args.target == "simple":
        raise ValueError("--early-percent must be a lower level (a larger share) than --by-top-percent")
    if args.target != "simple" and args.unseen_dir is None:
        raise ValueError(f"--target {args.target} needs --unseen-dir")
    if args.carry == "wind" and args.wind_dir is None:
        raise ValueError("--carry wind needs --wind-dir")
    first, timestamps, _ = flags(args.score_dir, args.top_percent)
    event, other_times, _ = flags(args.by_dir, args.by_top_percent)
    early = flags(args.by_dir, args.early_percent)[0]
    if not timestamps.equals(other_times) or first.shape != event.shape:
        raise ValueError("The two scores are not over the same timestamps and cells")
    if args.carry != "none":
        early = carried(early, timestamps, json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))[
            "grid"], args.carry, args.wind_dir)
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
    components, sampled = None, [([], []) for _ in CASES]  # spreads and whether followed, by case
    if args.components_dir is not None:
        from scripts.check_era5_components_by_group import load_spread_normals, open_components

        if not pd.DatetimeIndex(np.load(args.components_dir / "test_timestamps.npy")).equals(timestamps):
            raise ValueError("The run with the dropout draws is not over the timestamps of the scores")
        components = open_components(args.components_dir)
        normals = load_spread_normals(args.spread_normals)
        rng = np.random.default_rng(0)
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
        if components is not None and t % args.components_stride == 0:
            from scripts.check_era5_components_by_group import spread_normal, spreads_at

            values = spreads_at(components, t, spread_normal(normals, timestamps[t]))[calm]
            followed = near_outcome[args.radius[0]][t + 1:t + leads[0] + 1].any(axis=0)[calm]
            for c in range(len(CASES)):
                chosen = case == c
                if c == len(CASES) - 1:
                    chosen &= rng.random(len(case)) < args.neither_share
                sampled[c][0].append(values[chosen])
                sampled[c][1].append(followed[chosen])
    names = [path.name for path in (args.score_dir, args.by_dir)]
    common = len(os.path.commonprefix(names))
    short = [name[common:] or name for name in names]
    print(f"[precursor] {timestamps[0].date()} to {timestamps[-1].date()} | "
          + ("cells: all where the event is not there now" if args.all_cells else
             f"calm: no flag of {short[1]} at its highest {args.by_top_percent:g}% within {args.quiet_radius} cells in "
             f"the {args.quiet_hours} h before") + " | watchers: "
          f"{short[0]} at {args.top_percent:g}%, {short[1]} at {args.early_percent:g}%"
          + ("" if args.carry == "none" else f" carried one step by the {args.carry}") + " | event: "
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
    if components is not None:
        from scripts.check_era5_components_by_group import compare_spreads, print_spreads

        spreads = {}
        for c, name in enumerate(CASES):
            if sampled[c][0]:
                spreads[name] = compare_spreads(np.concatenate(sampled[c][1]), np.concatenate(sampled[c][0]))
        print_spreads(f"dropout spreads at the cells followed by an event ({args.radius[0]} cell(s), "
                      f"{args.lead_hours[0]}h) and not, by case", list(spreads.items()), hit_name="followed")
        result["spreads"] = spreads
    return result


if __name__ == "__main__":
    main()
