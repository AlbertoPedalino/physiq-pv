"""Does the side of an event where only the model flags point where the event goes next?

A value is flagged by a score when it is in its highest `--top-percent` (`--by-top-percent` for the
second score). At every timestamp the cells the second score flags make areas (touching cells,
diagonals included); an area of at least MIN_CELLS cells is an event at that timestamp. Every cell
within EDGE cells of an area belongs to the nearest one. For every event:
- its edge by the first score: the cells within EDGE cells of the area that only the first score
  flags; the direction of the edge is from the centre of the area to the centre of those cells;
- where it goes: the cells the second score flags STEPS_HOURS later within REACH cells of the area
  (the same event, moved or grown). Two directions: from the centre of the area to the centre of
  those cells (the shift of the event) and to the centre of the cells among them that were not
  flagged before (where it grows).
The cosine of the angle between the direction of the edge and where the event goes says whether
the edge points the right way: positive when it points the same side, ALIGNED or more within about
45 degrees. Directions are in cells of the grid, rows and columns alike.

The same is done with two references in place of the edge of the first score:
- ring: every cell within EDGE cells of the area, flagged or not. Its direction is not zero where
  the area is not symmetric or the border of the grid cuts the ring: what the shape alone gives;
- with `--early-percent`, the cells the second score flags at that lower level only: whether the
  weaker flags of the simple score point the way as well.
With no information the cosine is positive half of the time and its mean is zero.

    python scripts/check_era5_edge_direction.py --score-dir outputs/run_gen_clim --by-dir outputs/run_simple_max \
        --early-percent 10 --csv outputs/edge_direction_5.csv
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

STEPS_HOURS = (3, 6)
EDGE = 2  # cells around an area that make its edge
REACH = 4  # cells around an area where the later flags of the second score are the same event
MIN_CELLS = 4
ALIGNED = 0.7  # cosine of about 45 degrees
TARGETS = ("shift", "growth")
NAME, WIDTH = 44, 11


def centres(labels, rows, cols, size):
    """Number of cells of every label (0 to size - 1) and the mean row and column of its cells."""
    count = np.bincount(labels, minlength=size).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (count, np.bincount(labels, weights=rows, minlength=size) / count,
                np.bincount(labels, weights=cols, minlength=size) / count)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score (the model)")
    parser.add_argument("--by-dir", type=Path, required=True, help="Run with the second score")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--by-top-percent", type=float, help="Level of the second score; default --top-percent")
    parser.add_argument("--early-percent", type=float, help="Also the cells the second score flags at this lower level")
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
    if any(hours % step_hours for hours in STEPS_HOURS):
        raise ValueError(f"The hours must be multiples of {step_hours}")
    steps = [hours // step_hours for hours in STEPS_HOURS]
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    if len(rows) != first.shape[1]:
        raise ValueError("The grid of the run does not match the cells of the scores")
    valid = np.zeros(shape, dtype=bool)
    valid[rows, cols] = True
    grid_rows, grid_cols = np.indices(shape)

    def lattice(frame):
        result = np.zeros(shape, dtype=bool)
        result[rows, cols] = frame
        return result

    names = [path.name for path in (args.score_dir, args.by_dir)]
    common = len(os.path.commonprefix(names))
    short = [name[common:] or name for name in names]
    edges = [f"{short[0]} only", "ring"] + ([f"{short[1]} at {args.early_percent:g}% only"] if early is not None else [])
    # [edge, target, step]: events, cosine above 0, at least ALIGNED, sum of the cosines
    tally = np.zeros((len(edges), len(TARGETS), len(steps), 4))
    events = 0
    times = len(timestamps)
    for step in range(times - min(steps)):
        area = lattice(second[step])
        labels, count = ndimage.label(area, structure=np.ones((3, 3)))
        if not count:
            continue
        distance, (near_rows, near_cols) = ndimage.distance_transform_cdt(~area, metric="chessboard",
                                                                          return_indices=True)
        nearest = labels[near_rows, near_cols]
        size = count + 1
        cells, centre_row, centre_col = centres(labels[area], grid_rows[area], grid_cols[area], size)
        large = cells >= MIN_CELLS
        large[0] = False
        events += int(large.sum())
        band = valid & (distance >= 1) & (distance <= EDGE)
        sets = [lattice(first[step] & ~second[step]) & band, band]
        if early is not None:
            sets.append(lattice(early[step] & ~second[step]) & band)
        directions = []
        for chosen in sets:
            found, edge_row, edge_col = centres(nearest[chosen], grid_rows[chosen], grid_cols[chosen], size)
            directions.append((found > 0, edge_row - centre_row, edge_col - centre_col))
        for s, ahead in enumerate(steps):
            if step + ahead >= times:
                continue
            later = lattice(second[step + ahead]) & (distance <= REACH)
            for t, target in enumerate((later, later & ~area)):
                found, to_row, to_col = centres(nearest[target], grid_rows[target], grid_cols[target], size)
                go_row, go_col = to_row - centre_row, to_col - centre_col
                for e, (present, edge_row, edge_col) in enumerate(directions):
                    with np.errstate(invalid="ignore", divide="ignore"):
                        norm = np.hypot(edge_row, edge_col) * np.hypot(go_row, go_col)
                        cosine = (edge_row * go_row + edge_col * go_col) / norm
                    use = large & present & (found > 0) & (norm > 0)
                    value = cosine[use]
                    tally[e, t, s] += (len(value), (value > 0).sum(), (value >= ALIGNED).sum(), value.sum())

    print(f"[edge-direction] {timestamps[0].date()} to {timestamps[-1].date()} | flagged: the highest {args.top_percent:g}% "
          f"of {short[0]}, {by_top:g}% of {short[1]} | event: an area of at least {MIN_CELLS} touching cells of {short[1]} "
          f"at a timestamp ({events} in all) | edge: within {EDGE} cells | later: flags of {short[1]} within {REACH} cells")
    print("\ncosine between the direction of the edge and where the event goes: % of the events with a positive cosine, "
          f"with at least {ALIGNED} (within about 45 degrees), and the mean")
    print(f"{'edge, where the event goes, hours':<{NAME}}{'events':>{WIDTH}}{'same side':>{WIDTH}}{'within 45':>{WIDTH}}"
          f"{'mean cos':>{WIDTH}}")
    rows_out = []
    for s, hours in enumerate(STEPS_HOURS):
        for t, target in enumerate(TARGETS):
            for e, edge in enumerate(edges):
                total, positive, aligned, summed = tally[e, t, s]
                share = lambda value: 100 * value / total if total else np.nan
                mean = summed / total if total else np.nan
                print(f"{f'{edge}, {target}, {hours}h'[:NAME - 1]:<{NAME}}{int(total):>{WIDTH}d}{share(positive):>{WIDTH}.1f}"
                      f"{share(aligned):>{WIDTH}.1f}{mean:>{WIDTH}.3f}")
                rows_out.append({"edge": ("first only", "ring", "second early only")[e], "target": target,
                                 "hours": hours, "events": int(total), "same_side": share(positive),
                                 "within_45": share(aligned), "mean_cosine": mean, "top_percent": args.top_percent,
                                 "by_top_percent": by_top, "early_percent": args.early_percent})
            print()
    print("shift: towards the centre of the flags of the event later; growth: towards the centre of those not flagged "
          "before. With no information: same side 50%, within 45 degrees 25%, mean 0. Ring: what the shape of the "
          "area and the border of the grid alone give")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(args.csv, index=False)
        print(f"written {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
