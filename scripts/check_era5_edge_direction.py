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
45 degrees. Directions are in cells of the grid.

The same is done with references in place of the edge of the first score:
- ring: every cell within EDGE cells of the area, flagged or not. Its direction is not zero where
  the area is not symmetric or the border of the grid cuts the ring: what the shape alone gives;
- with `--early-percent`, the cells the second score flags at that lower level only: whether the
  weaker flags of the simple score point the way as well;
- past motion: the event goes on as it came, from the centre of the flags of the second score within
  REACH cells of the area as many hours before to the centre of the area now;
- with `--wind-dir` (the monthly ERA5 files at 850 hPa, `u` and `v`), the wind: the mean wind over
  the cells of the area, turned into cells of the grid (a degree of longitude is shorter by the
  cosine of the latitude), where the flow carries the event.
With no information the cosine is positive half of the time and its mean is zero.

With `--target cape`, `gust` or `cape_or_gust` (and `--unseen-dir`), where the event goes is judged by weather
no score reads: the cells within REACH cells of the area where that weather is in its highest `--unusual-percent`
for the cell, month and time of day STEPS_HOURS later and was not at the timestamp of the event (shift), and
those of them outside the area (growth). Past motion still comes from the second score.

With `--components-dir` (the run scored with Monte Carlo dropout) the check also reads the spread of the dropout
draws of G and of D over the edge of the first score of every event: its medians where the edge points the right
way and where it does not (shift, 3 h) and how well each spread tells the two apart (the area under the ROC curve).

Every edge and reference is first read on the events where it has a direction; a second table reads
them on the same events, those where the edge of the first score, past motion and wind have a
direction (ring and weaker flags where they also have one). A last table puts those
events in four cases by whether the edge of the first score and each of past motion and wind are on
the right side: whether the edge says something the flow and the past motion do not; the weaker flags
of the second score, when given, are a reference there too, on the events where they also have a direction.

    python scripts/check_era5_edge_direction.py --score-dir outputs/run_gen_clim --by-dir outputs/run_simple_max \
        --early-percent 10 --wind-dir data/era5/pressure_850 --csv outputs/edge_direction_5.csv
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
from scripts.check_era5_score_against_unseen import read_month

STEPS_HOURS = (3, 6)
EDGE = 2  # cells around an area that make its edge
REACH = 4  # cells around an area where the flags of the second score before and after are the same event
MIN_CELLS = 4
ALIGNED = 0.7  # cosine of about 45 degrees
TARGETS = ("shift", "growth")
CASES = ("both right", "edge only", "reference only", "both wrong")
NAME, WIDTH = 44, 11


def centres(labels, rows, cols, size, weights=None):
    """Number of cells of every label (0 to size - 1) and the mean row and column of its cells (or the mean of
    the columns of `weights`, when given)."""
    count = np.bincount(labels, minlength=size).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        if weights is not None:
            return count, *(np.bincount(labels, weights=column, minlength=size) / count for column in weights)
        return (count, np.bincount(labels, weights=rows, minlength=size) / count,
                np.bincount(labels, weights=cols, minlength=size) / count)


def wind_of_month(wind_dir, timestamps, positions, latitudes, longitudes):
    """Wind u and v [timestamp of `positions`, cell, 2] at 850 hPa, read from the monthly files."""
    month = timestamps.month[positions[0]]
    parts = [read_month(wind_dir, year, month, latitudes, longitudes) for year in sorted(set(timestamps.year[positions]))]
    names = parts[0][2]
    if any(part[2] != names for part in parts) or not {"u", "v"} <= set(names):
        raise ValueError(f"The wind files of month {month} do not all hold u and v")
    times = parts[0][0].append([part[0] for part in parts[1:]])
    found = timestamps.get_indexer(times)
    kept = found >= 0  # a file may hold timestamps the score does not have
    values = np.concatenate([part[1][:, :, [names.index("u"), names.index("v")]] for part in parts])[kept]
    order = np.argsort(found[kept])
    if not np.array_equal(found[kept][order], positions):
        raise ValueError(f"The wind of month {month} does not cover the timestamps of the scores")
    return values[order]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score (the model)")
    parser.add_argument("--by-dir", type=Path, required=True, help="Run with the second score")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--by-top-percent", type=float, help="Level of the second score; default --top-percent")
    parser.add_argument("--early-percent", type=float, help="Also the cells the second score flags at this lower level")
    parser.add_argument("--wind-dir", type=Path, help="Monthly ERA5 files at 850 hPa with u and v, a folder per year")
    parser.add_argument("--csv", type=Path, help="Also write the shares here")
    parser.add_argument("--target", choices=("simple", "cape", "gust", "cape_or_gust"), default="simple",
                        help="Where the event goes: the flags of the second score later, or where the CAPE, the gust, "
                             "either of them becomes unusual")
    parser.add_argument("--unseen-dir", type=Path, help="Monthly files of the CAPE and the gust, for those targets")
    parser.add_argument("--unusual-percent", type=float, default=5, help="Unusual: this highest share of the CAPE or gust")
    parser.add_argument("--components-dir", type=Path, help="Run scored with dropout draws, for their spreads on the edge")
    parser.add_argument("--spread-normals", type=Path,
                        help="Usual spreads (build_era5_spread_normals.py): read each spread over its usual value")
    args = parser.parse_args(argv)
    if args.target != "simple" and args.unseen_dir is None:
        raise ValueError(f"--target {args.target} needs --unseen-dir")
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
    grid_latitudes, grid_longitudes = np.asarray(grid["latitudes"], dtype=float), np.asarray(grid["longitudes"], dtype=float)
    latitude_step, longitude_step = grid_latitudes[1] - grid_latitudes[0], grid_longitudes[1] - grid_longitudes[0]
    valid = np.zeros(shape, dtype=bool)
    valid[rows, cols] = True
    grid_rows, grid_cols = np.indices(shape)
    # A wind of 1 m/s northwards moves 1 / latitude_step rows per degree; eastwards, 1 / (cos(latitude) longitude_step).
    row_per_north = np.zeros(shape)
    column_per_east = np.zeros(shape)
    row_per_north[:] = 1 / latitude_step
    column_per_east[:] = (1 / (np.cos(np.radians(grid_latitudes)) * longitude_step))[:, None]
    unusual = None
    if args.target != "simple":  # weather no score reads, unusual for the cell, month and time of day
        from scripts.nowcast_era5_anomalies import TARGETS as WEATHER, unseen_flags

        unusual = unseen_flags(args.unseen_dir, timestamps, grid_latitudes[rows], grid_longitudes[cols],
                               args.unusual_percent)[:, :, list(WEATHER[args.target])].any(axis=2)

    def lattice(frame):
        result = np.zeros(shape, dtype=bool)
        result[rows, cols] = frame
        return result

    names = [path.name for path in (args.score_dir, args.by_dir)]
    common = len(os.path.commonprefix(names))
    short = [name[common:] or name for name in names]
    edges = [f"{short[0]} only", "ring"]
    if early is not None:
        edges.append(f"{short[1]} at {args.early_percent:g}% only")
    edges.append("past motion")
    if args.wind_dir is not None:
        edges.append("wind at 850 hPa")
    past, wind_index = edges.index("past motion"), edges.index("wind at 850 hPa") if args.wind_dir is not None else None
    flows = [past] + ([wind_index] if wind_index is not None else [])
    references = flows + ([edges.index(f"{short[1]} at {args.early_percent:g}% only")] if early is not None else [])
    # [edge, target, step]: events, cosine above 0, at least ALIGNED, sum of the cosines; on its own events and on shared ones
    tally, shared = (np.zeros((len(edges), len(TARGETS), len(steps), 4)) for _ in range(2))
    cases = np.zeros((len(references), len(TARGETS), len(steps), len(CASES)))
    # [reference, target, step]: events, the edge and the reference on the same side, right when they agree,
    # reference right and edge right when they do not, the combined direction right, the reference right
    agreement = np.zeros((len(references), len(TARGETS), len(steps), 7))
    events = 0
    times = len(timestamps)
    components, edge_spreads, edge_right = None, [], []
    if args.components_dir is not None:
        from scripts.check_era5_components_by_group import (load_spread_normals, open_components, spread_normal,
                                                            spreads_at)

        if not pd.DatetimeIndex(np.load(args.components_dir / "test_timestamps.npy")).equals(timestamps):
            raise ValueError("The run with the dropout draws is not over the timestamps of the scores")
        components = open_components(args.components_dir)
        normals = load_spread_normals(args.spread_normals)
    for month in range(1, 13):
        positions = np.nonzero(timestamps.month == month)[0]
        if not len(positions):
            continue
        wind = None
        if args.wind_dir is not None:
            wind = wind_of_month(args.wind_dir, timestamps, positions, grid_latitudes[rows], grid_longitudes[cols])
        for index, step in enumerate(positions):
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
            fixed = []
            for chosen in sets:
                found, edge_row, edge_col = centres(nearest[chosen], grid_rows[chosen], grid_cols[chosen], size)
                fixed.append((found > 0, edge_row - centre_row, edge_col - centre_col))
            if components is not None:  # the mean spreads over the edge of the first score of every event
                values = spreads_at(components, step, spread_normal(normals, timestamps[step]))
                chosen = sets[0]
                spread_lattices = []
                for k in range(3):
                    frame = np.zeros(shape)
                    frame[rows, cols] = values[:, k]
                    spread_lattices.append(frame[chosen])
                _, *event_spreads = centres(nearest[chosen], None, None, size, weights=spread_lattices)
            if wind is not None:
                u, v = np.zeros(shape), np.zeros(shape)
                u[rows, cols], v[rows, cols] = wind[index, :, 0], wind[index, :, 1]
                towards = (u * column_per_east, v * row_per_north)
                _, go_row, go_col = centres(labels[area], None, None, size, weights=(towards[1][area], towards[0][area]))
                with np.errstate(invalid="ignore"):
                    moving = np.isfinite(go_row) & np.isfinite(go_col)
                wind_direction = (moving, np.nan_to_num(go_row), np.nan_to_num(go_col))
            for s, ahead in enumerate(steps):
                if step + ahead >= times or step - ahead < 0:
                    continue
                before = lattice(second[step - ahead]) & (distance <= REACH)
                found, from_row, from_col = centres(nearest[before], grid_rows[before], grid_cols[before], size)
                directions = list(fixed) + [(found > 0, centre_row - from_row, centre_col - from_col)]
                if wind is not None:
                    directions.append(wind_direction)
                if unusual is None:
                    later = lattice(second[step + ahead]) & (distance <= REACH)
                else:  # where the weather becomes unusual
                    later = lattice(unusual[step + ahead] & ~unusual[step]) & (distance <= REACH)
                for t, target in enumerate((later, later & ~area)):
                    found, to_row, to_col = centres(nearest[target], grid_rows[target], grid_cols[target], size)
                    go_row, go_col = to_row - centre_row, to_col - centre_col
                    cosines, usable = [], []
                    for present, edge_row, edge_col in directions:
                        with np.errstate(invalid="ignore", divide="ignore"):
                            norm = np.hypot(edge_row, edge_col) * np.hypot(go_row, go_col)
                            cosine = (edge_row * go_row + edge_col * go_col) / norm
                        cosines.append(cosine)
                        usable.append(large & present & (found > 0) & (norm > 0))
                    every = np.logical_and.reduce([usable[0]] + [usable[r] for r in flows])
                    if components is not None and s == 0 and t == 0:
                        edge_spreads.append(np.stack(event_spreads, axis=1)[usable[0]])
                        edge_right.append(cosines[0][usable[0]] > 0)
                    for e, (cosine, use) in enumerate(zip(cosines, usable)):
                        for table, chosen in ((tally, use), (shared, every & use)):
                            value = cosine[chosen]
                            table[e, t, s] += (len(value), (value > 0).sum(), (value >= ALIGNED).sum(), value.sum())
                    edge_row, edge_col = directions[0][1], directions[0][2]
                    for r, reference in enumerate(references):
                        pair = every & usable[reference]
                        model, other = cosines[0][pair] > 0, cosines[reference][pair] > 0
                        cases[r, t, s] += ((model & other).sum(), (model & ~other).sum(), (~model & other).sum(),
                                           (~model & ~other).sum())
                        # Do the edge and the reference point the same side, and which is right when they do not?
                        # The combined direction is the mean of the two directions made of length one.
                        ref_row, ref_col = directions[reference][1], directions[reference][2]
                        with np.errstate(invalid="ignore", divide="ignore"):
                            agree = ((edge_row * ref_row + edge_col * ref_col) > 0)[pair]
                            edge_norm, ref_norm = np.hypot(edge_row, edge_col), np.hypot(ref_row, ref_col)
                            both_row = edge_row / edge_norm + ref_row / ref_norm
                            both_col = edge_col / edge_norm + ref_col / ref_norm
                            combined = ((both_row * go_row + both_col * go_col) > 0)[pair]
                        agreement[r, t, s] += (agree.size, agree.sum(), (agree & other).sum(), (~agree & other).sum(),
                                               (~agree & model).sum(), combined.sum(), other.sum())

    print(f"[edge-direction] {timestamps[0].date()} to {timestamps[-1].date()} | flagged: the highest {args.top_percent:g}% "
          f"of {short[0]}, {by_top:g}% of {short[1]} | event: an area of at least {MIN_CELLS} touching cells of {short[1]} "
          f"at a timestamp ({events} in all) | edge: within {EDGE} cells | before and later: flags of {short[1]} within "
          f"{REACH} cells" + ("" if unusual is None else f" | where it goes: {args.target.replace('_', ' ')} becoming "
                              f"unusual (highest {args.unusual_percent:g}%) within {REACH} cells"))
    rows_out = []
    for title, table, kind in (("on the events where each has a direction", tally, "own events"),
                               ("on the same events, where the edge, past motion and wind have a direction", shared, "same events")):
        print(f"\ncosine between the direction and where the event goes, {title}: % of the events with a positive "
              f"cosine, with at least {ALIGNED} (within about 45 degrees), and the mean")
        print(f"{'direction, where the event goes, hours':<{NAME}}{'events':>{WIDTH}}{'same side':>{WIDTH}}"
              f"{'within 45':>{WIDTH}}{'mean cos':>{WIDTH}}")
        for s, hours in enumerate(STEPS_HOURS):
            for t, target in enumerate(TARGETS):
                for e, edge in enumerate(edges):
                    total, positive, aligned, summed = table[e, t, s]
                    share = lambda value: 100 * value / total if total else np.nan
                    mean = summed / total if total else np.nan
                    print(f"{f'{edge}, {target}, {hours}h'[:NAME - 1]:<{NAME}}{int(total):>{WIDTH}d}"
                          f"{share(positive):>{WIDTH}.1f}{share(aligned):>{WIDTH}.1f}{mean:>{WIDTH}.3f}")
                    rows_out.append({"table": kind, "direction": edge if e else "first only", "target": target,
                                     "hours": hours, "events": int(total), "same_side": share(positive),
                                     "within_45": share(aligned), "mean_cosine": mean, "top_percent": args.top_percent,
                                     "by_top_percent": by_top, "early_percent": args.early_percent})
                print()
    print(f"on the same events: is the edge of {short[0]} on the right side, and the reference? (% of the events)")
    print(f"{'reference, where the event goes, hours':<{NAME}}" + "".join(f"{case:>{WIDTH + 4}}" for case in CASES)
          + f"{'edge right when ref wrong':>{WIDTH + 16}}")
    for s, hours in enumerate(STEPS_HOURS):
        for t, target in enumerate(TARGETS):
            for r, reference in enumerate(references):
                counts = cases[r, t, s]
                total = counts.sum()
                share = 100 * counts / total if total else np.full(len(CASES), np.nan)
                rescue = 100 * counts[1] / (counts[1] + counts[3]) if counts[1] + counts[3] else np.nan
                print(f"{f'{edges[reference]}, {target}, {hours}h'[:NAME - 1]:<{NAME}}"
                      + "".join(f"{value:>{WIDTH + 4}.1f}" for value in share) + f"{rescue:>{WIDTH + 16}.1f}")
                for case, value in zip(CASES, share):
                    rows_out.append({"table": "cases", "direction": edges[reference], "target": target, "hours": hours,
                                     "case": case, "share": value, "top_percent": args.top_percent,
                                     "by_top_percent": by_top, "early_percent": args.early_percent})
    print(f"\non the same events: when the edge of {short[0]} and the reference point the same side (an angle under 90 "
          f"degrees between them), how often are they right, and when they do not, which one is? (% of the events)")
    columns = ("agree", "right if agree", "ref right if not", "edge right if not", "combined right", "ref right")
    print(f"{'reference, where the event goes, hours':<{NAME}}" + "".join(f"{name:>{WIDTH + 7}}" for name in columns))
    for s, hours in enumerate(STEPS_HOURS):
        for t, target in enumerate(TARGETS):
            for r, reference in enumerate(references):
                total, agree, right_agree, ref_not, edge_not, combined, ref_right = agreement[r, t, s]
                share = lambda part, whole: 100 * part / whole if whole else np.nan
                values = (share(agree, total), share(right_agree, agree), share(ref_not, total - agree),
                          share(edge_not, total - agree), share(combined, total), share(ref_right, total))
                print(f"{f'{edges[reference]}, {target}, {hours}h'[:NAME - 1]:<{NAME}}"
                      + "".join(f"{value:>{WIDTH + 7}.1f}" for value in values))
                rows_out.append({"table": "agreement", "direction": edges[reference], "target": target, "hours": hours,
                                 **dict(zip(columns, values)), "top_percent": args.top_percent, "by_top_percent": by_top,
                                 "early_percent": args.early_percent})
    print("shift: towards the centre of the flags of the event later; growth: towards the centre of those not flagged "
          "before. With no information: same side 50%, within 45 degrees 25%, mean 0. Ring: what the shape of the "
          "area and the border of the grid alone give. Edge right when ref wrong: above 50%, the edge "
          "points the right way where the reference does not. Combined: the mean of the two directions of length one")
    if components is not None and edge_spreads:
        from scripts.check_era5_components_by_group import compare_spreads, print_spreads

        print_spreads(f"dropout spreads over the edge of {short[0]}, by whether it points the right way (shift, "
                      f"{STEPS_HOURS[0]}h)", [("edge", compare_spreads(np.concatenate(edge_right),
                                                                        np.concatenate(edge_spreads)))],
                      hit_name="right")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(args.csv, index=False)
        print(f"written {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
