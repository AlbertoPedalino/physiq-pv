"""Where only one score flags a cell next to what the other flags, does the other reach that cell later?

A value is flagged by a score when it is in its highest `--top-percent`. At every timestamp the cells
the second score flags make its area; every other cell has a distance from it, in cells of the grid,
counted as the larger of the row and column offsets (BANDS). Two groups of cells outside that area:
- first only: the first score flags the cell, the second does not;
- neither: no score flags it, the reference.
For every group, distance band and window (WINDOWS_HOURS) a value is put in one of four kinds by what
the second score does at the same cell within the window:
- front: it flags the cell after, not before: the first score saw the weather coming;
- trail: it flagged the cell before, not after: the first score holds on to weather that is leaving;
- both: before and after, the cell is inside a longer episode;
- none: neither before nor after.
At equal distance from the area of the second score, a share of fronts larger for "first only" than
for "neither" is a lead of the first score, in time (the hours to the arrival) and in space (the
distance of the cell). A trail share larger by as much says it extends the weather both ways.

With `--unseen-dir`, for the same groups and kinds (window UNSEEN_WINDOW_HOURS), the share of the values
where a variable no score saw is in its highest `--unusual-percent` for the cell, the calendar month
and the time of day: whether the fronts are weather already.

With several `--by-dir` the second score is any of them: a value is flagged when at least one of
them flags it, each at its own highest `--by-top-percent`, and "first only" is what none of them
flags.

The whole scored period is read. With `--csv` the shares of the kinds go to a file.

With `--events` (a CSV of known events: `id`, `tipo`, `inizio`, `fine`, `lat_min`, `lat_max`, `lon_min`, `lon_max`),
the same kinds are also counted only on the values of the events: the cells of the box of every event and of a ring
of `--ring-km` around it, from `--before-hours` before its onset (the measured onset of `--onsets`, the inizi.csv of
check_era5_known_events_lead.py, main variant; otherwise the first day of the source) to 21 UTC of its last day.
The cells of one event are not independent, so the uncertainty comes from `--bootstrap` resamples of the events:
the 95% interval of "front x" and of the excess of the first score only, (front - trail) of the first only minus
(front - trail) of neither at the same distance, which is positive when the first score sees the cells the second
reaches next more often than chance and than the cells it leaves. Written to `--events-csv`.

    python scripts/check_era5_edge_lead.py --score-dir outputs/run_gen_clim --by-dir outputs/run_simple_max \
        --unseen-dir data/era5_independent/independent --csv outputs/edge_lead_5.csv
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import numpy as np
import pandas as pd
from scipy import ndimage

from scripts.check_era5_lead import flags
from scripts.check_era5_score_against_unseen import read_month, unusual_values

CHUNK = 256  # timestamps read at a time
BANDS = ((1, 1), (2, 2), (3, 5), (6, 10), (11, None))  # cells of 0.5 degrees from the area of the second score
WINDOWS_HOURS = (3, 6, 12, 24)
UNSEEN_WINDOW_HOURS = 6
GROUPS = ("first only", "neither")
KINDS = ("none", "front", "trail", "both")
NAME, WIDTH = 34, 10


def since_and_until(flagged, cap):
    """Steps since the last flag strictly before every timestamp, and to the next one strictly after, per cell,
    capped at `cap`: two [time, cell] arrays of int8."""
    times, cells = flagged.shape
    since, until = np.empty((times, cells), dtype=np.int8), np.empty((times, cells), dtype=np.int8)
    count = np.full(cells, cap, dtype=np.int8)
    for step in range(times):
        since[step] = count
        count = np.where(flagged[step], 1, np.minimum(count + 1, cap)).astype(np.int8)
    count[:] = cap
    for step in range(times - 1, -1, -1):
        until[step] = count
        count = np.where(flagged[step], 1, np.minimum(count + 1, cap)).astype(np.int8)
    return since, until


def band_of(area, rows, cols, shape):
    """Distance band (index in BANDS) of every cell from the flagged cells of `area` [time, cell]."""
    result = np.empty(area.shape, dtype=np.int8)
    edges = np.array([low for low, _ in BANDS[1:]])
    for index, frame in enumerate(area):
        if not frame.any():
            result[index] = len(BANDS) - 1  # nothing flagged: every cell is far
            continue
        lattice = np.ones(shape, dtype=bool)
        lattice[rows[frame], cols[frame]] = False
        distance = ndimage.distance_transform_cdt(lattice, metric="chessboard")[rows, cols]
        result[index] = np.searchsorted(edges, distance, side="right")
    return result


def event_windows(events, onsets, grid, ring_km, before_hours):
    """(id, type, cells of the box and of the ring, first and last timestamp) of every event."""
    # imported here: these modules import, through others, since_and_until of this one
    from scripts.check_era5_known_events import in_box
    from scripts.check_era5_known_events_lead import Layout, ring_cells
    layout = Layout(grid)
    out = []
    for _, event in events.iterrows():
        onset = onsets.get(event["id"], event["inizio"].normalize())
        cells = in_box(event, layout.latitudes, layout.longitudes) | ring_cells(event, layout, ring_km)
        out.append((event["id"], event.get("tipo", ""), np.asarray(cells),
                    onset - pd.Timedelta(hours=before_hours), event["fine"].normalize() + pd.Timedelta(hours=21)))
    return out


def measures(counts):
    """From counts [..., group, band, window, kind]: front and trail % of both groups, front x, and the excess
    (front - trail of the first only) - (front - trail of neither), all [..., band, window]."""
    with np.errstate(invalid="ignore", divide="ignore"):
        share = 100 * counts / counts.sum(axis=-1, keepdims=True)
        front, trail = share[..., 1], share[..., 2]
        return {"first front": front[..., 0, :, :], "first trail": trail[..., 0, :, :],
                "neither front": front[..., 1, :, :], "neither trail": trail[..., 1, :, :],
                "front x": front[..., 0, :, :] / front[..., 1, :, :],
                "excess": (front[..., 0, :, :] - trail[..., 0, :, :]) - (front[..., 1, :, :] - trail[..., 1, :, :])}


def event_report(event_kinds, windows_of_events, band_names, draws, seed, short, path):
    """Print and write the kinds on the values of the known events, with bootstrap intervals over the events."""
    total = measures(event_kinds.sum(axis=0))
    rng = np.random.default_rng(seed)
    count = len(event_kinds)
    boot = measures(event_kinds[rng.integers(0, count, size=(draws, count))].sum(axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low = {key: np.nanpercentile(value, 2.5, axis=0) for key, value in boot.items()}
        high = {key: np.nanpercentile(value, 97.5, axis=0) for key, value in boot.items()}
    values = event_kinds[:, 0, :, 0, :].sum(axis=-1)  # [event, band]: values of the first only
    print(f"\n[edge-lead, known events] {count} events, box and ring; {short[0]} only vs neither at the same "
          f"distance; 95% intervals from {draws} resamples of the events")
    print(f"{'distance, window':<22}{'values':>9}{'events':>8}{'front':>8}{'trail':>8}{'n front':>9}{'n trail':>9}"
          f"{'front x':>9}{'  (95%)':<16}{'excess pp':>10}{'  (95%)':<16}")
    rows = []
    for b, band in enumerate(band_names):
        for w, hours in enumerate(WINDOWS_HOURS):
            pick = lambda table, key: table[key][b, w]
            interval = lambda key: f"  ({low[key][b, w]:.2f}, {high[key][b, w]:.2f})"
            print(f"{f'{band}, {hours} h':<22}{int(values[:, b].sum()):>9d}{int((values[:, b] > 0).sum()):>8d}"
                  f"{pick(total, 'first front'):>8.1f}{pick(total, 'first trail'):>8.1f}"
                  f"{pick(total, 'neither front'):>9.1f}{pick(total, 'neither trail'):>9.1f}"
                  f"{pick(total, 'front x'):>9.2f}{interval('front x'):<16}{pick(total, 'excess'):>10.1f}"
                  f"{interval('excess'):<16}")
            rows.append({"type": "all", "distance": band, "window_hours": hours, "values": int(values[:, b].sum()),
                         "events with values": int((values[:, b] > 0).sum()),
                         **{key: float(pick(total, key)) for key in total},
                         **{f"{key} 2.5%": float(low[key][b, w]) for key in ("front x", "excess")},
                         **{f"{key} 97.5%": float(high[key][b, w]) for key in ("front x", "excess")}})
    print("excess: (front - trail) of the first only minus (front - trail) of neither, percentage points; "
          "front x: front of the first only over front of neither")
    kinds = sorted({kind for _, kind, *_ in windows_of_events})
    print("\nby type, distance 1, within 3 h")
    print(f"{'type':<10}{'events':>8}{'values':>9}{'front':>8}{'trail':>8}{'n front':>9}{'n trail':>9}{'front x':>9}"
          f"{'excess pp':>10}")
    for kind in kinds:
        chosen = [index for index, (_, other, *_) in enumerate(windows_of_events) if other == kind]
        part = measures(event_kinds[chosen].sum(axis=0))
        print(f"{kind:<10}{len(chosen):>8d}{int(values[chosen, 0].sum()):>9d}{part['first front'][0, 0]:>8.1f}"
              f"{part['first trail'][0, 0]:>8.1f}{part['neither front'][0, 0]:>9.1f}{part['neither trail'][0, 0]:>9.1f}"
              f"{part['front x'][0, 0]:>9.2f}{part['excess'][0, 0]:>10.1f}")
        rows.append({"type": kind, "distance": band_names[0], "window_hours": WINDOWS_HOURS[0],
                     "values": int(values[chosen, 0].sum()), "events with values": len(chosen),
                     **{key: float(value[0, 0]) for key, value in part.items()}})
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"written {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score (the model)")
    parser.add_argument("--by-dir", type=Path, nargs="+", required=True,
                        help="Run with the second score; with several, a value is flagged when any of them flags it")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--by-top-percent", type=float, help="Level of the second score; default --top-percent")
    parser.add_argument("--unseen-dir", type=Path, help="Monthly NetCDF files of the unseen variables")
    parser.add_argument("--unusual-percent", type=float, default=5, help="Unusual: this highest share of an unseen variable")
    parser.add_argument("--csv", type=Path, help="Also write the shares of the kinds here")
    parser.add_argument("--events", type=Path, help="CSV of known events: also count on their boxes and days only")
    parser.add_argument("--onsets", type=Path, help="inizi.csv of check_era5_known_events_lead.py (measured onsets)")
    parser.add_argument("--ring-km", type=float, default=250)
    parser.add_argument("--before-hours", type=float, default=72)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--events-csv", type=Path, help="Write the kinds on the known events here")
    args = parser.parse_args(argv)
    first, timestamps, _ = flags(args.score_dir, args.top_percent)
    by_top = args.top_percent if args.by_top_percent is None else args.by_top_percent
    second = None
    for path in args.by_dir:
        flagged, other_timestamps, _ = flags(path, by_top)
        if not timestamps.equals(other_timestamps):
            common = timestamps.intersection(other_timestamps)  # runs scored over slightly different spans
            if len(common) < 2:
                raise ValueError(f"The scores have no timestamps in common: {path}")
            print(f"[edge-lead] {path.name}: {len(common)} timestamps in common of {len(timestamps)} and "
                  f"{len(other_timestamps)}", flush=True)
            first = first[timestamps.get_indexer(common)]
            second = None if second is None else second[timestamps.get_indexer(common)]
            flagged = flagged[other_timestamps.get_indexer(common)]
            timestamps = common
        if first.shape != flagged.shape:
            raise ValueError(f"The scores are not over the same cells: {path}")
        second = flagged if second is None else second | flagged
        del flagged
    step_hours = int(round((timestamps[1] - timestamps[0]) / pd.Timedelta(hours=1)))
    if any(hours % step_hours for hours in (*WINDOWS_HOURS, UNSEEN_WINDOW_HOURS)):
        raise ValueError(f"The windows must be multiples of {step_hours} hours")
    windows = [hours // step_hours for hours in WINDOWS_HOURS]
    unseen_window = UNSEEN_WINDOW_HOURS // step_hours
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    if len(rows) != first.shape[1]:
        raise ValueError("The grid of the run does not match the cells of the scores")
    latitudes = np.asarray(grid["latitudes"], dtype=float)[rows]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[cols]
    longest = max(windows)
    since, until = since_and_until(second, longest + 1)
    sizes = (len(GROUPS), len(BANDS))
    kinds = np.zeros((*sizes, len(windows), len(KINDS)))
    arrival = np.zeros((*sizes, longest + 2))  # steps to the next flag of the second, for the fronts of the longest window
    hits = defined_counts = names = None
    windows_of_events = event_kinds = None
    if args.events is not None:
        onsets = {}
        if args.onsets is not None:
            table = pd.read_csv(args.onsets, parse_dates=["inizio"])
            table = table[table["variante"] == "principale"] if "variante" in table else table
            onsets = dict(zip(table["id"], table["inizio"]))
        from scripts.check_era5_known_events import read_events
        windows_of_events = event_windows(read_events(args.events), onsets, grid, args.ring_km, args.before_hours)
        event_kinds = np.zeros((len(windows_of_events), *sizes, len(windows), len(KINDS)))
    read = 0
    for month in range(1, 13):
        positions = np.nonzero(timestamps.month == month)[0]
        if not len(positions):
            continue
        unusual = None
        if args.unseen_dir is not None:
            years = sorted(set(timestamps.year[positions]))
            parts = [read_month(args.unseen_dir, year, month, latitudes, longitudes) for year in years]
            names = parts[0][2] if names is None else names
            if any(part[2] != names for part in parts):
                raise ValueError(f"The files of month {month} do not hold the variables of the other months")
            times = parts[0][0].append([part[0] for part in parts[1:]])
            found = timestamps.get_indexer(times)
            kept = found >= 0  # a file may hold timestamps the score does not have
            values = np.concatenate([part[1] for part in parts])[kept]
            del parts
            order = np.argsort(found[kept])
            if not np.array_equal(found[kept][order], positions):
                raise ValueError(f"The unseen variables of month {month} do not cover the timestamps of the scores")
            unusual, defined = unusual_values(values[order], timestamps.hour[positions].to_numpy(), args.unusual_percent)
            del values
            if hits is None:
                hits, defined_counts = (np.zeros((*sizes, len(KINDS), len(names))) for _ in range(2))
        for start in range(0, len(positions), CHUNK):
            chosen = positions[start:start + CHUNK]
            model, other = first[chosen], second[chosen]
            group = np.full(model.shape, -1, dtype=np.int64)
            group[model & ~other] = 0
            group[~model & ~other] = 1
            band = band_of(other, rows, cols, shape).astype(np.int64)
            valid = group >= 0
            cell_base = group * len(BANDS) + band
            base = cell_base[valid]
            chunk_since, chunk_until = since[chosen], until[chosen]
            before, after = chunk_since[valid], chunk_until[valid]
            if windows_of_events is not None:
                chunk_times = timestamps[chosen]
                for e, (_, _, cells, first_time, last_time) in enumerate(windows_of_events):
                    at = np.asarray((chunk_times >= first_time) & (chunk_times <= last_time))
                    if not at.any():
                        continue
                    mask = valid & at[:, None] & cells[None, :]
                    inside, since_e, until_e = cell_base[mask], chunk_since[mask], chunk_until[mask]
                    for w, window in enumerate(windows):
                        kind = (until_e <= window).astype(np.int64) + 2 * (since_e <= window)
                        event_kinds[e, ..., w, :] += np.bincount(
                            inside * len(KINDS) + kind, minlength=len(GROUPS) * len(BANDS) * len(KINDS)
                        ).reshape(*sizes, len(KINDS))
            for w, window in enumerate(windows):
                kind = (after <= window).astype(np.int64) + 2 * (before <= window)
                kinds[..., w, :] += np.bincount(base * len(KINDS) + kind, minlength=kinds[..., w, :].size
                                                ).reshape(*sizes, len(KINDS))
            front = (after <= longest) & (before > longest)
            arrival += np.bincount(base[front] * (longest + 2) + after[front], minlength=arrival.size).reshape(arrival.shape)
            if unusual is not None:
                kind = (after <= unseen_window).astype(np.int64) + 2 * (before <= unseen_window)
                index = base * len(KINDS) + kind
                block, known = unusual[start:start + CHUNK][valid], defined[start:start + CHUNK][valid]
                for v in range(len(names)):
                    hits[..., v] += np.bincount(index, weights=block[:, v], minlength=hits[..., v].size).reshape(
                        *sizes, len(KINDS))
                    defined_counts[..., v] += np.bincount(index, weights=known[:, v],
                                                          minlength=hits[..., v].size).reshape(*sizes, len(KINDS))
        read += len(positions)
        del unusual

    names_of_runs = [path.name for path in (args.score_dir, *args.by_dir)]
    common = len(os.path.commonprefix(names_of_runs))
    short = [name[common:] or name for name in names_of_runs]
    short = [short[0], " or ".join(short[1:])]
    band_names = [f"{low}" if low == high else (f"{low}-{high}" if high else f">{low - 1}") for low, high in BANDS]
    print(f"[edge-lead] {timestamps[0].date()} to {timestamps[-1].date()} | flagged: the highest {args.top_percent:g}% "
          f"of {short[0]}, {by_top:g}% of {short[1]} | distance: cells of the grid from the area {short[1]} flags at the "
          f"same timestamp, the larger of the row and column offsets")
    counts = kinds[..., 0, :].sum(axis=-1)  # [group, band]
    print(f"\nvalues by distance from the area of {short[1]} (% of the group)")
    print(f"{'':<{NAME}}{'values':>{WIDTH + 4}}" + "".join(f"{name:>{WIDTH}}" for name in band_names))
    for g, name in enumerate((f"{short[0]} only", "neither")):
        total = counts[g].sum()
        print(f"{name:<{NAME}}{int(total):>{WIDTH + 4}d}"
              + "".join(f"{100 * value / max(total, 1):>{WIDTH}.1f}" for value in counts[g]))
    rows_out = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for w, hours in enumerate(WINDOWS_HOURS):
            share = 100 * kinds[..., w, :] / kinds[..., w, :].sum(axis=-1, keepdims=True)
            print(f"\nwithin {hours} h at the same cell, % of the values of each group and distance: {short[1]} flags it "
                  f"after and not before (front), before and not after (trail), both, none")
            print(f"{'distance, group':<{NAME}}" + "".join(f"{kind:>{WIDTH}}" for kind in ("front", "trail", "both", "none"))
                  + f"{'front x':>{WIDTH}}{'trail x':>{WIDTH}}")
            for b, band in enumerate(band_names):
                for g, name in enumerate((f"{short[0]} only", "neither")):
                    ratios = ([share[0, b, 1] / share[1, b, 1], share[0, b, 2] / share[1, b, 2]] if g == 0
                              else [np.nan, np.nan])
                    print(f"{f'{band}, {name}'[:NAME - 1]:<{NAME}}"
                          + "".join(f"{share[g, b, k]:>{WIDTH}.1f}" for k in (1, 2, 3, 0))
                          + "".join(f"{'' if np.isnan(r) else format(r, '.2f'):>{WIDTH}}" for r in ratios))
                    for k, kind in enumerate(KINDS):
                        rows_out.append({"group": "first only" if g == 0 else "neither", "first": short[0],
                                         "second": short[1], "distance": band, "window_hours": hours, "kind": kind,
                                         "share": share[g, b, k], "values": int(counts[g, b]),
                                         "top_percent": args.top_percent, "by_top_percent": by_top})
        print(f"\nfronts within {WINDOWS_HOURS[-1]} h: hours until {short[1]} flags the cell (% of the fronts)")
        steps = range(1, longest + 1)
        print(f"{'distance, group':<{NAME}}{'fronts':>{WIDTH + 4}}" + "".join(f"{f'{s * step_hours}h':>{WIDTH}}" for s in steps)
              + f"{'median h':>{WIDTH}}")
        for b, band in enumerate(band_names):
            for g, name in enumerate((f"{short[0]} only", "neither")):
                counted = arrival[g, b, 1:longest + 1]
                total = counted.sum()
                median = (np.searchsorted(np.cumsum(counted), total / 2) + 1) * step_hours if total else np.nan
                print(f"{f'{band}, {name}'[:NAME - 1]:<{NAME}}{int(total):>{WIDTH + 4}d}"
                      + "".join(f"{100 * value / max(total, 1):>{WIDTH}.1f}" for value in counted)
                      + f"{'-' if np.isnan(median) else format(median, '.0f'):>{WIDTH}}")
        if hits is not None:
            print(f"\n% of the values where an unseen variable is in its highest {args.unusual_percent:g}% for the cell, "
                  f"the calendar month and the time of day, by kind within {UNSEEN_WINDOW_HOURS} h")
            print(f"{'distance, group, variable':<{NAME}}" + "".join(f"{kind:>{WIDTH}}" for kind in ("front", "trail", "both", "none")))
            for b, band in enumerate(band_names):
                for g, name in enumerate((f"{short[0]} only", "neither")):
                    for v, variable in enumerate(names):
                        share = 100 * hits[g, b, :, v] / defined_counts[g, b, :, v]
                        print(f"{f'{band}, {name}, {variable}'[:NAME - 1]:<{NAME}}"
                              + "".join(f"{'-' if np.isnan(share[k]) else format(share[k], '.1f'):>{WIDTH}}" for k in (1, 2, 3, 0)))
    print("front: the second score flags the cell after and not before, within the window; trail: before and not after; "
          "front x, trail x: the share of the first score only over that of neither, at the same distance")
    if args.unseen_dir is not None and read != len(timestamps):
        raise ValueError(f"Read {read} of the {len(timestamps)} timestamps")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(args.csv, index=False)
        print(f"written {args.csv}")
    if windows_of_events is not None:
        event_report(event_kinds, windows_of_events, band_names, args.bootstrap, args.seed, short, args.events_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
