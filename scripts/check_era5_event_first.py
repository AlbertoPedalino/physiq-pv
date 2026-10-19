"""In the events both scores flag, which one flags the event first, by how many hours and how far apart?

A value is flagged by a score when it is in its highest `--top-percent` (`--by-top-percent` for the
second score). An event is a group of values flagged by either score, linked in space (touching cells,
diagonals included, at the same timestamp) and in time (the same cell at consecutive timestamps):
the same rule for both scores, so neither starts the events by construction. For every event both
scores flag, the check takes the first timestamp each flags in it and the cells it flags then:
- who is first: the first score, the second, or both at the same timestamp, and by how many hours;
- how far apart the first cells are: the smallest distance, in cells of the grid (the larger of the
  row and column offsets), between the first cells of one and the first cells of the other.
The two scores are treated alike: a lead of the first score is a share "first score first" larger
than "second score first". The events are counted once each, whatever their size, and also by size.

With `--csv` every event both scores flag goes to a file: first timestamps, lead, distance, size.

    python scripts/check_era5_event_first.py --score-dir outputs/run_gen_clim --by-dir outputs/run_simple_max \
        --csv outputs/event_first_5.csv
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

CHUNK = 1024  # timestamps written at a time
LEAD_BANDS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, None))  # steps one score is first by
DISTANCE_BANDS = ((0, 0), (1, 1), (2, 2), (3, 5), (6, None))  # cells between the first cells
SIZES = ((1, 9), (10, 99), (100, 999), (1000, None))  # values of the event (cell-steps flagged by either)
NAME, WIDTH = 36, 10


def first_flags(flagged, labels, rows, cols, size):
    """For every event: the first step a score flags in it (-1: never), and the event, row and column of every
    value it flags at that step."""
    steps, cells = np.nonzero(flagged)  # in order of time
    events = labels[steps, rows[cells], cols[cells]]
    first = np.full(size, -1, dtype=np.int64)
    found, position = np.unique(events, return_index=True)
    first[found] = steps[position]
    chosen = steps == first[events]
    return first, events[chosen], rows[cells[chosen]], cols[cells[chosen]]


def grouped(events, rows, cols):
    """Rows and columns of the values of each event, sorted by event: the events, their starts and the arrays."""
    order = np.argsort(events, kind="stable")
    events, rows, cols = events[order], rows[order], cols[order]
    found, starts = np.unique(events, return_index=True)
    return found, np.append(starts, len(events)), rows, cols


def bands(values, edges):
    """Index of the band of every value; edges are (low, high) pairs, high None for no limit."""
    result = np.full(len(values), -1, dtype=np.int64)
    for index, (low, high) in enumerate(edges):
        result[(values >= low) & ((values <= high) if high is not None else True)] = index
    return result


def band_names(edges, unit):
    return [f"{low * unit}" if low == high else (f"{low * unit}-{high * unit}" if high else f">{(low - 1) * unit}")
            for low, high in edges]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score (the model)")
    parser.add_argument("--by-dir", type=Path, required=True, help="Run with the second score")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--by-top-percent", type=float, help="Level of the second score; default --top-percent")
    parser.add_argument("--csv", type=Path, help="Also write every event both scores flag here")
    args = parser.parse_args(argv)
    first, timestamps, _ = flags(args.score_dir, args.top_percent)
    by_top = args.top_percent if args.by_top_percent is None else args.by_top_percent
    second, other_timestamps, _ = flags(args.by_dir, by_top)
    if not timestamps.equals(other_timestamps) or first.shape != second.shape:
        raise ValueError("The two scores are not over the same timestamps and cells")
    step_hours = int(round((timestamps[1] - timestamps[0]) / pd.Timedelta(hours=1)))
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    if len(rows) != first.shape[1]:
        raise ValueError("The grid of the run does not match the cells of the scores")
    times = len(timestamps)
    either = np.zeros((times, *shape), dtype=bool)
    for start in range(0, times, CHUNK):
        either[start:start + CHUNK, rows, cols] = first[start:start + CHUNK] | second[start:start + CHUNK]
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1] = True  # touching cells at the same timestamp, diagonals included
    structure[0, 1, 1] = structure[2, 1, 1] = True  # the same cell at the timestamps before and after
    labels, count = ndimage.label(either, structure=structure, output=np.int32)
    del either
    size = count + 1
    values = np.bincount(labels.ravel(), minlength=size)
    values[0] = 0
    starts, ends = np.full(size, times, dtype=np.int64), np.full(size, -1, dtype=np.int64)
    for start in range(0, times, CHUNK):
        block = labels[start:start + CHUNK].reshape(min(CHUNK, times - start), -1)
        for offset, frame in enumerate(block):
            present = np.unique(frame)
            starts[present] = np.minimum(starts[present], start + offset)
            ends[present] = start + offset
    first_step, model_events, model_rows, model_cols = first_flags(first, labels, rows, cols, size)
    second_step, other_events, other_rows, other_cols = first_flags(second, labels, rows, cols, size)
    del labels
    both = np.nonzero((first_step >= 0) & (second_step >= 0))[0]
    only_first = int(((first_step >= 0) & (second_step < 0)).sum())
    only_second = int(((first_step < 0) & (second_step >= 0)).sum())
    lead = second_step[both] - first_step[both]  # steps the first score is first by (negative: the second is first)

    # Smallest distance between the first cells of the two scores, event by event.
    found_a, bounds_a, rows_a, cols_a = grouped(model_events, model_rows, model_cols)
    found_b, bounds_b, rows_b, cols_b = grouped(other_events, other_rows, other_cols)
    where_a, where_b = np.searchsorted(found_a, both), np.searchsorted(found_b, both)
    distance = np.empty(len(both), dtype=np.int64)
    for index, (a, b) in enumerate(zip(where_a, where_b)):
        ra, ca = rows_a[bounds_a[a]:bounds_a[a + 1]], cols_a[bounds_a[a]:bounds_a[a + 1]]
        rb, cb = rows_b[bounds_b[b]:bounds_b[b + 1]], cols_b[bounds_b[b]:bounds_b[b + 1]]
        if len(ra) * len(rb) > 4_000_000:  # large first areas: distance from a transform of the grid
            lattice = np.ones(shape, dtype=bool)
            lattice[rb, cb] = False
            distance[index] = ndimage.distance_transform_cdt(lattice, metric="chessboard")[ra, ca].min()
        else:
            distance[index] = np.maximum(np.abs(ra[:, None] - rb[None, :]), np.abs(ca[:, None] - cb[None, :])).min()

    names = [path.name for path in (args.score_dir, args.by_dir)]
    common = len(os.path.commonprefix(names))
    short = [name[common:] or name for name in names]
    duration = (ends - starts + 1) * step_hours
    print(f"[event-first] {timestamps[0].date()} to {timestamps[-1].date()} | flagged: the highest {args.top_percent:g}% "
          f"of {short[0]}, {by_top:g}% of {short[1]} | event: values flagged by either, touching in space or at the same "
          f"cell at consecutive timestamps")
    largest = int(np.argmax(values))
    print(f"events: {count}; flagged by both {len(both)}, only by {short[0]} {only_first}, only by {short[1]} "
          f"{only_second} | the largest: {values[largest]} values ({100 * values[largest] / values.sum():.1f}% of all), "
          f"{duration[largest]} h")
    lead_names, distance_names = band_names(LEAD_BANDS, step_hours), band_names(DISTANCE_BANDS, 1)
    size_class = bands(values[both], SIZES)
    groups = [("all events of both", np.ones(len(both), dtype=bool))] + [
        (f"{name} values", size_class == index) for index, name in enumerate(band_names(SIZES, 1))]
    with np.errstate(invalid="ignore", divide="ignore"):
        print(f"\nwho flags the event first (% of the events), and by how many hours")
        print(f"{'events':<{NAME}}{'count':>{WIDTH}}{f'{short[0]} 1st':>{WIDTH + 4}}{'together':>{WIDTH}}"
              f"{f'{short[1]} 1st':>{WIDTH + 4}}" + "".join(f"{f'+{name}h':>{WIDTH}}" for name in lead_names)
              + "".join(f"{f'-{name}h':>{WIDTH}}" for name in lead_names))
        for name, chosen in groups:
            part = lead[chosen]
            total = max(len(part), 1)
            ahead, behind = bands(part, LEAD_BANDS), bands(-part, LEAD_BANDS)
            print(f"{name[:NAME - 1]:<{NAME}}{len(part):>{WIDTH}d}{100 * (part > 0).sum() / total:>{WIDTH + 4}.1f}"
                  f"{100 * (part == 0).sum() / total:>{WIDTH}.1f}{100 * (part < 0).sum() / total:>{WIDTH + 4}.1f}"
                  + "".join(f"{100 * (ahead == b).sum() / total:>{WIDTH}.1f}" for b in range(len(LEAD_BANDS)))
                  + "".join(f"{100 * (behind == b).sum() / total:>{WIDTH}.1f}" for b in range(len(LEAD_BANDS))))
        print(f"\ncells between the first cells of the two scores (% of the events of each kind)")
        print(f"{'events':<{NAME}}{'count':>{WIDTH}}" + "".join(f"{name:>{WIDTH}}" for name in distance_names)
              + f"{'median':>{WIDTH}}")
        kinds = [(f"{short[0]} first", lead > 0), ("together", lead == 0), (f"{short[1]} first", lead < 0)]
        for name, chosen in kinds:
            part = distance[chosen]
            total = max(len(part), 1)
            within = bands(part, DISTANCE_BANDS)
            median = f"{np.median(part):.0f}" if len(part) else "-"
            print(f"{name[:NAME - 1]:<{NAME}}{len(part):>{WIDTH}d}"
                  + "".join(f"{100 * (within == b).sum() / total:>{WIDTH}.1f}" for b in range(len(DISTANCE_BANDS)))
                  + f"{median:>{WIDTH}}")
    print(f"+Nh: {short[0]} flags the event N hours before {short[1]}; -Nh: after. Distance: the smallest offset, in "
          f"cells of the grid, between the cells each score flags at its first timestamp in the event")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"event": both, "first_time": timestamps[first_step[both]],
                      "by_first_time": timestamps[second_step[both]], "lead_hours": lead * step_hours,
                      "distance_cells": distance, "values": values[both], "duration_hours": duration[both],
                      "top_percent": args.top_percent, "by_top_percent": by_top}).to_csv(args.csv, index=False)
        print(f"written {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
