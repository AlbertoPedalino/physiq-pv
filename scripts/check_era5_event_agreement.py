"""Do two runs find the same events, or does one of them miss some whole events?

Two runs can share most of their flagged values and still disagree on whole events: a small event the first
run flags may be just under the threshold of the other at all its values. A value is flagged when it is in the
highest `--top-percent` of the scores of its run. The flagged values of a run are grouped in events: touching
cells at a timestamp (diagonals included) and the same cell at the timestamps before and after, as in
`check_era5_unreached.py`. Every event of one run is then, against the other run:
- found, when the other run flags at least one of its values;
- near, when it does not, but at least one of its values is in the highest `--near-percent` of the other run;
- missed, when none of its values is even there.
The shares are read by size of the event (values), both ways, with the largest events each run misses.

    python scripts/check_era5_event_agreement.py --score-dir outputs/serene14v/val_gen_clim_day \
        --other-dir outputs/good2v/val_gen_clim_day
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import ndimage

from scripts.check_era5_lead import flags
from scripts.check_era5_unreached import CHUNK, SIZES, band_label

LARGEST = 10  # missed events listed for every run
NAME, WIDTH = 12, 10


def events_of(flagged, rows, cols, shape):
    """The event of every flagged value [time, cell], in the order of np.nonzero, and the number of events."""
    times = len(flagged)
    lattice = np.zeros((times, *shape), dtype=bool)
    for start in range(0, times, CHUNK):
        lattice[start:start + CHUNK, rows, cols] = flagged[start:start + CHUNK]
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1] = True  # touching cells at the same timestamp, diagonals included
    structure[0, 1, 1] = structure[2, 1, 1] = True  # the same cell at the timestamps before and after
    labels, count = ndimage.label(lattice, structure=structure, output=np.int32)
    del lattice
    steps, cells = np.nonzero(flagged)
    events = labels[steps, rows[cells], cols[cells]].astype(np.int64)
    return steps, cells, events, count


def agreement(name, other_name, flagged, other, other_near, rows, cols, shape, timestamps, latitudes, longitudes,
              top_percent, near_percent):
    """Print how the events of `flagged` are found, near or missed by the other run."""
    steps, cells, events, count = events_of(flagged, rows, cols, shape)
    size = count + 1
    values = np.bincount(events, minlength=size)
    found = np.bincount(events, weights=other[steps, cells], minlength=size) > 0
    near = ~found & (np.bincount(events, weights=other_near[steps, cells], minlength=size) > 0)
    missed = ~found & ~near
    real = np.arange(1, size)
    print(f"\nevents of {name} ({count}), against {other_name}: % found (it flags a value of the event), near (a value "
          f"in its highest {near_percent:g}%, none in its highest {top_percent:g}%) and missed (neither)")
    print(f"{'size':<{NAME}}{'events':>{WIDTH}}{'found':>{WIDTH}}{'near':>{WIDTH}}{'missed':>{WIDTH}}"
          f"{'values missed':>{WIDTH + 6}}")
    for low, high in (*SIZES, (1, None)):
        chosen = real[(values[real] >= low) & ((values[real] <= high) if high else True)]
        total = max(len(chosen), 1)
        label = "all" if (low, high) == (1, None) else band_label(low, high)
        print(f"{label:<{NAME}}{len(chosen):>{WIDTH}}" + "".join(f"{100 * np.count_nonzero(kind[chosen]) / total:>{WIDTH}.1f}"
                                                             for kind in (found, near, missed))
              + f"{100 * values[chosen][missed[chosen]].sum() / max(values[chosen].sum(), 1):>{WIDTH + 6}.1f}")
    first = np.full(size, len(timestamps), dtype=np.int64)
    np.minimum.at(first, events, steps)
    last = np.zeros(size, dtype=np.int64)
    np.maximum.at(last, events, steps)
    latitude = np.bincount(events, weights=latitudes[cells], minlength=size) / np.maximum(values, 1)
    longitude = np.bincount(events, weights=longitudes[cells], minlength=size) / np.maximum(values, 1)
    step_hours = (timestamps[1] - timestamps[0]) / pd.Timedelta(hours=1) if len(timestamps) > 1 else 0
    largest = real[missed[real]][np.argsort(-values[real][missed[real]], kind="stable")][:LARGEST]
    print(f"largest events of {name} that {other_name} misses")
    print(f"{'start':<20}{'hours':>{WIDTH}}{'values':>{WIDTH}}{'latitude':>{WIDTH}}{'longitude':>{WIDTH}}")
    for event in largest:
        print(f"{str(timestamps[first[event]]):<20}{(last[event] - first[event] + 1) * step_hours:>{WIDTH}.0f}"
              f"{values[event]:>{WIDTH}d}{latitude[event]:>{WIDTH}.1f}{longitude[event]:>{WIDTH}.1f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="The first run")
    parser.add_argument("--other-dir", type=Path, required=True, help="The other run, over the same timestamps and cells")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--near-percent", type=float, default=20, help="Near: in this highest share of the other run")
    args = parser.parse_args(argv)
    if args.near_percent <= args.top_percent:
        raise ValueError("near-percent must be above top-percent")
    first, timestamps, _ = flags(args.score_dir, args.top_percent)
    second, other_timestamps, _ = flags(args.other_dir, args.top_percent)
    if not timestamps.equals(other_timestamps) or first.shape != second.shape:
        raise ValueError("The two runs are not over the same timestamps and cells")
    first_near, _, _ = flags(args.score_dir, args.near_percent)
    second_near, _, _ = flags(args.other_dir, args.near_percent)
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    if len(rows) != first.shape[1]:
        raise ValueError("The grid of the run does not match the cells of the scores")
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    latitudes = np.asarray(grid["latitudes"], dtype=float)[rows]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[cols]
    names = [f"{path.parent.name}/{path.name}" for path in (args.score_dir, args.other_dir)]
    print(f"[event-agreement] {timestamps[0].date()} to {timestamps[-1].date()} | flagged: the highest "
          f"{args.top_percent:g}% of each run | values flagged by both: "
          f"{100 * np.count_nonzero(first & second) / max(np.count_nonzero(first), 1):.1f}% of {names[0]}")
    for name, other_name, mine, other, other_near in ((names[0], names[1], first, second, second_near),
                                                      (names[1], names[0], second, first, first_near)):
        agreement(name, other_name, mine, other, other_near, rows, cols, shape, timestamps, latitudes, longitudes,
                  args.top_percent, args.near_percent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
