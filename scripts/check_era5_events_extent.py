"""Is the part of an event only its score flags the same weather as the part the other scores flag too?

The events of a score are the output of `run_era5_stgan.py events` on it. An event the other scores
flag only in part (between ALONE and SHARED of its cell-steps, as in `check_era5_events_alone.py`) is
split into the cell-steps the other scores flag too and those only the score of the events flags.
Around every such event, at the same timestamp, a ring of `--ring` cells outside every event is the
weather next to it. For the four parts (the part the others flag, the part only the score flags, the
ring, all the values) the check says in how many values each unseen variable is unusual: in its
highest `--unusual-percent` for the cell, the calendar month and the time of day. The part only the
score flags is the same weather as the rest of its event when it is unusual as often as that rest,
and more than the ring around. The same is read for the events only the score flags, against their ring.

With `--wake-hours` the check also says, for the same parts, how many of their values the other scores
flag in the same cell within that many hours before and after. A part only the score flags that the
others flag later, more than the ring around, is the event seen arriving; one they flagged before is
the trail it leaves. `--unseen-dir` can then be left out to read this alone.

With `--run-dir` and `--variables` in the place of `--unseen-dir` the variables are those of the run, read
standardised from its shared cache, and with `--tail low` their lowest values are the unusual ones: a low
solar radiation around an event, for instance.

    python scripts/check_era5_events_extent.py --events-dir outputs/run_clim_events_top5 \
        --score-dir outputs/run_clim --by-dirs outputs/run_no_model_clim --unseen-dir data/era5_independent/independent
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

from scripts.check_era5_dust_lead import THRESHOLD_STRIDE, score_file
from scripts.check_era5_events_alone import ALONE, SHARED
from scripts.check_era5_score_against_unseen import cache_of, read_month, read_month_of_the_run, unusual_values

CHUNK = 64  # timestamps read at a time
PARTS = ("events in part: the others flag", "events in part: score only", "events in part: ring around",
         "events of the score alone", "events of the score alone: ring", "all values")
NAME, WIDTH = 40, 14


def read_scores(paths, timestamps, cells):
    """The score arrays of the runs `paths`, opened as memory maps."""
    opened = []
    for path in paths:
        if not pd.DatetimeIndex(np.load(path / "test_timestamps.npy")).equals(timestamps):
            raise ValueError(f"The score is not over the same timestamps: {path}")
        manifest = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        opened.append(np.load(path / score_file(manifest, "combined"), mmap_mode="r"))
        if opened[-1].shape != (len(timestamps), cells):
            raise ValueError(f"The score is not over the cells of the events: {path}")
    return opened


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events-dir", type=Path, required=True, help="Output of the events step on the score")
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the score of the events")
    parser.add_argument("--by-dirs", type=Path, nargs="+", required=True, help="Runs with the other scores")
    parser.add_argument("--unseen-dir", type=Path, help="Monthly NetCDF files of the unseen variables")
    parser.add_argument("--run-dir", type=Path, help="Training run with the seasonal normalization: its cache is read instead")
    parser.add_argument("--variables", nargs="+", help="With --run-dir: the variables of the run to ask")
    parser.add_argument("--tail", choices=("high", "low"), default="high", help="Unusual: the highest or the lowest values")
    parser.add_argument("--wake-hours", type=int, default=0,
                        help="Also read whether the other scores flag the parts within this many hours before and after")
    parser.add_argument("--top-percent", type=float, default=5, help="Flagged: this highest share of each score")
    parser.add_argument("--unusual-percent", type=float, default=5, help="Unusual: this highest share of a variable")
    parser.add_argument("--ring", type=int, default=2, help="Width of the ring around an event, in cells")
    args = parser.parse_args(argv)
    if args.ring < 1:
        raise ValueError("ring must be at least 1")
    if args.unseen_dir is not None and args.run_dir is not None:
        raise ValueError("Give --unseen-dir or --run-dir, not both")
    if (args.run_dir is not None) != bool(args.variables):
        raise ValueError("Give --run-dir with --variables")
    if args.unseen_dir is None and args.run_dir is None and args.wake_hours <= 0:
        raise ValueError("Give --unseen-dir, --wake-hours, or both")
    asked = args.unseen_dir is not None or args.run_dir is not None
    manifest = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))
    grid = manifest["grid"]
    rows, cols = np.asarray(grid["rows"]), np.asarray(grid["cols"])
    latitudes = np.asarray(grid["latitudes"], dtype=float)[rows]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[cols]
    timestamps = pd.DatetimeIndex(np.load(args.score_dir / "test_timestamps.npy"))
    if not pd.DatetimeIndex(np.load(args.events_dir / "timestamps.npy")).equals(timestamps):
        raise ValueError("The events are not over the timestamps of the score")
    clusters = pd.read_csv(args.events_dir / "clusters.csv", usecols=["cluster_id", "event_id"])
    event_of = np.zeros(int(clusters["cluster_id"].max()) + 1, dtype=np.int64)
    event_of[clusters["cluster_id"].to_numpy()] = clusters["event_id"].to_numpy()
    size = int(clusters["event_id"].max()) + 1
    paths = [args.score_dir, *args.by_dirs]
    labels = np.load(args.events_dir / "cluster_labels.npy", mmap_mode="r")
    scores = []
    try:
        if labels.shape[0] != len(timestamps):
            raise ValueError("The labels of the events are not over the timestamps of the score")
        scores = read_scores(paths, timestamps, len(rows))
        limits = [float(np.nanpercentile(np.asarray(array[::THRESHOLD_STRIDE]), 100 - args.top_percent)) for array in scores]

        def frames(read):
            """Event of every cell and the flags of the scores at the timestamps `read`: [timestamps, cells]."""
            event = event_of[np.asarray(labels[read])[:, rows, cols]]
            values = [np.asarray(array[read]) for array in scores]
            with np.errstate(invalid="ignore"):
                marks = [block > limit for block, limit in zip(values, limits)]
            finite = np.logical_and.reduce([np.isfinite(block) for block in values])
            return event, marks[0], np.logical_or.reduce(marks[1:]), finite

        # The class of every event, from the share of its cell-steps the other scores flag.
        cells, seen = np.zeros(size), np.zeros(size)
        for start in range(0, len(timestamps), CHUNK):
            event, _, others, _ = frames(slice(start, min(start + CHUNK, len(timestamps))))
            inside = event > 0
            cells += np.bincount(event[inside], minlength=size)
            seen += np.bincount(event[inside], weights=others[inside].astype(float), minlength=size)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = seen / cells
        in_part = (cells > 0) & (share >= ALONE) & (share < SHARED)
        alone = (cells > 0) & (share < ALONE)
        names, hits, sizes = None, None, None
        structure = np.ones((1, 3, 3), dtype=bool)
        lattice = labels.shape[1:]

        def groups_of(event, model, others, finite):
            """The masks of PARTS for the cells [timestamps, cells]."""
            part_of = in_part[event] & (event > 0)
            alone_of = alone[event] & (event > 0)
            in_any = event > 0
            rings = []
            for chosen in (part_of, alone_of):
                stack = np.zeros((len(event), *lattice), dtype=bool)
                stack[:, rows, cols] = chosen
                around = ndimage.binary_dilation(stack, structure=structure, iterations=args.ring)[:, rows, cols]
                rings.append(around & ~in_any & finite)
            return (part_of & others & finite, part_of & model & ~others & finite, rings[0],
                    alone_of & model & finite, rings[1], finite)

        steps = args.wake_hours // 3
        wake = np.zeros((len(PARTS), 3))  # values, flagged by the others before, after
        if steps > 0:
            for start in range(0, len(timestamps), CHUNK):
                stop = min(start + CHUNK, len(timestamps))
                low, high = max(0, start - steps), min(len(timestamps), stop + steps)
                event, model, others, finite = frames(slice(low, high))
                total = np.concatenate((np.zeros((1, others.shape[1])), np.cumsum(others, axis=0)))
                at = np.arange(start, stop) - low
                before = total[at] - total[np.maximum(at - steps, 0)] > 0
                after = total[np.minimum(at + steps + 1, len(total) - 1)] - total[at + 1] > 0
                for index, mask in enumerate(groups_of(event[at], model[at], others[at], finite[at])):
                    wake[index] += (mask.sum(), before[mask].sum(), after[mask].sum())
        standardised = None
        if args.run_dir is not None:
            standardised, first, features = cache_of(args.run_dir, timestamps, len(rows))
            unknown = [name for name in args.variables if name not in features]
            if unknown:
                raise ValueError(f"Unknown variables: {unknown}; the run has {features}")
            columns, names = [features.index(name) for name in args.variables], list(args.variables)
        for month in range(1 if asked else 13, 13):
            years = sorted(set(timestamps.year[timestamps.month == month]))
            if not years:
                continue
            if standardised is not None:
                positions = np.nonzero(timestamps.month == month)[0]
                values, hours = read_month_of_the_run(standardised, first, positions, columns), timestamps.hour[positions]
            else:
                parts = [read_month(args.unseen_dir, year, month, latitudes, longitudes) for year in years]
                names = parts[0][2] if names is None else names
                times = parts[0][0].append([part[0] for part in parts[1:]])
                positions = timestamps.get_indexer(times)
                kept = positions >= 0
                values, hours, positions = np.concatenate([part[1] for part in parts])[kept], times[kept].hour, positions[kept]
                del parts
            unusual, defined = unusual_values(values, hours.to_numpy(), args.unusual_percent, low=args.tail == "low")
            del values
            if hits is None:
                hits, sizes = (np.zeros((len(PARTS), len(names))) for _ in range(2))
            for start in range(0, len(positions), CHUNK):
                read = positions[start:start + CHUNK]
                groups = groups_of(*frames(read))
                block, known = unusual[start:start + CHUNK], defined[start:start + CHUNK]
                for index, mask in enumerate(groups):
                    hits[index] += block[mask].sum(axis=0)
                    sizes[index] += known[mask].sum(axis=0)
    finally:
        for array in [labels, *scores, *([standardised] if standardised is not None else [])]:
            array._mmap.close()
    labels_of_runs = [path.name for path in paths]
    common = len(os.path.commonprefix(labels_of_runs))
    print(f"[events-extent] {args.events_dir.name} | flagged: the highest {args.top_percent:g}% of each score | others: "
          f"{', '.join(name[common:] or name for name in labels_of_runs[1:])} | events in part: {int(in_part.sum())}, "
          f"of the score alone: {int(alone.sum())} | ring: {args.ring} cells around, outside every event")
    with np.errstate(invalid="ignore", divide="ignore"):
        if hits is not None:
            print(f"\n% of the values where the variable is in its {'lowest' if args.tail == 'low' else 'highest'} "
                  f"{args.unusual_percent:g}% for the cell, "
                  f"the calendar month and the time of day")
            print(f"{'':<{NAME}}{'values':>{WIDTH}}" + "".join(f"{name:>{WIDTH}}" for name in names))
            for index, name in enumerate(PARTS):
                print(f"{name:<{NAME}}{int(sizes[index].max()):>{WIDTH}}"
                      + "".join(f"{'-' if not size else format(100 * hit / size, '.1f'):>{WIDTH}}"
                                for hit, size in zip(hits[index], sizes[index])))
        if steps > 0:
            print(f"\n% of the values the other scores flag in the same cell within {3 * steps} h before and after")
            print(f"{'':<{NAME}}" + "".join(f"{name:>{WIDTH}}" for name in ("values", "before", "after")))
            for name, (count, before, after) in zip(PARTS, wake):
                print(f"{name:<{NAME}}{int(count):>{WIDTH}}"
                      + "".join(f"{'-' if not count else format(100 * value / count, '.1f'):>{WIDTH}}"
                                for value in (before, after)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
