"""Which events of a score do the other scores see too, and how sure are the dropout draws of them?

The events of a score are the output of `run_era5_stgan.py events` on it: clusters of flagged cells,
linked in time. The check reads, for every cell of every event at every timestamp (a cell-step),
whether the other scores flag it, and whether the score of the events flags it solidly, that is
still above its threshold one spread of its dropout draws lower (the run of the score must keep that
spread). A value is flagged by a score when it is in its highest `--top-percent`.

Every event is then put in a class by the share of its cell-steps the other scores flag: seen by the
score alone (less than ALONE), partly, or shared (at least SHARED). An event is convinced when at
least half of the cell-steps the score flags in it are solid. The check says, for every class and by
duration, how many events there are and how many are convinced; and, over the cell-steps of the class,
the share the other scores flag and the share only the score of the events flags.

    python scripts/check_era5_events_alone.py --events-dir outputs/run_clim_events_top5 \
        --score-dir outputs/run_clim --by-dirs outputs/run_no_model_clim outputs/run_previous_clim
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

from scripts.check_era5_dust_lead import THRESHOLD_STRIDE, score_file

CHUNK = 64  # timestamps read at a time
ALONE, SHARED = 0.10, 0.50  # shares of the cell-steps of an event the other scores flag
CONVINCED = 0.5  # share of the flagged cell-steps of an event that are solid
DURATIONS = ((0, 3, "one timestep"), (3, 24, "up to 24 h"), (24, np.inf, "more than 24 h"))
NAME, WIDTH = 46, 14


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events-dir", type=Path, required=True, help="Output of the events step on the score")
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the score of the events and its spread")
    parser.add_argument("--by-dirs", type=Path, nargs="+", required=True, help="Runs with the other scores")
    parser.add_argument("--top-percent", type=float, default=5, help="Flagged: this highest share of each score")
    args = parser.parse_args(argv)
    manifest = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))
    spread_path = manifest.get("anomaly_std_file")
    if not spread_path:
        raise ValueError("The run of the score kept no spread of the dropout draws")
    grid = manifest["grid"]
    rows, cols = np.asarray(grid["rows"]), np.asarray(grid["cols"])
    timestamps = pd.DatetimeIndex(np.load(args.score_dir / "test_timestamps.npy"))
    if not pd.DatetimeIndex(np.load(args.events_dir / "timestamps.npy")).equals(timestamps):
        raise ValueError("The events are not over the timestamps of the score")
    clusters = pd.read_csv(args.events_dir / "clusters.csv", usecols=["cluster_id", "event_id"])
    events = pd.read_csv(args.events_dir / "events.csv", usecols=["event_id", "duration_hours"]).set_index("event_id")
    event_of = np.zeros(int(clusters["cluster_id"].max()) + 1, dtype=np.int64)
    event_of[clusters["cluster_id"].to_numpy()] = clusters["event_id"].to_numpy()
    size = int(events.index.max()) + 1
    cells, flagged, others, solid, alone_steps = (np.zeros(size) for _ in range(5))
    paths = [args.score_dir, *args.by_dirs]
    opened = []
    try:
        labels = np.load(args.events_dir / "cluster_labels.npy", mmap_mode="r")
        opened.append(labels)
        if labels.shape[0] != len(timestamps):
            raise ValueError("The labels of the events are not over the timestamps of the score")
        scores = []
        for path in paths:
            if not pd.DatetimeIndex(np.load(path / "test_timestamps.npy")).equals(timestamps):
                raise ValueError(f"The score is not over the same timestamps: {path}")
            of_score = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            scores.append(np.load(path / score_file(of_score, "combined"), mmap_mode="r"))
            opened.append(scores[-1])
            if scores[-1].shape != (len(timestamps), len(rows)):
                raise ValueError(f"The score is not over the cells of the events: {path}")
        spread = np.load(args.score_dir / spread_path, mmap_mode="r")
        opened.append(spread)
        limits = [float(np.nanpercentile(np.asarray(array[::THRESHOLD_STRIDE]), 100 - args.top_percent)) for array in scores]
        for start in range(0, len(timestamps), CHUNK):
            stop = min(start + CHUNK, len(timestamps))
            event = event_of[np.asarray(labels[start:stop])[:, rows, cols]]
            inside = event > 0
            values = [np.asarray(array[start:stop]) for array in scores]
            with np.errstate(invalid="ignore"):
                marks = [block > limit for block, limit in zip(values, limits)]
                sure = marks[0] & (values[0] - np.asarray(spread[start:stop]) > limits[0])
            seen = np.logical_or.reduce(marks[1:])
            chosen = event[inside]
            for total, mask in ((cells, None), (flagged, marks[0]), (others, seen), (solid, sure),
                                (alone_steps, marks[0] & ~seen)):
                total += np.bincount(chosen, weights=None if mask is None else mask[inside].astype(float), minlength=size)
    finally:
        for array in opened:
            array._mmap.close()
    table = events.reindex(np.arange(size)).assign(cells=cells, flagged=flagged, others=others, solid=solid,
                                                   alone=alone_steps)
    table = table[table["cells"] > 0]
    share = table["others"] / table["cells"]
    table["class"] = np.where(share < ALONE, "alone", np.where(share < SHARED, "partly", "shared"))
    with np.errstate(invalid="ignore", divide="ignore"):
        table["convinced"] = (table["solid"] / table["flagged"]) >= CONVINCED
    labels = [path.name for path in paths]
    common = len(os.path.commonprefix(labels))

    def row(label, values):
        print(f"{label:<{NAME}}" + "".join(f"{value:>{WIDTH}}" for value in values))

    def numbers(part, of):
        flags = part["flagged"].sum()
        steps = part["cells"].sum()
        return (f"{len(part)}", f"{100 * len(part) / max(of, 1):.1f}", f"{int(steps)}",
                f"{100 * part['convinced'].mean():.1f}" if len(part) else "-",
                f"{100 * part['solid'].sum() / flags:.1f}" if flags else "-",
                f"{100 * part['others'].sum() / steps:.1f}" if steps else "-",
                f"{100 * part['alone'].sum() / steps:.1f}" if steps else "-")

    print(f"[events-alone] {args.events_dir.name}: {len(table)} events | flagged: the highest {args.top_percent:g}% of "
          f"each score | others: {', '.join(name[common:] or name for name in labels[1:])}")
    print(f"class of an event: the share of its cell-steps the other scores flag, alone under {100 * ALONE:g}%, shared "
          f"from {100 * SHARED:g}% | convinced: at least {100 * CONVINCED:g}% of the cell-steps the score flags in it "
          f"stay flagged one spread of the dropout draws lower")
    row("", ("events", "% of events", "cell-steps", "% convinced", "% steps solid", "% others", "% score only"))
    for name, chosen in (("all events", table.index == table.index),
                         ("seen by the score alone", table["class"] == "alone"),
                         ("partly", table["class"] == "partly"),
                         ("shared with the other scores", table["class"] == "shared")):
        row(name, numbers(table[chosen], len(table)))
    print("\nevents seen by the score alone, by duration")
    row("", ("events", "% of events", "cell-steps", "% convinced", "% steps solid", "% others", "% score only"))
    alone = table[table["class"] == "alone"]
    for low, high, name in DURATIONS:
        part = alone[(alone["duration_hours"] > low) & (alone["duration_hours"] <= high)]
        row(name, numbers(part, len(alone)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
