"""On which days does a score flag a place when the other scores do not?

One score, for instance that of a trained run, and some others over the same period and cells,
for instance the scores that need no model. A value is flagged by a score when it is in its
highest `--top-percent`. For a place given as a box in degrees, every day is read as the share of
the values of the place (its cells at the timestamps of the day) that the first score flags and
none of the others does. The days with the highest share are listed, with the share each score
flags on that day and, with `--feature-errors-dir`, the variables that make most of the error
behind the first score in the values it flags alone.

The list gives dates to look at, one by one: it does not say what the weather was on them.

    python scripts/list_era5_score_alone_days.py --score-dir outputs/run_clim \
        --by-dirs outputs/run_no_model_clim outputs/run_previous_clim --feature-errors-dir outputs/run \
        --zone Piedmont 44 46.5 6.5 9.5
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

from scripts.check_era5_dust_lead import COMPONENTS, THRESHOLD_STRIDE, score_file

CHUNK = 2048  # timestamps read at a time
VARIABLES_SHOWN = 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the score that is asked about")
    parser.add_argument("--by-dirs", type=Path, nargs="+", required=True, help="Runs with the other scores, over the same period")
    parser.add_argument("--score-component", choices=COMPONENTS, default="combined")
    parser.add_argument("--feature-errors-dir", type=Path, help="Run with scores/feature_scores.npy behind the first score")
    parser.add_argument("--zone", nargs=5, metavar=("NAME", "SOUTH", "NORTH", "WEST", "EAST"),
                        default=["Piedmont", "44", "46.5", "6.5", "9.5"])
    parser.add_argument("--months", type=int, nargs="+", choices=range(1, 13), help="Only the days of these calendar months")
    parser.add_argument("--top-percent", type=float, default=5, help="Flagged: this highest share of each score")
    parser.add_argument("--days-shown", type=int, default=10)
    args = parser.parse_args(argv)
    name, (south, north, west, east) = args.zone[0], map(float, args.zone[1:])
    paths = [args.score_dir, *args.by_dirs]
    common = os.path.commonprefix([path.name for path in paths])
    labels = [path.name[len(common):] or path.name for path in paths]
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    latitudes = np.asarray(grid["latitudes"], dtype=float)[np.asarray(grid["rows"])]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[np.asarray(grid["cols"])]
    cells = np.nonzero((latitudes >= south) & (latitudes <= north) & (longitudes >= west) & (longitudes <= east))[0]
    if not len(cells):
        raise ValueError("The place holds no cell of the score")
    timestamps = pd.DatetimeIndex(np.load(args.score_dir / "test_timestamps.npy"))
    opened = []
    try:
        for path in paths:
            if not pd.DatetimeIndex(np.load(path / "test_timestamps.npy")).equals(timestamps):
                raise ValueError(f"The score is not over the same timestamps: {path}")
            manifest = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            component = args.score_component if not opened else "combined"
            opened.append(np.load(path / score_file(manifest, component), mmap_mode="r"))
            if opened[-1].shape != (len(timestamps), len(latitudes)):
                raise ValueError(f"The score is not over the same cells: {path}")
        limits = [float(np.nanpercentile(np.asarray(array[::THRESHOLD_STRIDE]), 100 - args.top_percent)) for array in opened]
        # The scores of the place alone: [scores, timestamps, cells of the place].
        place = np.stack([np.concatenate([np.asarray(array[start:start + CHUNK])[:, cells]
                                          for start in range(0, len(timestamps), CHUNK)]) for array in opened])
    finally:
        for array in opened:
            array._mmap.close()
    finite = np.isfinite(place).all(axis=0)
    with np.errstate(invalid="ignore"):
        marks = [finite & (block > limit) for block, limit in zip(place, limits)]
    alone = marks[0] & ~np.logical_or.reduce(marks[1:])
    counts = pd.DataFrame({"alone": alone.sum(axis=1), "values": finite.sum(axis=1),
                           **{label: mark.sum(axis=1) for label, mark in zip(labels, marks)}},
                          index=timestamps).groupby(timestamps.normalize()).sum()
    counts = counts[counts["values"] > 0]
    if args.months:
        counts = counts[counts.index.month.isin(args.months)]
    shares = counts.drop(columns="values").div(counts["values"], axis=0) * 100
    chosen = shares.sort_values("alone", ascending=False).head(args.days_shown)
    made_of = {}
    if args.feature_errors_dir is not None:
        source = json.loads((args.feature_errors_dir / "metadata.json").read_text(encoding="utf-8"))
        errors = np.load(args.feature_errors_dir / "scores" / "feature_scores.npy", mmap_mode="r")
        try:
            if errors.shape[:2] != (len(timestamps), len(latitudes)):
                raise ValueError("The errors by variable and the score are not over the same timestamps and cells")
            names = list(source.get("preparation", {}).get("features") or
                         [f"feature_{index}" for index in range(errors.shape[2])])
            saved = source.get("backend", {}).get("reconstruction_feature_weights")
            weights = np.array([saved[item] for item in names]) if saved else np.ones(len(names))
            for day in chosen.index:
                rows = np.nonzero(timestamps.normalize() == day)[0]
                block = np.asarray(errors[rows[0]:rows[-1] + 1], dtype=np.float32)[:, cells]
                sums = block[alone[rows[0]:rows[-1] + 1]].sum(axis=0, dtype=np.float64) * weights
                if sums.sum() > 0:
                    order = np.argsort(sums)[::-1][:VARIABLES_SHOWN]
                    made_of[day] = ", ".join(f"{names[index]} {100 * sums[index] / sums.sum():.0f}%" for index in order)
        finally:
            errors._mmap.close()
    print(f"[alone-days] {name} {south:g}-{north:g}N {west:g}-{east:g}E, {len(cells)} cells | flagged: the highest "
          f"{args.top_percent:g}% of each score | score: {labels[0]}, {args.score_component}"
          + (f" | months {' '.join(map(str, sorted(args.months)))}" if args.months else ""))
    print("days ranked by the share of the values of the place the score flags and none of the others does; then the "
          "share each score flags on the day")
    width = max(12, *(len(label) + 2 for label in labels))
    print(f"{'day':<12}{'alone':>8}" + "".join(f"{label:>{width}}" for label in labels)
          + ("   most of the error of the score, where it flags alone" if made_of else ""))
    for day, row in chosen.iterrows():
        print(f"{day.date()!s:<12}{row['alone']:>7.0f}%" + "".join(f"{row[label]:>{width - 1}.0f}%" for label in labels)
              + (f"   {made_of.get(day, '-')}" if made_of else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
