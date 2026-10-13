"""Where a score flags, how many variables are away from their normal, and how far?

A score can flag a value because one variable is very far from its seasonal normal, or because
many are a little away from it together. The check reads the standardised data of the shared cache
of a run with the seasonal normalization: for every variable, the distance from the seasonal normal
in the standard deviations of the training years. A variable is away at a cell and timestamp when
that distance is more than AWAY, and far when it is more than FAR.

A value is flagged by a score when it is in its highest `--top-percent`. The values are in groups:
flagged by the first score and by none of the others, by the first and at least one other, by the
others and not by the first, by none. For every group the check says how many of the variables are
away and how many are far, in how many of its values no variable is far, and how far the farthest
variable is. Then, variable by variable, in how many of the values of the group it is away.

Everything is read on one timestamp every `--stride`, an odd number so that every time of day is
read. Nothing is written.

    python scripts/check_era5_variables_away.py --run-dir outputs/run --score-dir outputs/run_clim \
        --by-dirs outputs/run_no_model_clim
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

from physiq_pv.anomaly_detection.stgan.shared_cache import MANIFEST
from scripts.check_era5_dust_lead import COMPONENTS, THRESHOLD_STRIDE, score_file

CHUNK = 64  # timestamps read at a time
AWAY, FAR = 1.0, 2.0  # distances from the seasonal normal, in standard deviations of the training years
GROUPS = ("score alone", "with another", "others alone", "none", "all values")
STEP, LARGEST = 0.05, 20.0  # the farthest variable is counted in steps of this width, up to this distance
NAME, WIDTH = 46, 14


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run with the seasonal normalization: it names the cache")
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the score that is asked about")
    parser.add_argument("--by-dirs", type=Path, nargs="+", required=True, help="Runs with the other scores, over the same period")
    parser.add_argument("--score-component", choices=COMPONENTS, default="combined")
    parser.add_argument("--top-percent", type=float, default=5, help="Flagged: this highest share of each score")
    parser.add_argument("--stride", type=int, default=9, help="Timestamps read: one every this many")
    args = parser.parse_args(argv)
    if args.stride < 1 or args.stride % 2 == 0:
        raise ValueError("stride must be odd and at least 1")
    manifest = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    seasonal = manifest.get("backend", {}).get("seasonal_normalization")
    if not seasonal:
        raise ValueError("The run has no seasonal normalization: there is no cache of standardised data")
    cache = Path(seasonal["mean_file"]).parent
    if not (cache / MANIFEST).is_file() or not (cache / "test.npy").is_file():
        raise ValueError(f"The shared cache of the run is not there: {cache}")
    paths = [args.score_dir, *args.by_dirs]
    common = os.path.commonprefix([path.name for path in paths])
    labels = [path.name[len(common):] or path.name for path in paths]
    timestamps = pd.DatetimeIndex(np.load(args.score_dir / "test_timestamps.npy"))
    opened = []
    try:
        standardised = np.load(cache / "test.npy", mmap_mode="r")
        opened.append(standardised)
        first = len(standardised) - len(timestamps)  # the cache holds a few timestamps of context before
        if first < 0:
            raise ValueError("The cache and the score are not over the same period")
        names = list(manifest.get("preparation", {}).get("features") or
                     [f"feature_{index}" for index in range(standardised.shape[2])])
        scores = []
        for path in paths:
            if not pd.DatetimeIndex(np.load(path / "test_timestamps.npy")).equals(timestamps):
                raise ValueError(f"The score is not over the same timestamps: {path}")
            of_score = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            scores.append(np.load(path / score_file(of_score, args.score_component if not scores else "combined"),
                                  mmap_mode="r"))
            opened.append(scores[-1])
            if scores[-1].shape != (len(timestamps), standardised.shape[1]):
                raise ValueError(f"The score is not over the same cells: {path}")
        limits = [float(np.nanpercentile(np.asarray(array[::THRESHOLD_STRIDE]), 100 - args.top_percent)) for array in scores]
        groups, variables = len(GROUPS), len(names)
        count, away, far, none_far = (np.zeros(groups) for _ in range(4))
        by_variable = np.zeros((groups, variables))
        farthest = np.zeros((groups, int(LARGEST / STEP) + 1))
        for start in range(0, len(timestamps), CHUNK * args.stride):
            read = slice(start, min(start + CHUNK * args.stride, len(timestamps)), args.stride)
            distance = np.abs(np.asarray(standardised[first + read.start:first + read.stop:read.step], dtype=np.float32))
            values = [np.asarray(array[read]) for array in scores]
            finite = np.logical_and.reduce([np.isfinite(block) for block in values]) & np.isfinite(distance).all(axis=2)
            with np.errstate(invalid="ignore"):
                marks = [finite & (block > limit) for block, limit in zip(values, limits)]
                is_away, is_far = distance > AWAY, distance > FAR
            some = np.logical_or.reduce(marks[1:])
            how_many, how_many_far = is_away.sum(axis=2), is_far.sum(axis=2)
            steps = np.minimum(distance.max(axis=2) / STEP, farthest.shape[1] - 1).astype(np.int64)
            for index, mask in enumerate((marks[0] & ~some, marks[0] & some, ~marks[0] & some,
                                          finite & ~marks[0] & ~some, finite)):
                count[index] += mask.sum()
                away[index] += how_many[mask].sum()
                far[index] += how_many_far[mask].sum()
                none_far[index] += (how_many_far[mask] == 0).sum()
                by_variable[index] += is_away[mask].sum(axis=0)
                farthest[index] += np.bincount(steps[mask], minlength=farthest.shape[1])
    finally:
        for array in opened:
            array._mmap.close()

    def row(label, values, form="{:.1f}"):
        print(f"{label:<{NAME}}" + "".join(
            f"{'-' if isinstance(value, float) and np.isnan(value) else form.format(value):>{WIDTH}}" for value in values))

    with np.errstate(invalid="ignore", divide="ignore"):
        median = [STEP * (np.searchsorted(np.cumsum(line), line.sum() / 2) + 0.5) if line.sum() else float("nan")
                  for line in farthest]
        print(f"[variables-away] {timestamps[0].date()} to {timestamps[-1].date()}, one timestamp every {args.stride} | "
              f"flagged: the highest {args.top_percent:g}% of each score | away: more than {AWAY:g} standard deviation of "
              f"the training years from the seasonal normal, far: more than {FAR:g}")
        print(f"score: {labels[0]}, {args.score_component} | others: {', '.join(labels[1:])}")
        row("", GROUPS, "{}")
        row("values", count, "{:.0f}")
        row(f"variables away, mean of {variables}", away / count, "{:.2f}")
        row(f"variables far, mean of {variables}", far / count, "{:.2f}")
        row("% of the values with no variable far", 100 * none_far / count)
        row("the farthest variable, median distance", median, "{:.2f}")
        print("\n% of the values of the group where the variable is away")
        row("", GROUPS, "{}")
        for index, name in enumerate(names):
            row(name, 100 * by_variable[:, index] / count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
