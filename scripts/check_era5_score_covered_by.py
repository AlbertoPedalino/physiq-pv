"""How much of what a score flags do other scores flag, alone and together?

One score, for instance that of a trained run, and some others over the same period and cells,
for instance the scores that need no model (`score_era5_seasonal_normal.py`, with the seasonal
normal and with the previous value as the answer). A value is flagged by a score when it is in its
highest `--top-percent`. Of the values the first score flags, the check says how many each of the
others flags, how many at least one of them flags and how many none of them flags: the last ones
are what the first score finds that the others, even together, do not.

It also says how many of those values the first score flags again at the next timestamp, in the
same cell, and, with `--feature-errors-dir`, what the error behind the first score is made of by
variable where none of the others flags and where at least one does.

Everything is read on one timestamp every `--stride`, an odd number so that every time of day is
read. Nothing is written.

    python scripts/check_era5_score_covered_by.py --score-dir outputs/run_clim \
        --by-dirs outputs/run_no_model_clim outputs/run_previous_clim --feature-errors-dir outputs/run
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

from scripts.check_era5_dust_lead import COMPONENTS, THRESHOLD_STRIDE, score_file, score_limit

CHUNK = 64  # timestamps read at a time
PARTS = ("none of them", "at least one", "all values")
NAME, WIDTH = 44, 14


def covered(scores, limits, errors, stride):
    """Counts over one timestamp every `stride`. `scores` are the score and the others, `limits` their
    thresholds, `errors` the errors by variable behind the first score, or None."""
    result = {"flagged": 0, "by": np.zeros(len(scores) - 1), "any": 0, "parts": np.zeros(3), "with_next": np.zeros(2),
              "again": np.zeros(2), "error": None}
    for start in range(0, len(scores[0]), CHUNK * stride):
        read = slice(start, min(start + CHUNK * stride, len(scores[0])), stride)
        values = [np.asarray(array[read]) for array in scores]
        later = np.asarray(scores[0][read.start + 1:read.stop + 1:stride])
        finite = np.logical_and.reduce([np.isfinite(block) for block in values])
        with np.errstate(invalid="ignore"):
            marks = [finite & (block > limit) for block, limit in zip(values, limits)]
            later = later > limits[0]
        some = np.logical_or.reduce(marks[1:])
        result["flagged"] += marks[0].sum()
        result["by"] += [(marks[0] & mark).sum() for mark in marks[1:]]
        result["any"] += some.sum()
        parts = (marks[0] & ~some, marks[0] & some, finite)
        result["parts"] += [part.sum() for part in parts]
        for index, part in enumerate(parts[:2]):
            has_next = part[:len(later)]  # the last timestamp of the period has no next one
            result["with_next"][index] += has_next.sum()
            result["again"][index] += later[has_next].sum()
        if errors is not None:
            block = np.asarray(errors[read], dtype=np.float32)
            sums = np.array([block[part].sum(axis=0, dtype=np.float64) for part in parts])
            result["error"] = sums if result["error"] is None else result["error"] + sums
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the score that is asked about")
    parser.add_argument("--by-dirs", type=Path, nargs="+", required=True, help="Runs with the other scores, over the same period")
    parser.add_argument("--feature-errors-dir", type=Path, help="Run with scores/feature_scores.npy behind the first score")
    parser.add_argument("--score-component", choices=COMPONENTS, default="combined",
                        help="Part of the first score to read, for a training run: the reconstruction error (generator), the discriminator part, or their sum")
    parser.add_argument("--top-percent", type=float, default=5, help="Flagged: this highest share of each score")
    parser.add_argument("--stride", type=int, default=9, help="Timestamps read: one every this many")
    args = parser.parse_args(argv)
    if args.stride < 1 or args.stride % 2 == 0:
        raise ValueError("stride must be odd and at least 1")
    paths = [args.score_dir, *args.by_dirs]
    common = os.path.commonprefix([path.name for path in paths])
    labels = [path.name[len(common):] or path.name for path in paths]
    timestamps = pd.DatetimeIndex(np.load(args.score_dir / "test_timestamps.npy"))
    opened, names, saved = [], None, None
    try:
        for path in paths:
            if not pd.DatetimeIndex(np.load(path / "test_timestamps.npy")).equals(timestamps):
                raise ValueError(f"The score is not over the same timestamps: {path}")
            manifest = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            component = args.score_component if not opened else "combined"
            opened.append(np.load(path / score_file(manifest, component), mmap_mode="r"))
            if opened[-1].shape != opened[0].shape:
                raise ValueError(f"The score is not over the same cells: {path}")
        scores = list(opened)
        limits = [score_limit(array, args.top_percent) for array in scores]
        errors = None
        if args.feature_errors_dir is not None:
            source = json.loads((args.feature_errors_dir / "metadata.json").read_text(encoding="utf-8"))
            errors = np.load(args.feature_errors_dir / "scores" / "feature_scores.npy", mmap_mode="r")
            opened.append(errors)
            if errors.shape[:2] != scores[0].shape:
                raise ValueError("The errors by variable and the score are not over the same timestamps and cells")
            names = list(source.get("preparation", {}).get("features") or
                         [f"feature_{index}" for index in range(errors.shape[2])])
            saved = source.get("backend", {}).get("reconstruction_feature_weights")
        result = covered(scores, limits, errors, args.stride)
    finally:
        for array in opened:
            array._mmap.close()
    flagged = result["flagged"]

    def row(label, values, form="{:.1f}"):
        print(f"{label:<{NAME}}" + "".join(f"{form.format(value):>{WIDTH}}" for value in values))

    print(f"[covered-by] {timestamps[0].date()} to {timestamps[-1].date()}, one timestamp every {args.stride} | "
          f"flagged: the highest {args.top_percent:g}% of each score")
    print(f"score: {labels[0]}, {args.score_component}, {int(flagged)} flagged values read")
    chance = 100 * (1 - (1 - args.top_percent / 100) ** len(args.by_dirs))
    print(f"\n% of the values the score flags that are flagged also by ({args.top_percent:g} each, and {chance:.1f} "
          f"at least one, if they had nothing to do with the score)")
    for label, count in zip(labels[1:], result["by"]):
        row(label, [100 * count / flagged])
    row("at least one of them", [100 * result["parts"][1] / flagged])
    row("none of them", [100 * result["parts"][0] / flagged])
    print("\n% of the values at least one of them flags that the score flags")
    row("", [100 * result["parts"][1] / result["any"]])
    print(f"\n% flagged again by the score at the next timestamp, same cell ({args.top_percent:g} if the time did not matter)")
    row("of those none of them flags", [100 * result["again"][0] / result["with_next"][0]])
    row("of those at least one of them flags", [100 * result["again"][1] / result["with_next"][1]])
    if result["error"] is not None:
        weights = np.array([saved[name] for name in names]) if saved else np.ones(len(names))
        sums = result["error"] * weights
        shares = 100 * sums / sums.sum(axis=1, keepdims=True)
        print(f"\nwhat the error behind the score is made of, % by variable ({100 / len(names):.1f} each if equal), "
              f"{'with the weights of the run' if saved else 'the run has no weights'}: where the score flags and "
              f"none of the others does, where at least one does, and over all the values")
        row("", PARTS, "{}")
        for index, name in enumerate(names):
            row(name, shares[:, index])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
