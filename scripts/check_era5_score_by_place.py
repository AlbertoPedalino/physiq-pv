"""Where does the score flag, and which variables make the error there?

A value is flagged when it is in the highest `--top-percent` of the scores of the run. The share of
flagged values is read by latitude band and calendar month: if the place and the season did not
matter, every band and month would have that same share.

With `--feature-errors-dir`, the run whose scoring saved the squared error by variable, the check
also says what the error of the flagged values is made of, south and north of SPLIT degrees and in
the four seasons: the share of each variable in the sum of the errors of the flagged values, with
the weights of that run if it has them. That part is read on one timestamp every
`--feature-stride`, an odd number so that every time of day is read.

    python scripts/check_era5_score_by_place.py --run-dir outputs/run --score-component generator \
        --feature-errors-dir outputs/run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts.check_era5_dust_lead import COMPONENTS, THRESHOLD_STRIDE, score_file

CHUNK = 64  # timestamps read at a time
BANDS = ((50, 60), (40, 50), (30, 40), (20, 30))  # degrees north, from the north to the south
SPLIT = 37  # degrees north: northern Africa and the south of the Mediterranean below it, Europe above
SEASONS = {"Dec-Feb": (12, 1, 2), "Mar-May": (3, 4, 5), "Jun-Aug": (6, 7, 8), "Sep-Nov": (9, 10, 11)}
NAME = 36


def flagged_by_place(scores, months, bands, threshold):
    """Share of the finite values above `threshold` for every band of cells and calendar month: [bands, 12]."""
    flagged, finite = np.zeros((len(bands), 12)), np.zeros((len(bands), 12))
    for start in range(0, len(scores), CHUNK):
        block = np.asarray(scores[start:start + CHUNK])
        month = np.asarray(months[start:start + CHUNK]) - 1
        for index, cells in enumerate(bands):
            np.add.at(flagged[index], month, (block[:, cells] > threshold).sum(axis=1))
            np.add.at(finite[index], month, np.isfinite(block[:, cells]).sum(axis=1))
    return flagged / finite


def error_shares(scores, errors, weights, months, places, threshold, stride):
    """What the error of the flagged values is made of: per place and season, the number of flagged values read
    and the share of each variable in the sum of their weighted errors ([places, seasons], [places, seasons, variables])."""
    counts = np.zeros((len(places), len(SEASONS)))
    sums = np.zeros((len(places), len(SEASONS), errors.shape[2]))
    for start in range(0, len(scores), CHUNK * stride):
        read = slice(start, min(start + CHUNK * stride, len(scores)), stride)
        flagged = np.asarray(scores[read]) > threshold
        block = np.asarray(errors[read], dtype=np.float32)
        month = np.asarray(months[read])
        for season, of_season in enumerate(SEASONS.values()):
            rows = np.isin(month, of_season)
            if not rows.any():
                continue
            for place, cells in enumerate(places):
                chosen = flagged[rows][:, cells]
                counts[place, season] += chosen.sum()
                sums[place, season] += (block[rows][:, cells] * chosen[..., None]).sum(axis=(0, 1), dtype=np.float64)
    sums = sums * weights
    with np.errstate(invalid="ignore"):
        return counts, sums / sums.sum(axis=2, keepdims=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Run with scores, e.g. a rebalanced run")
    parser.add_argument("--score-component", choices=COMPONENTS, default="combined")
    parser.add_argument("--top-percent", type=float, default=5, help="Score threshold: this highest share of the values")
    parser.add_argument("--feature-errors-dir", type=Path, help="Run with scores/feature_scores.npy over the same period")
    parser.add_argument("--feature-stride", type=int, default=9, help="The errors by variable are read on one timestamp every this many")
    args = parser.parse_args(argv)
    if args.feature_stride < 1:
        raise ValueError("feature-stride must be at least 1")
    manifest = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    grid = manifest["grid"]
    latitudes = np.asarray(grid["latitudes"], dtype=float)[np.asarray(grid["rows"])]
    timestamps = pd.DatetimeIndex(np.load(args.run_dir / "test_timestamps.npy"))
    months = timestamps.month.to_numpy()
    bands = {f"{south:g}-{north:g}N": np.nonzero((latitudes >= south) & (latitudes < north))[0] for south, north in BANDS}
    bands = {name: cells for name, cells in bands.items() if len(cells)}
    places = {f"south of {SPLIT}N": np.nonzero(latitudes < SPLIT)[0], f"north of {SPLIT}N": np.nonzero(latitudes >= SPLIT)[0]}
    scores = np.load(args.run_dir / score_file(manifest, args.score_component), mmap_mode="r")
    errors = None
    try:
        threshold = float(np.nanpercentile(np.asarray(scores[::THRESHOLD_STRIDE]), 100 - args.top_percent))
        shares = flagged_by_place(scores, months, list(bands.values()), threshold)
        if args.feature_errors_dir is not None:
            source = json.loads((args.feature_errors_dir / "metadata.json").read_text(encoding="utf-8"))
            errors = np.load(args.feature_errors_dir / "scores" / "feature_scores.npy", mmap_mode="r")
            if errors.shape[:2] != scores.shape:
                raise ValueError("The errors by variable and the score are not over the same timestamps and cells")
            names = list(source.get("preparation", {}).get("features") or
                         [f"feature_{index}" for index in range(errors.shape[2])])
            saved = source.get("backend", {}).get("reconstruction_feature_weights")
            weights = np.array([saved[name] for name in names]) if saved else np.ones(len(names))
            counts, made_of = error_shares(scores, errors, weights, months, list(places.values()), threshold,
                                           args.feature_stride)
    finally:
        scores._mmap.close()
        if errors is not None:
            errors._mmap.close()
    print(f"[by-place] run read from {timestamps[0].date()} to {timestamps[-1].date()} | flagged: the highest "
          f"{args.top_percent:g}% of the scores | share of flagged values, {args.top_percent:g}% if place and season did not matter")
    print(f"\n{'latitude':<12}{'cells':>7}" + "".join(f"{month:>7}" for month in range(1, 13)))
    for (name, cells), row in zip(bands.items(), shares):
        print(f"{name:<12}{len(cells):>7}" + "".join(f"{100 * value:>7.1f}" for value in row))
    if errors is None:
        return 0
    columns = [(place, season) for place in range(len(places)) for season in range(len(SEASONS))]
    print(f"\nwhat the error of the flagged values is made of, % by variable ({100 / len(names):.1f} each if equal), "
          f"{'with the weights of the run' if saved else 'the run has no weights'}, one timestamp every {args.feature_stride}")
    print(f"{'':<{NAME}}" + "".join(f"{name:>{9 * len(SEASONS)}}" for name in places))
    print(f"{'':<{NAME}}" + "".join(f"{season:>9}" for _ in places for season in SEASONS))
    print(f"{'flagged values read':<{NAME}}" + "".join(f"{int(counts[column]):>9}" for column in columns))
    for index, name in enumerate(names):
        print(f"{name:<{NAME}}" + "".join(f"{100 * made_of[column][index]:>9.1f}" for column in columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
