"""Which score flags the places where the weather changes fast?

No hand-picked event is used. How fast the weather changes is read in the data themselves: for
every cell and timestamp, the squared difference of each standardised variable from its value
`--lags` timestamps before, over the mean of that difference for the variable, averaged over the
variables. Every variable then counts the same, the slow ones and the fast ones. The standardised
data are those of the shared cache of the run: the distance from the seasonal normal, so a change
there is a change of the weather and not the cycle of the day or of the year.

The values are put in bands by how fast the data change, from the slowest half to the fastest 1%.
For every band and every score the check says how many of the values the score flags, that is how
many are in its highest `--top-percent`: a score that follows the fast changes flags much more than
that share in the fastest bands, one that does not care flags that share everywhere.

Everything is read on one timestamp every `--stride`, an odd number so that every time of day is read.
Nothing is written.

    python scripts/check_era5_score_against_change.py --run-dir outputs/run \
        --score-dirs outputs/run_clim outputs/run_no_model_clim
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
from scripts.check_era5_dust_lead import THRESHOLD_STRIDE, score_file

CHUNK = 64  # timestamps read at a time
BANDS = ((0, 50), (50, 90), (90, 95), (95, 99), (99, 100))  # percentiles of the change of the data
MEAN_STRIDE = 10  # the mean change of a variable is read on one of the read timestamps every this many
NAME, WIDTH = 30, 20


def squared_changes(standardised, rows, lag):
    """Squared difference of every variable from `lag` timestamps before, at the cache rows `rows` (a slice)."""
    now = np.asarray(standardised[rows], dtype=np.float32)
    before = np.asarray(standardised[rows.start - lag:rows.stop - lag:rows.step], dtype=np.float32)
    return (now - before) ** 2


def change_of_the_data(standardised, first, read, lag):
    """How fast the data change at the scored timestamps `read` (a range): [timestamps, cells], each variable
    over its mean change. `first` is the cache row of the first scored timestamp."""
    chunks = [slice(first + start, first + min(start + CHUNK * read.step, read.stop), read.step)
              for start in range(read.start, read.stop, CHUNK * read.step)]
    total, count = 0.0, 0
    for rows in chunks[::MEAN_STRIDE] if len(chunks) >= MEAN_STRIDE else chunks:
        block = squared_changes(standardised, rows, lag)
        total, count = total + block.sum(axis=(0, 1), dtype=np.float64), count + block.shape[0] * block.shape[1]
    means = total / count
    if not (np.isfinite(means).all() and (means > 0).all()):
        raise ValueError("A variable does not change at all: there is nothing to scale its change by")
    return np.concatenate([(squared_changes(standardised, rows, lag) / means.astype(np.float32)).mean(axis=2)
                           for rows in chunks])


def flagged_by_band(change, flagged, finite):
    """Per band of `BANDS` of the change: the number of values with a finite score and the share of them flagged."""
    inner = sorted({edge for band in BANDS for edge in band} - {0, 100})
    edges = dict(zip(inner, np.percentile(change, inner)))
    result = []
    for low, high in BANDS:
        inside = finite.copy()
        if low > 0:
            inside &= change > edges[low]
        if high < 100:
            inside &= change <= edges[high]
        result.append((int(inside.sum()), float(flagged[inside].mean()) if inside.any() else float("nan")))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run with the seasonal normalization: it names the cache")
    parser.add_argument("--score-dirs", type=Path, nargs="+", required=True, help="Runs with a score over the period of the run")
    parser.add_argument("--cache-dir", type=Path, help="Shared cache of the run; default: the one the run names")
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 8], help="The change is read over these many timestamps")
    parser.add_argument("--top-percent", type=float, default=5, help="Flagged: this highest share of each score")
    parser.add_argument("--stride", type=int, default=9, help="Timestamps read: one every this many")
    args = parser.parse_args(argv)
    if args.stride < 1 or args.stride % 2 == 0:
        raise ValueError("stride must be odd and at least 1")
    if min(args.lags) < 1:
        raise ValueError("lags must be at least 1")
    manifest = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    seasonal = manifest.get("backend", {}).get("seasonal_normalization")
    if args.cache_dir is None and not seasonal:
        raise ValueError("The run has no seasonal normalization: give the cache of the standardised data")
    cache = args.cache_dir if args.cache_dir is not None else Path(seasonal["mean_file"]).parent
    if not (cache / MANIFEST).is_file() or not (cache / "test.npy").is_file():
        raise ValueError(f"The shared cache of the run is not there: {cache}")
    timestamps = pd.DatetimeIndex(np.load(args.run_dir / "test_timestamps.npy"))
    common = os.path.commonprefix([path.name for path in args.score_dirs]) if len(args.score_dirs) > 1 else ""
    labels = [path.name[len(common):] or path.name for path in args.score_dirs]
    standardised = np.load(cache / "test.npy", mmap_mode="r")
    try:
        first = len(standardised) - len(timestamps)  # the cache holds a few timestamps of context before the scored period
        if first < 0:
            raise ValueError("The cache and the run are not over the same period")
        read = range(max(0, max(args.lags) - first), len(timestamps), args.stride)
        flagged, finite = [], []
        for path in args.score_dirs:
            of_score = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            if not pd.DatetimeIndex(np.load(path / "test_timestamps.npy")).equals(timestamps):
                raise ValueError(f"The score is not over the timestamps of the run: {path}")
            scores = np.load(path / score_file(of_score, "combined"), mmap_mode="r")
            try:
                if scores.shape != (len(timestamps), standardised.shape[1]):
                    raise ValueError(f"The score is not over the cells of the run: {path}")
                threshold = float(np.nanpercentile(np.asarray(scores[::THRESHOLD_STRIDE]), 100 - args.top_percent))
                values = np.array(scores[read.start:read.stop:read.step])  # a copy: the file is closed below
            finally:
                scores._mmap.close()
            finite.append(np.isfinite(values))
            with np.errstate(invalid="ignore"):
                flagged.append(values > threshold)
        tables = {}
        for lag in args.lags:
            change = change_of_the_data(standardised, first, read, lag)
            tables[lag] = [flagged_by_band(change, marked, known) for marked, known in zip(flagged, finite)]
    finally:
        standardised._mmap.close()
    hours = (timestamps[1] - timestamps[0]) / pd.Timedelta(hours=1)
    print(f"[score-against-change] {timestamps[0].date()} to {timestamps[-1].date()}, one timestamp every {args.stride} | "
          f"flagged: the highest {args.top_percent:g}% of each score")
    for lag, table in tables.items():
        print(f"\nchange of the data in {lag * hours:g} hours: % of the values each score flags, by how fast the data "
              f"change there ({args.top_percent:g} if the change did not matter)")
        print(f"{'band of the change':<{NAME}}{'values':>12}" + "".join(f"{label[:WIDTH - 1]:>{WIDTH}}" for label in labels))
        for index, (low, high) in enumerate(BANDS):
            name = f"{low}-{high}" + (", slowest half" if low == 0 else ", fastest" if high == 100 else "")
            print(f"{name:<{NAME}}{table[0][index][0]:>12}"
                  + "".join(f"{100 * of_score[index][1]:>{WIDTH}.1f}" for of_score in table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
