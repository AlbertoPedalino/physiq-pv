"""When does the score flag? The share of flagged values by calendar month, by time of day and by year.

A value is flagged when it is in the highest `--top-percent` of the scores of the run. If the score
did not depend on the season, every month would have that same share: months far above it are
those the detector finds unusual as a whole, which is what a seasonal normalization should remove.
A share that grows with the years says that the scored period drifts away from the training one.

    python scripts/check_era5_score_by_month.py --run-dir outputs/run_bal_rarity
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

CHUNK = 512  # timestamps read at a time


def flagged_shares(scores, timestamps, threshold):
    """Share of the finite values above `threshold`, by calendar month, by hour of the day and by year (three series)."""
    flagged, finite = np.empty(len(scores)), np.empty(len(scores))
    for start in range(0, len(scores), CHUNK):
        block = np.asarray(scores[start:start + CHUNK])
        flagged[start:start + CHUNK] = (block > threshold).sum(axis=1)
        finite[start:start + CHUNK] = np.isfinite(block).sum(axis=1)
    counts = pd.DataFrame({"flagged": flagged, "finite": finite})
    share = lambda keys: (lambda total: total.flagged / total.finite)(counts.groupby(np.asarray(keys)).sum())
    return share(timestamps.month), share(timestamps.hour), share(timestamps.year)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Run with scores, e.g. a rebalanced run")
    parser.add_argument("--score-component", choices=COMPONENTS, default="combined")
    parser.add_argument("--top-percent", type=float, default=5, help="Score threshold: this highest share of the values")
    args = parser.parse_args(argv)
    manifest = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    timestamps = pd.DatetimeIndex(np.load(args.run_dir / "test_timestamps.npy"))
    scores = np.load(args.run_dir / score_file(manifest, args.score_component), mmap_mode="r")
    try:
        threshold = float(np.nanpercentile(np.asarray(scores[::THRESHOLD_STRIDE]), 100 - args.top_percent))
        months, hours, years = flagged_shares(scores, timestamps, threshold)
    finally:
        scores._mmap.close()
    print(f"[by-month] run read from {timestamps[0].date()} to {timestamps[-1].date()} | flagged: the highest "
          f"{args.top_percent:g}% of the scores | share of flagged values, {args.top_percent:g}% if the season did not matter")
    for title, series in (("month", months), ("hour (UTC)", hours), ("year", years)):
        print(f"\n{title:<12}" + "".join(f"{int(key):>7}" for key in series.index))
        print(f"{'flagged %':<12}" + "".join(f"{100 * value:>7.1f}" for value in series))
        print(f"lowest {100 * series.min():.1f}% | highest {100 * series.max():.1f}% | highest over lowest "
              f"{series.max() / series.min():.1f}" if series.min() > 0 else
              f"lowest 0.0% | highest {100 * series.max():.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
