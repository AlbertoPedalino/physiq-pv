"""Does the score flag the documented days of 2019?

The days of 2019 with a documented event over Piedmont and over France and northern Italy (the cold
of January and February, the days of April, the heat of June and July, the floods of October and
November) are read against the score of a run. The reading is the one of
`check_era5_dust_events.py`: per place, the share of its cells above the score threshold on that
day, as a percentile among the days of the same calendar month.

    python scripts/check_era5_known_days.py --run-dir outputs/run_bal_rarity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts.check_era5_dust_events import ALARM, month_percentile
from scripts.check_era5_dust_lead import COMPONENTS, THRESHOLD_STRIDE, score_file

# name: south, north, west, east
KNOWN_ZONES = {"Piedmont": (44, 46.5, 6.5, 9.5), "France and northern Italy": (43, 49, 0, 12)}
KNOWN_DAYS = ("2019-01-30", "2019-01-31", "2019-02-01", "2019-02-02", "2019-02-03", "2019-04-22", "2019-04-23",
              "2019-04-24", "2019-04-25", "2019-06-25", "2019-06-26", "2019-06-27", "2019-06-28", "2019-06-29",
              "2019-06-30", "2019-07-15", "2019-07-25", "2019-10-19", "2019-10-20", "2019-10-21", "2019-10-22",
              "2019-10-23", "2019-11-15", "2019-11-22", "2019-11-23", "2019-11-24")


def zone_percentiles(scores, days, latitudes, longitudes, threshold):
    """Per zone: the daily share of its cells above `threshold`, as a percentile within the calendar month."""
    result = {}
    for name, (south, north, west, east) in KNOWN_ZONES.items():
        cells = np.nonzero((latitudes >= south) & (latitudes <= north) & (longitudes >= west) & (longitudes <= east))[0]
        if not len(cells):
            raise ValueError(f"Zone {name} has no cell in the score grid")
        share = pd.Series(np.nanmean(np.asarray(scores[:, cells], dtype=float) > threshold, axis=1), index=days)
        result[name] = month_percentile(share.groupby(level=0).mean())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Run with scores, e.g. a rebalanced run")
    parser.add_argument("--score-component", choices=COMPONENTS, default="combined")
    parser.add_argument("--top-percent", type=float, default=5, help="Score threshold: this highest share of the values")
    args = parser.parse_args(argv)
    manifest = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    grid = manifest["grid"]
    latitudes = np.asarray(grid["latitudes"], dtype=float)[np.asarray(grid["rows"])]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[np.asarray(grid["cols"])]
    days = pd.DatetimeIndex(np.load(args.run_dir / "test_timestamps.npy")).normalize()
    scores = np.load(args.run_dir / score_file(manifest, args.score_component), mmap_mode="r")
    try:
        threshold = float(np.nanpercentile(np.asarray(scores[::THRESHOLD_STRIDE]), 100 - args.top_percent))
        percentile = zone_percentiles(scores, days, latitudes, longitudes, threshold)
    finally:
        scores._mmap.close()
    known = [day for day in pd.DatetimeIndex(KNOWN_DAYS) if day in days]
    print(f"[known-days] run read from {days[0].date()} to {days[-1].date()} | documented days in it: {len(known)} of "
          f"{len(KNOWN_DAYS)} | score: percentile of the day within its calendar month, alarm at {ALARM}")
    if not known:
        return 0
    print(f"{'day':<12}" + "".join(f"{name:>28}" for name in KNOWN_ZONES))
    for day in known:
        print(f"{str(day.date()):<12}" + "".join(f"{percentile[name][day]:>28.0f}" for name in KNOWN_ZONES))
    readings = len(known) * len(KNOWN_ZONES)
    print(f"\ndays and places at or above {ALARM}, of {readings}: "
          + " | ".join(f"{name} {int(sum(percentile[name][day] >= ALARM for day in known))}" for name in KNOWN_ZONES)
          + f" | both {int(sum(percentile[name][day] >= ALARM for name in KNOWN_ZONES for day in known))} "
          f"({(1 - ALARM / 100) * readings:.1f} expected by chance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
