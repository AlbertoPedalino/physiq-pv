"""Does the score rise in the hours before the dust arrives?

The onsets are read on the 3-hourly dust of the reanalysis (`download_cams_dust.py`): for each
arrival region, a timestamp with the mean dust of the region at or above `--dust-percentile` of
its values, after `--quiet-hours` below it. Around every onset, three readings are followed from
`--before-hours` before to `--after-hours` after, in the arrival region and in the region upwind
of it, each as the share of the cells of the region that are flagged:

- the score: cells above the score threshold;
- the dust: cells with the dust above their own `--dust-percentile`;
- the rise of the dust: cells whose dust rose over the last 24 hours more than their own
  `--dust-percentile` of such rises.

The last two need no model. The score of a region says something before the arrival only if it
is high earlier, or more often, than they are in that same region. Every reading is the percentile of the timestamp among
those of the same calendar month and time of day, so that the season and the time of day do not
count, and an alarm is a percentile of ALARM or more. The onsets of all the regions are read
together; those of a large episode come in several regions and are not independent.

    python scripts/check_era5_dust_onset_hours.py --run-dir outputs/run_bal_rarity --dust-dir data/cams_dust_3h
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts.check_era5_dust_events import ALARM, BOXES, SOURCES
from scripts.check_era5_dust_lead import COMPONENTS, THRESHOLD_STRIDE, score_file
from scripts.download_cams_dust import load_fields

READINGS = ("score, arrival region", "dust, arrival region", "rise of the dust, arrival", "score, upwind", "dust, upwind",
            "rise of the dust, upwind")
RISE_HOURS = 24  # the rise of the dust is read over this many hours
STEP_HOURS = 3


def clock_percentile(series):
    """Percentile of every timestamp among those of the same calendar month and time of day."""
    return series.groupby([series.index.month, series.index.hour]).rank(pct=True) * 100


def inside(latitudes, longitudes, name):
    """The points [..] of a grid, given the coordinates of each, that are in the box `name`."""
    south, north, west, east = BOXES[name]
    return (latitudes >= south) & (latitudes <= north) & (longitudes >= west) & (longitudes <= east)


def onsets(regional, percentile, quiet_hours):
    """Timestamps at which `regional` is at or above its `percentile` after `quiet_hours` below it."""
    above = (regional >= np.percentile(regional, percentile)).astype(float)
    quiet = above.shift(1).rolling(quiet_hours // STEP_HOURS).max() == 0
    return regional.index[(above == 1) & quiet]


def dust_readings(dust, times, percentile):
    """Share of the cells [times, cells] with the dust above their own percentile, and with its rise
    over RISE_HOURS above the percentile of their rises (two series)."""
    steps = RISE_HOURS // STEP_HOURS
    rise = np.full(dust.shape, np.nan, dtype=np.float32)
    rise[steps:] = dust[steps:] - dust[:-steps]
    high = dust > np.percentile(dust, percentile, axis=0)
    rising = rise > np.nanpercentile(rise, percentile, axis=0)  # a comparison with the missing first hours is False
    return pd.Series(high.mean(axis=1), index=times), pd.Series(rising.mean(axis=1), index=times)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Run with scores, e.g. a rebalanced run")
    parser.add_argument("--dust-dir", type=Path, required=True, help="3-hourly dust files of download_cams_dust.py")
    parser.add_argument("--score-component", choices=COMPONENTS, default="combined")
    parser.add_argument("--top-percent", type=float, default=5, help="Score threshold: this highest share of the values")
    parser.add_argument("--dust-percentile", type=float, default=95)
    parser.add_argument("--quiet-hours", type=int, default=72, help="Hours below the level before an onset")
    parser.add_argument("--before-hours", type=int, default=48)
    parser.add_argument("--after-hours", type=int, default=24)
    parser.add_argument("--years", type=int, nargs=2, metavar=("FIRST", "LAST"), help="Read only these years of the run")
    args = parser.parse_args(argv)
    manifest = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    grid = manifest["grid"]
    latitudes = np.asarray(grid["latitudes"], dtype=float)[np.asarray(grid["rows"])]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[np.asarray(grid["cols"])]
    times = pd.DatetimeIndex(np.load(args.run_dir / "test_timestamps.npy"))
    period = slice(None)
    if args.years is not None:
        kept = np.flatnonzero((times.year >= args.years[0]) & (times.year <= args.years[1]))
        if not len(kept):
            raise ValueError(f"The run has no timestamp in {args.years[0]}-{args.years[1]}")
        period = slice(int(kept[0]), int(kept[-1]) + 1)
        times = times[period]
    if not (np.diff(times.asi8) == pd.Timedelta(hours=STEP_HOURS).value).all():
        raise ValueError(f"The run must have a timestamp every {STEP_HOURS} hours")
    fields, field_times, dust_latitudes, dust_longitudes = load_fields(args.dust_dir)
    position = pd.Index(field_times).get_indexer(times)
    if (position < 0).any():
        raise ValueError("The dust does not cover every timestamp of the run")
    fields = fields[position]
    dust_grid = np.meshgrid(dust_latitudes, dust_longitudes, indexing="ij")
    stored = np.load(args.run_dir / score_file(manifest, args.score_component), mmap_mode="r")
    scores = stored[period]
    try:
        threshold = float(np.nanpercentile(np.asarray(scores[::THRESHOLD_STRIDE]), 100 - args.top_percent))
        flagged = {name: pd.Series(np.nanmean(np.asarray(scores[:, inside(latitudes, longitudes, name)], dtype=float)
                                              > threshold, axis=1), index=times)
                   for name in {*SOURCES, *SOURCES.values()}}
    finally:
        del scores
        stored._mmap.close()
    lags = range(-args.before_hours, args.after_hours + 1, STEP_HOURS)
    window = RISE_HOURS // STEP_HOURS
    rows, before, anytime, counts = {name: [] for name in READINGS}, {name: [] for name in READINGS}, \
        {name: [] for name in READINGS}, {}
    for region in SOURCES:
        dust = fields[:, inside(*dust_grid, region)]
        if not dust.shape[1]:
            raise ValueError(f"Region {region} has no point in the dust grid")
        found = onsets(pd.Series(dust.mean(axis=1, dtype=np.float64), index=times), args.dust_percentile, args.quiet_hours)
        # An onset too close to an end of the run has no complete window around it.
        found = found[(found >= times[0] + pd.Timedelta(hours=args.before_hours))
                      & (found <= times[-1] - pd.Timedelta(hours=args.after_hours))]
        counts[region] = len(found)
        upwind = fields[:, inside(*dust_grid, SOURCES[region])]
        if not upwind.shape[1]:
            raise ValueError(f"Region {SOURCES[region]} has no point in the dust grid")
        readings = dict(zip(READINGS, (flagged[region], *dust_readings(dust, times, args.dust_percentile),
                                       flagged[SOURCES[region]], *dust_readings(upwind, times, args.dust_percentile))))
        for name, series in readings.items():
            percentile = clock_percentile(series)
            rows[name].extend([float(percentile[onset + pd.Timedelta(hours=lag)]) for lag in lags] for onset in found)
            # An alarm at any of the timestamps of the RISE_HOURS before, at the onsets and at every timestamp.
            earlier = (percentile >= ALARM).astype(float).shift(1).rolling(window).max()
            before[name].extend(earlier[found])
            anytime[name].append(earlier.dropna())
    total = sum(counts.values())
    print(f"[onset-hours] run read from {times[0]} to {times[-1]} | onset: mean dust of the region at or above its "
          f"{args.dust_percentile:g}th percentile after {args.quiet_hours} hours below | alarm at {ALARM}, "
          f"{100 - ALARM}% of the timestamps by construction")
    print("onsets: " + " | ".join(f"{region} {count}" for region, count in counts.items()) + f" | all {total}")
    if not total:
        return 0
    tables = {name: np.array(values) for name, values in rows.items()}
    for title, value in (("mean percentile (50 = usual)", lambda table: table.mean(axis=0)),
                         (f"onsets with an alarm, % ({100 - ALARM} = by chance)", lambda table: 100 * (table >= ALARM).mean(axis=0))):
        print(f"\n{title}\n{'hours from the onset':<26}" + "".join(f"{lag:>5d}" for lag in lags))
        for name in READINGS:
            print(f"{name:<26}" + "".join(f"{number:>5.0f}" for number in value(tables[name])))
    print(f"\nan alarm at any time in the {RISE_HOURS} hours before, % of the onsets (% of all the timestamps)")
    for name in READINGS:
        usual = pd.concat(anytime[name])
        print(f"{name:<26}{100 * np.nanmean(before[name]):>5.0f} ({100 * usual.mean():.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
