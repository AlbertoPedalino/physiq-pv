"""Does the score rise where the dust comes from, before documented Saharan dust events?

A list of dust transport events with the regions they reached is read against the score of a run.
The dates of the list are transport windows, so each event is first given its own arrival day in
each region: the first day, from 3 days before the window to its end, in which the regional dust
of the CAMS reanalysis is above `--dust-percentile` of all the days. The score is then read in the
part of northern Africa upwind of the region on the 3 days before the arrival and on the arrival
day, as the percentile of that day among the days of the same calendar month.

An event with no region inside the domain (the Canary Islands) or after the last year of dust has
no arrival day: the score upwind is read on the 3 days before its window and during it.

The boxes upwind are fixed in advance and the disturbance can be elsewhere. With `--events-dir`,
the events database of the same score, the check also looks anywhere over northern Africa: for
every day, the largest anomalous area whose centre is there, as a percentile within the calendar
month, and where that area was before each arrival.

Whatever its cause, something may be flagged near an arrival. For every arrival the check also
takes the highest of these readings: the arrival region on the day before, on the arrival day and
on the day after, and the places upwind on the day before and on the arrival day. The same reading
on every day of the run says how often it is high anyway. With `--feature-errors-dir`, the run
whose scoring saved the squared error by variable, it lists the variables whose error in the
arrival region is unusual on the arrival day.

An event that the run does not cover is left out. With `--years`, only those years of the run are
read: the score threshold and the percentiles within the calendar month are then those of these
years alone, which is what a run scored on fewer years has, so that the two can be compared.

    python scripts/check_era5_dust_events.py --run-dir outputs/run_bal_rarity --dust-dir data/cams_dust \
        --events-dir outputs/run_bal_rarity_events_top5 --feature-errors-dir outputs/run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts.check_era5_dust_lead import THRESHOLD_STRIDE, load_dust, score_file

IBERIA, FRANCE, ITALY, GREECE, CENTRAL = "Iberian Peninsula", "France", "Italy", "Greece and Balkans", "Central Europe"
# name: south, north, west, east. The first five are where the dust arrives, the others where it comes from.
BOXES = {IBERIA: (36, 44, -10, 3), FRANCE: (44.5, 49, -2, 6.5), ITALY: (37, 46, 7, 19), GREECE: (35, 45, 19.5, 29),
         CENTRAL: (47, 54, 7, 20), "Northwest Africa": (27, 36, -10, 3), "Central North Africa": (27, 37, 3.5, 17),
         "Northeast Africa": (25, 33, 17.5, 30), "Western Sahara": (21, 28, -14.5, -5)}
SOURCES = {IBERIA: "Northwest Africa", FRANCE: "Northwest Africa", ITALY: "Central North Africa",
           GREECE: "Northeast Africa", CENTRAL: "Central North Africa"}
OUTSIDE = ("Western Sahara", "Northwest Africa")  # upwind of the Canary Islands, which are outside the domain
AFRICA = (20.5, 37, -14.5, 35)  # south, north, west, east: northern Africa as far as the domain goes
# Transport windows and the regions of the domain they reached; none: the Canary Islands.
EVENTS = (
    ("2018-04-07", "2018-04-09", (CENTRAL,)), ("2018-04-23", "2018-04-23", (CENTRAL,)),
    ("2018-05-26", "2018-05-28", (CENTRAL,)), ("2019-04-22", "2019-04-24", (CENTRAL,)),
    ("2019-06-10", "2019-06-15", (CENTRAL,)), ("2020-02-03", "2020-02-05", ()), ("2020-02-22", "2020-02-29", ()),
    ("2020-05-13", "2020-05-13", (CENTRAL,)), ("2021-02-06", "2021-02-07", (FRANCE, CENTRAL)),
    ("2021-02-15", "2021-02-21", ()), ("2021-02-22", "2021-02-24", (CENTRAL,)),
    ("2021-03-02", "2021-03-04", (CENTRAL,)), ("2022-01-14", "2022-01-17", ()), ("2022-01-29", "2022-02-01", ()),
    ("2022-03-14", "2022-03-20", (IBERIA, FRANCE)), ("2022-03-29", "2022-03-30", (CENTRAL,)),
    ("2023-02-20", "2023-02-24", (IBERIA, FRANCE, CENTRAL)), ("2024-03-21", "2024-03-25", (IBERIA,)),
    ("2024-03-27", "2024-03-29", (ITALY, GREECE)), ("2024-04-06", "2024-04-09", (IBERIA, FRANCE, CENTRAL)),
    ("2024-04-22", "2024-04-24", (GREECE,)), ("2025-02-01", "2025-02-04", (ITALY, GREECE)),
    ("2025-06-01", "2025-06-10", (IBERIA, ITALY, GREECE)),
)
LEAD_DAYS = 3
NEAR = (-1, 0, 1)  # days from the arrival on which the arrival region is read for any signal
ONSET = (-1, 0)  # days from the arrival on which the places upwind are read for any signal
FEATURE_CHUNK = 512  # timestamps of the errors by variable read at a time
ALARM = 95  # percentile of the day among the days of its calendar month
DAY = pd.Timedelta(days=1)


def month_percentile(series):
    """Percentile of every day among the days of the same calendar month, all years."""
    return series.groupby(series.index.month).rank(pct=True) * 100


def score_percentiles(scores, days, latitudes, longitudes, threshold):
    """Per box: the daily share of its cells above `threshold`, as a percentile within the calendar month."""
    result = {}
    for name, (south, north, west, east) in BOXES.items():
        cells = np.nonzero((latitudes >= south) & (latitudes <= north) & (longitudes >= west) & (longitudes <= east))[0]
        if not len(cells):
            raise ValueError(f"Region {name} has no cell in the score grid")
        share = pd.Series(np.nanmean(np.asarray(scores[:, cells], dtype=float) > threshold, axis=1), index=days)
        result[name] = month_percentile(share.groupby(level=0).mean())
    return result


def dust_percentiles(dust, dust_days, dust_latitudes, dust_longitudes):
    """Per arrival region: the percentile of its mean dust among all the days."""
    result = {}
    for name in SOURCES:
        south, north, west, east = BOXES[name]
        inside = (((dust_latitudes >= south) & (dust_latitudes <= north))[:, None]
                  & ((dust_longitudes >= west) & (dust_longitudes <= east))[None, :])
        result[name] = pd.Series(dust[:, inside].mean(axis=1), index=dust_days).rank(pct=True) * 100
    return result


def africa_areas(events_dir, years=None):
    """Per day of the events database: the largest anomalous area whose centre is over northern Africa
    (cells, area_km2, centroid_lat, centroid_lon); a day without one has 0 cells. `years`: only the
    days of these years, first and last."""
    south, north, west, east = AFRICA
    database = sqlite3.connect(Path(events_dir) / "events.sqlite")
    try:
        clusters = pd.read_sql_query(
            "SELECT timestamp, cells, area_km2, centroid_lat, centroid_lon FROM clusters "
            "WHERE centroid_lat BETWEEN ? AND ? AND centroid_lon BETWEEN ? AND ?", database,
            params=(south, north, west, east))
        days = pd.DatetimeIndex(pd.read_sql_query("SELECT timestamp FROM frames", database).timestamp).normalize().unique()
    finally:
        database.close()
    clusters["day"] = pd.to_datetime(clusters.timestamp).dt.normalize()
    columns = ["cells", "area_km2", "centroid_lat", "centroid_lon"]
    largest = (clusters.loc[clusters.groupby("day").cells.idxmax()].set_index("day")[columns]
               if len(clusters) else pd.DataFrame(columns=columns, dtype=float))
    if years is not None:
        days = days[(days.year >= years[0]) & (days.year <= years[1])]
    return largest.reindex(days).fillna({"cells": 0})


def feature_percentiles(run_dir, days, latitudes, longitudes, period=slice(None)):
    """Per arrival region: the daily mean over its cells of the squared error of every variable of
    `run_dir`, as a percentile within the calendar month (a frame of days by variables). `period`:
    the timestamps of the run that `days` are."""
    run_dir = Path(run_dir)
    names = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))["preparation"]["features"]
    stored = np.load(run_dir / "scores" / "feature_scores.npy", mmap_mode="r")
    errors = stored[period]
    try:
        if errors.shape[0] != len(days) or errors.shape[2] != len(names):
            raise ValueError("The errors by variable do not match the timestamps or the variables of the run")
        cells, parts = {}, {name: [] for name in SOURCES}
        for name in SOURCES:
            south, north, west, east = BOXES[name]
            cells[name] = np.nonzero((latitudes >= south) & (latitudes <= north)
                                     & (longitudes >= west) & (longitudes <= east))[0]
        for start in range(0, len(errors), FEATURE_CHUNK):
            block = np.asarray(errors[start:start + FEATURE_CHUNK])
            for name in SOURCES:
                parts[name].append(block[:, cells[name], :].mean(axis=1, dtype=np.float64))
    finally:
        del errors
        stored._mmap.close()
    return {name: month_percentile(pd.DataFrame(np.concatenate(values), index=days, columns=names).groupby(level=0).mean())
            for name, values in parts.items()}


def near_signal(score, region, area_percentile=None):
    """Per day, as if it were an arrival in `region`: the highest percentile among the region on the
    NEAR days and the places upwind on the ONSET days."""
    parts = ([score[region].shift(-lag) for lag in NEAR] + [score[SOURCES[region]].shift(-lag) for lag in ONSET]
             + ([] if area_percentile is None else [area_percentile.shift(-lag) for lag in ONSET]))
    return pd.concat(parts, axis=1).max(axis=1)


def arrival(dust, start, end, level):
    """The first day from LEAD_DAYS before `start` to `end` with the dust at or above `level`, and the
    highest dust percentile of that period; no day: None."""
    window = dust[(dust.index >= start - LEAD_DAYS * DAY) & (dust.index <= end)]
    above = window[window >= level]
    return (None if above.empty else above.index[0]), float(window.max())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Run with scores, e.g. a rebalanced run")
    parser.add_argument("--dust-dir", type=Path, required=True, help="Yearly dust files of check_era5_dust_lead.py")
    parser.add_argument("--score-component", choices=("combined", "generator", "discriminator"), default="combined")
    parser.add_argument("--top-percent", type=float, default=5, help="Score threshold: this highest share of the values")
    parser.add_argument("--dust-percentile", type=float, default=95,
                        help="The arrival day is the first with the regional dust above this percentile of all days")
    parser.add_argument("--events-dir", type=Path,
                        help="Events database of the same score: also look for an anomalous area anywhere over northern Africa")
    parser.add_argument("--feature-errors-dir", type=Path,
                        help="Run with scores/feature_scores.npy: list the variables with an unusual error on the arrival day")
    parser.add_argument("--years", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="Read only these years of the run, e.g. those of a run scored on fewer years")
    args = parser.parse_args(argv)
    manifest = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    grid = manifest["grid"]
    latitudes = np.asarray(grid["latitudes"], dtype=float)[np.asarray(grid["rows"])]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[np.asarray(grid["cols"])]
    days = pd.DatetimeIndex(np.load(args.run_dir / "test_timestamps.npy")).normalize()
    period = slice(None)
    if args.years is not None:
        inside = np.nonzero((days.year >= args.years[0]) & (days.year <= args.years[1]))[0]
        if not len(inside):
            raise ValueError(f"The run has no timestamp in {args.years[0]}-{args.years[1]}")
        period = slice(int(inside[0]), int(inside[-1]) + 1)
        days = days[period]
    stored = np.load(args.run_dir / score_file(manifest, args.score_component), mmap_mode="r")
    scores = stored[period]
    try:
        threshold = float(np.nanpercentile(np.asarray(scores[::THRESHOLD_STRIDE]), 100 - args.top_percent))
        score = score_percentiles(scores, days, latitudes, longitudes, threshold)
    finally:
        del scores
        stored._mmap.close()
    print(f"[dust-events] run read from {days[0].date()} to {days[-1].date()}")
    dust = dust_percentiles(*load_dust(args.dust_dir))
    last_dust_day = next(iter(dust.values())).index[-1]
    print(f"[dust-events] arrival: first day with the regional dust above the {args.dust_percentile:g}th percentile of all "
          f"days, from {LEAD_DAYS} days before the window | score: percentile of the day within its calendar month, "
          f"alarm at {ALARM}")
    areas = area_percentile = None
    if args.events_dir is not None:
        areas = africa_areas(args.events_dir, args.years)
        area_percentile = month_percentile(areas.cells)
        print(f"[dust-events] northern Africa ({AFRICA[0]:g}-{AFRICA[1]:g}N, {AFRICA[2]:g}-{AFRICA[3]:g}E): size of the "
              f"largest anomalous area of the day with its centre there, as a percentile within the calendar month")
    variables = (None if args.feature_errors_dir is None
                 else feature_percentiles(args.feature_errors_dir, days, latitudes, longitudes, period))
    signal = {region: near_signal(score, region, area_percentile) for region in SOURCES}
    anchored, anywhere, near, unusual = [], [], [], []
    for start, end, regions in EVENTS:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if end > days[-1] or start - 2 * LEAD_DAYS * DAY < days[0]:  # the run does not cover the event
            continue
        print(f"\n===== {start.date()} -> {end.date()} =====")
        if regions and end <= last_dust_day:
            for region in regions:
                day, highest = arrival(dust[region], start, end, args.dust_percentile)
                if day is None:
                    print(f"{region:<19} dust below the level in the whole window (max {highest:.1f})")
                    continue
                source = score[SOURCES[region]]
                before = [float(source[day - lag * DAY]) for lag in range(LEAD_DAYS, -1, -1)]
                anchored.append(before)
                print(f"{region:<19} arrival {day.date()} (dust {dust[region][day]:.1f}, max {highest:.1f}) | "
                      f"{SOURCES[region]:<21} " + "  ".join(f"day {lag:+d}: {value:3.0f}"
                                                              for lag, value in zip(range(-LEAD_DAYS, 1), before))
                      + f" | arrival region on day 0: {score[region][day]:3.0f}")
                if areas is not None:
                    values = [float(area_percentile[day - lag * DAY]) for lag in range(LEAD_DAYS, -1, -1)]
                    anywhere.append(values)
                    earlier = areas[(areas.index >= day - LEAD_DAYS * DAY) & (areas.index < day)]
                    largest = earlier.loc[earlier.cells.idxmax()]
                    where = ("no anomalous area" if not largest.cells else
                             f"{earlier.cells.idxmax().date()}, {int(largest.cells)} cells, "
                             f"{largest.area_km2 / 1000:.0f} thousand km2, centre {largest.centroid_lat:.1f}N "
                             f"{largest.centroid_lon:.1f}E")
                    print(f"{'':<19} anywhere over northern Africa   " + "  ".join(
                        f"day {lag:+d}: {value:3.0f}" for lag, value in zip(range(-LEAD_DAYS, 1), values))
                          + f" | largest area before the arrival: {where}")
                near.append((region, float(signal[region][day])))
                print(f"{'':<19} any signal near the arrival (region day {NEAR[0]:+d}..{NEAR[-1]:+d}, upwind day "
                      f"{ONSET[0]:+d}..{ONSET[-1]:+d}): max {near[-1][1]:3.0f}")
                if variables is not None:
                    row = variables[region].loc[day]
                    high = row[row >= ALARM].sort_values(ascending=False)
                    unusual.append(list(high.index))
                    print(f"{'':<19} variables with an unusual error in the region on the arrival day: "
                          + (", ".join(f"{name} {value:.0f}" for name, value in high.items()) or "none"))
        else:
            reason = "outside the domain" if not regions else "no dust data"
            for source in (OUTSIDE if not regions else sorted({SOURCES[region] for region in regions})):
                series = score[source]
                before = series[(series.index >= start - LEAD_DAYS * DAY) & (series.index < start)]
                during = series[(series.index >= start) & (series.index <= end)]
                print(f"{source:<21} ({reason}: no arrival day) {LEAD_DAYS} days before: max {before.max():3.0f} | "
                      f"during: max {during.max():3.0f} on {during.idxmax().date()}, "
                      f"days at or above {ALARM}: {int((during >= ALARM).sum())} of {len(during)}")
    print(f"\nSUMMARY | arrivals with a dust day: {len(anchored)}")
    if anchored:
        table = np.array(anchored)
        expected = 1 - (ALARM / 100) ** LEAD_DAYS
        print(f"source at or above {ALARM} on at least one of the {LEAD_DAYS} days before: "
              f"{int((table[:, :LEAD_DAYS].max(axis=1) >= ALARM).sum())} ({100 * expected:.0f}% expected by chance, "
              f"{expected * len(table):.1f} arrivals) | on the day before: {int((table[:, -2] >= ALARM).sum())} | "
              f"on the day before or on the arrival day: {int((table[:, -2:].max(axis=1) >= ALARM).sum())}")
        print("mean percentile of the source, day " + "  ".join(
            f"{lag:+d}: {value:.0f}" for lag, value in zip(range(-LEAD_DAYS, 1), table.mean(axis=0))) + " (50 expected)")
        print("Several regions of one event share a source, so the arrivals are not independent.")
    if anywhere:
        table = np.array(anywhere)
        alarm = (area_percentile >= ALARM).astype(float)
        # What any day gives: an alarm on at least one of the days before it, counted on every day of the run.
        before_any_day = alarm.shift(1).rolling(LEAD_DAYS).max().dropna()
        around_any_day = alarm.rolling(2).max().dropna()
        print(f"\nSUMMARY, anywhere over northern Africa | arrivals: {len(table)}")
        print(f"largest area at or above {ALARM} on at least one of the {LEAD_DAYS} days before: "
              f"{int((table[:, :LEAD_DAYS].max(axis=1) >= ALARM).sum())} ({100 * before_any_day.mean():.0f}% of all days, "
              f"{before_any_day.mean() * len(table):.1f} arrivals) | on the day before or on the arrival day: "
              f"{int((table[:, -2:].max(axis=1) >= ALARM).sum())} ({100 * around_any_day.mean():.0f}% of all days, "
              f"{around_any_day.mean() * len(table):.1f} arrivals)")
        print("mean percentile of the largest area, day " + "  ".join(
            f"{lag:+d}: {value:.0f}" for lag, value in zip(range(-LEAD_DAYS, 1), table.mean(axis=0))) + " (50 expected)")
    if near:
        # The same reading on every day of the run: how often it is high whatever happens.
        expected = sum(float((signal[region] >= ALARM).mean()) for region, _ in near)
        print(f"\nSUMMARY, any signal near the arrival | arrivals: {len(near)}")
        print(f"at or above {ALARM}: {sum(value >= ALARM for _, value in near)} ({expected:.1f} expected: on any day of the "
              f"run the same reading is that high {100 * expected / len(near):.0f}% of the times)")
    if unusual:
        any_day = np.mean([float((variables[region] >= ALARM).sum(axis=1).mean()) for region, _ in near])
        counts = pd.Series([name for names in unusual for name in names], dtype=object).value_counts()
        print(f"\nSUMMARY, variables | variables with an unusual error on the arrival day, per arrival: "
              f"{np.mean([len(names) for names in unusual]):.1f} ({any_day:.1f} on any day)")
        print(f"arrivals with an unusual error, by variable ({(1 - ALARM / 100) * len(unusual):.1f} expected for each): "
              + (", ".join(f"{name} {count}" for name, count in counts.items()) or "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
