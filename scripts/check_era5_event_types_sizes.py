"""The events of a score, split by type of weather and by size: how often do the other scores find each kind?

The values a score flags (its highest `--top-percent`, with the threshold of `score_limit`) are grouped in
events in the hypercube of time, latitude and longitude: touching cells at a timestamp (diagonals included)
and the same cell at the timestamps before and after, as in `check_era5_event_agreement.py`. Every event has
- a size: the number of its values (cell and timestamp), in the bands 1, 2-4, 5-19, 20 or more;
- a type, read from weather none of the scores reads as such, with the rule of
  `check_era5_edge_event_types.py`: at every cell and timestamp the signals wind (10 m gust above its 98th
  percentile), heat and cold (2 m temperature above its 90th, below its 10th percentile), rain (the hour above
  the 95th percentile of the wet hours) and convective (CAPE above its 95th percentile in a wet hour), each
  percentile of the same cell, calendar month and time of day; a signal is in an event when its share of the
  values of the event is at least `--present-ratio` times its share over all the values of the same months.
  One signal gives the event its type, two or more make it mixed, none other.

For every type and size the check says how many events there are and how many values they hold, and for
every other score (`--others NAME=DIR`, or `NAME=DIR@component`) the share of the events it finds (it flags
at least one of their values, at its own highest `--top-percent`) and the median, over the events, of the
mean percentile of its score on their values. The same is given by type alone and by size alone.

    python scripts/check_era5_event_types_sizes.py --events G+clim=outputs/gen_clim \
        --others D+clim=outputs/disc_clim simple=outputs/simple_max chen=outputs/spatial/chen_multivariate \
        --prepared-dir data/era5_prepared --unseen-dir data/era5_independent/independent --csv types_sizes.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import numpy as np
import pandas as pd

from physiq_pv.anomaly_detection.stgan.loading import release_pages
from scripts.check_era5_dust_lead import score_file, score_limit
from scripts.check_era5_edge_event_types import SIGNALS, TYPES, rows_of_features, signals
from scripts.check_era5_event_agreement import events_of
from scripts.check_era5_score_correlation import parse_score
from scripts.check_era5_unreached import CHUNK, SIZES, band_label

PERCENTILES = np.linspace(0, 100, 1001)  # the percentile of a score is read on this grid of its quantiles
QUANTILE_STRIDE = 10  # the quantiles of a score are read on one timestamp every this many
NAME = 22


def open_score(source):
    """The score of a run (name, directory, component) opened as a memory map, its metadata and threshold."""
    name, directory, component = source
    manifest = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    return name, manifest, np.load(directory / score_file(manifest, component), mmap_mode="r")


def flagged_values(scores, limit):
    """[timestamps, cells] above `limit` (NaN is not flagged)."""
    flagged = np.zeros(scores.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        for start in range(0, len(scores), CHUNK):
            flagged[start:start + CHUNK] = np.asarray(scores[start:start + CHUNK]) > limit
            release_pages(scores)
    return flagged


def types_of(signal_sums, base_sums, counts, ratio):
    """Index in TYPES of every event, from the sums over its values of the signals and of the shares of the
    signals in their months, with the rule of check_era5_edge_event_types.event_types."""
    shares = signal_sums / np.maximum(counts, 1)[:, None]
    base = base_sums / np.maximum(counts, 1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        above = (shares > 0) & (shares >= ratio * base)  # a signal absent from the months is in no event
    kinds = above.sum(axis=1)
    return np.where(kinds == 0, TYPES.index("other"), np.where(kinds > 1, TYPES.index("mixed"), above.argmax(axis=1)))


def size_bands(sizes):
    """Index in SIZES of every event size."""
    bands = np.full(len(sizes), -1)
    for index, (low, high) in enumerate(SIZES):
        bands[(sizes >= low) & ((sizes <= high) if high else True)] = index
    return bands


def describe(flagged, timestamps, layout, signals_of, others, ratio):
    """Events of `flagged` and, per event: size, type, and per other score found / mean percentile.

    `signals_of(positions)` gives the signals [len(positions), cells, SIGNALS] at the timestamps `positions` of
    one calendar month; `others` is {name: (scores, limit, quantiles)}."""
    rows, cols, shape = layout
    steps, cells, events, count = events_of(flagged, rows, cols, shape)
    events -= 1  # labels start at 1
    sizes = np.bincount(events, minlength=count)
    signal_sums = np.zeros((count, len(SIGNALS)))
    base_sums = np.zeros((count, len(SIGNALS)))
    found = {name: np.zeros(count, dtype=bool) for name in others}
    percentile_sums = {name: np.zeros(count) for name in others}
    percentile_counts = {name: np.zeros(count) for name in others}
    months = timestamps.month.to_numpy()
    for month in range(1, 13):
        positions = np.nonzero(months == month)[0]
        if not len(positions):
            continue
        present = signals_of(positions)
        base = present.reshape(-1, len(SIGNALS)).mean(axis=0)
        at = np.full(len(timestamps), -1)
        at[positions] = np.arange(len(positions))
        chosen = at[steps] >= 0
        local, value_cells, value_events = at[steps[chosen]], cells[chosen], events[chosen]
        np.add.at(signal_sums, value_events, present[local, value_cells])
        np.add.at(base_sums, value_events, np.broadcast_to(base, (len(value_events), len(SIGNALS))))
        for name, (scores, limit, quantiles) in others.items():
            values = np.asarray(scores[positions], dtype=np.float64)[local, value_cells]
            release_pages(scores)
            with np.errstate(invalid="ignore"):
                found[name][value_events[values > limit]] = True
            finite = np.isfinite(values)
            np.add.at(percentile_sums[name], value_events[finite], np.interp(values[finite], quantiles, PERCENTILES))
            np.add.at(percentile_counts[name], value_events[finite], 1)
        del present
    types = types_of(signal_sums, base_sums, sizes, ratio)
    return dict(sizes=sizes, types=types, bands=size_bands(sizes), found=found,
                percentile={name: np.where(percentile_counts[name] > 0, total / np.maximum(percentile_counts[name], 1), np.nan)
                            for name, total in percentile_sums.items()},
                steps=steps, events=events)


def table(described, names):
    """Rows of the summary: by type and size, by type, by size."""
    types, bands, sizes = described["types"], described["bands"], described["sizes"]
    total_events, total_values = len(sizes), sizes.sum()
    groups = [(TYPES[t], band_label(*SIZES[b]), (types == t) & (bands == b)) for t in range(len(TYPES))
              for b in range(len(SIZES))]
    groups += [(TYPES[t], "all", types == t) for t in range(len(TYPES))]
    groups += [("all", band_label(*SIZES[b]), bands == b) for b in range(len(SIZES))]
    groups.append(("all", "all", np.ones(total_events, dtype=bool)))
    result = []
    for kind, size, chosen in groups:
        if not chosen.any():
            continue
        row = {"type": kind, "size": size, "events": int(chosen.sum()),
               "events %": 100 * chosen.sum() / total_events, "values %": 100 * sizes[chosen].sum() / total_values}
        for name in names:
            row[f"{name} found %"] = 100 * described["found"][name][chosen].mean()
            row[f"{name} percentile"] = float(np.nanmedian(described["percentile"][name][chosen]))
        result.append(row)
    return pd.DataFrame(result)


def prepared_signals(prepared_dir, unseen_dir, timestamps, rows, cols, latitudes, longitudes):
    """signals_of(positions) for describe(): the signals of the scored timestamps `positions` of one month, read
    from the gust and CAPE of the monthly files and from the 2 m temperature and rain of the prepared data."""
    from physiq_pv.era5.data import load_prepared
    from scripts.check_era5_score_against_unseen import read_month

    cubes = load_prepared(prepared_dir)[0]
    names = list(cubes.feature_names)
    t2m, tp = names.index("2m_temperature"), names.index("total_precipitation")
    sources = [(cubes.validation, cubes.validation_timestamps), (cubes.test, cubes.test_timestamps),
               (cubes.train, cubes.train_timestamps)]

    def signals_of(positions):
        times = timestamps[positions]
        parts = [read_month(unseen_dir, year, int(times.month[0]), latitudes, longitudes) for year in sorted(set(times.year))]
        unseen_names = parts[0][2]
        index = parts[0][0].append([part[0] for part in parts[1:]]).get_indexer(times)
        if (index < 0).any():
            raise ValueError(f"The unseen variables do not cover every timestamp of month {times.month[0]}")
        unseen = np.concatenate([part[1] for part in parts])[index]
        gust, cape = unseen[..., unseen_names.index("i10fg")], unseen[..., unseen_names.index("cape")]
        block = np.full((len(times), len(rows), 2), np.nan, dtype=np.float32)
        covered = np.zeros(len(times), dtype=bool)
        for data, data_times in sources:
            rows_of = pd.DatetimeIndex(data_times).get_indexer(times)
            here = (rows_of >= 0) & ~covered
            if here.any():
                block[here] = rows_of_features(data, rows_of[here], [t2m, tp])
                covered |= here
            release_pages(data)
        if not covered.all():
            raise ValueError(f"The prepared data do not cover every timestamp of month {times.month[0]}")
        return signals(gust, cape, block[..., 0], block[..., 1], times.hour.to_numpy())

    return signals_of, cubes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events", required=True, help="NAME=DIR[@component]: the score whose flags make the events")
    parser.add_argument("--others", nargs="*", default=[], help="NAME=DIR[@component]: scores read on the events")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--prepared-dir", type=Path, required=True, help="Prepared ERA5: the 2 m temperature and the rain")
    parser.add_argument("--unseen-dir", type=Path, required=True, help="Monthly files of the gust and the CAPE")
    parser.add_argument("--present-ratio", type=float, default=2.0, help="A signal is in an event at this many times its share")
    parser.add_argument("--csv", type=Path, help="Also write the table here")
    args = parser.parse_args(argv)
    sources = [parse_score(args.events), *(parse_score(text) for text in args.others)]
    if len({name for name, _, _ in sources}) != len(sources):
        raise ValueError("Give every score a different name")
    opened = [open_score(source) for source in sources]
    name, manifest, scores = opened[0]
    grid = manifest["grid"]
    timestamps = pd.DatetimeIndex(np.load(sources[0][1] / "test_timestamps.npy"))
    for (other, other_manifest, _), (_, directory, _) in zip(opened[1:], sources[1:]):
        if other_manifest["grid"] != grid or not timestamps.equals(pd.DatetimeIndex(np.load(directory / "test_timestamps.npy"))):
            raise ValueError(f"{other} is not over the grid and timestamps of {name}")
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    layout = (rows, cols, (len(grid["latitudes"]), len(grid["longitudes"])))
    latitudes = np.asarray(grid["latitudes"], dtype=float)[rows]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[cols]
    cubes = None
    try:
        flagged = flagged_values(scores, score_limit(scores, args.top_percent))
        others = {}
        for other, _, array in opened[1:]:
            sample = np.asarray(array[::QUANTILE_STRIDE], dtype=np.float64)
            others[other] = (array, score_limit(array, args.top_percent), np.nanpercentile(sample, PERCENTILES))
            release_pages(array)
        signals_of, cubes = prepared_signals(args.prepared_dir, args.unseen_dir, timestamps, rows, cols, latitudes,
                                             longitudes)
        described = describe(flagged, timestamps, layout, signals_of, others, args.present_ratio)
    finally:
        if cubes is not None:
            cubes.close()
        for _, _, array in opened:
            array._mmap.close()
    summary = table(described, list(others))
    print(f"[types-sizes] {timestamps[0].date()} to {timestamps[-1].date()} | events of {name}, its highest "
          f"{args.top_percent:g}% | {len(described['sizes'])} events | a signal is in an event at "
          f"{args.present_ratio:g} times its share | found: flagged by the other score at its highest {args.top_percent:g}%")
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_rows", None):
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.1f}"))
    if args.csv is not None:
        summary.to_csv(args.csv, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
