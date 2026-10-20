"""What kind of event does the model extend, where only the model flags the cells next to it?

A value is flagged by a score when it is above `score_limit` of its highest `--top-percent` (the threshold
of the validation for a run that names one). At every timestamp the cells the other scores flag
(`--by-dir`, any of them) make their area, and its 8-connected pieces are the events. A cell that only the
first score flags, at one cell from that area (the edge), is given the event it touches; when it touches
more than one, the largest.

Every event is typed by weather none of the scores reads as such, cell by cell, with percentiles of the
same cell, calendar month and time of day over the period read:
- wind: the 10 m gust above its 98th percentile (local exceedance as in the windstorm indices of Klawa and
  Ulbrich 2003 and Leckebusch et al. 2008);
- heat, cold: the 2 m temperature above its 90th, below its 10th percentile (as TX90p, TN10p of ETCCDI);
- rain: the precipitation of the hour above the 95th percentile of its wet hours (R95p of ETCCDI on wet
  days, with a wet hour of at least WET_M of the hourly accumulation);
- convective: the CAPE above its 95th percentile with a wet hour at the same cell (CAPE alone does not tell
  whether convection starts).
A signal is in an event when the share of its cells with it is at least `--present-ratio` times the share
over all the cells of that calendar month: one signal gives the event its type, two or more make it
mixed, none other. The same signal can be read without the typing, as the share of the event cells.

The check says, for the cells of the events of the other scores and for the edge cells of the first score
alone, the share of each type and of each signal: a type more frequent at the edge than among all event
cells is one the model extends more than the others. Then the same by season and latitude band.

    python scripts/check_era5_edge_event_types.py --score-dir outputs/test_gen_clim \
        --by-dir outputs/test_simple_max outputs/test_simple_cloud_clim outputs/test_simple_convective_clim \
        --prepared-dir data/era5_prepared --unseen-dir data/era5_independent/independent
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import numpy as np
import pandas as pd
from scipy import ndimage

from scripts.check_era5_dust_lead import score_file, score_limit
from scripts.check_era5_score_against_unseen import read_month

SIGNALS = ("wind", "heat", "cold", "rain", "convective")
TYPES = (*SIGNALS, "mixed", "other")
WET_M = 1e-4  # a wet hour: at least 0.1 mm of precipitation in the hour (ERA5 tp in metres)
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
BANDS = ((50, 90), (40, 50), (30, 40), (-90, 30))  # degrees north
NAME, WIDTH = 30, 11


def beyond(values, hours, percent, low=False, among=None):
    """[timestamps, cells] where `values` is above its `percent` percentile (with `low` below it), cell by cell
    among the timestamps of the same time of day; with `among` the percentile is of those values only, and a
    cell with none of them is never beyond."""
    result = np.zeros(values.shape, dtype=bool)
    for hour in np.unique(hours):
        chosen = hours == hour
        block = np.asarray(values[chosen], dtype=np.float64)
        sample = block if among is None else np.where(among[chosen], block, np.nan)
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)  # cells with no value in the sample: NaN, never beyond
            limit = np.nanpercentile(sample, percent, axis=0)
            result[chosen] = block < limit if low else block > limit
    return result


def signals(gust, cape, temperature, rain, hours):
    """[timestamps, cells, SIGNALS] booleans of one calendar month."""
    wet = rain >= WET_M
    return np.stack((beyond(gust, hours, 98), beyond(temperature, hours, 90), beyond(temperature, hours, 10, low=True),
                     wet & beyond(rain, hours, 95, among=wet), wet & beyond(cape, hours, 95)), axis=-1)


def rows_of_features(data, rows, columns):
    """[len(rows), cells, len(columns)] of `data` at the sorted `rows`, read one contiguous run of rows at a time
    so that only `columns` are held in memory."""
    runs = np.split(rows, np.nonzero(np.diff(rows) != 1)[0] + 1)
    return np.concatenate([np.asarray(data[run[0]:run[-1] + 1][:, :, columns], dtype=np.float32) for run in runs])


def event_types(labels, present, base, ratio):
    """Per event (label 1..n of `labels` [cells]): the index in TYPES of its type and its share of every signal,
    from `present` [cells, SIGNALS] and the shares `base` [SIGNALS] of the month."""
    count = np.bincount(labels, minlength=labels.max() + 1)[1:]
    shares = np.stack([np.bincount(labels, weights=present[:, k], minlength=len(count) + 1)[1:] for k in
                       range(len(SIGNALS))], axis=1) / np.maximum(count, 1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        above = (shares > 0) & (shares >= ratio * base[None])  # a signal absent from the month is in no event
    kinds = above.sum(axis=1)
    types = np.where(kinds == 0, TYPES.index("other"), np.where(kinds > 1, TYPES.index("mixed"), above.argmax(axis=1)))
    return types, shares


def edge_events(labels_lattice, rows, cols, shape):
    """For every cell [cells]: the label of the largest event among its 8 neighbours that touches it (0 if none)."""
    sizes = np.bincount(labels_lattice.ravel())
    sizes[0] = 0
    best = np.zeros(shape, dtype=np.int64)
    padded = np.pad(labels_lattice, 1)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == dc == 0:
                continue
            neighbour = padded[1 + dr:1 + dr + shape[0], 1 + dc:1 + dc + shape[1]]
            better = sizes[neighbour] > sizes[best]
            best = np.where(better, neighbour, best)
    return best[rows, cols]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score (the model)")
    parser.add_argument("--by-dir", type=Path, nargs="+", required=True, help="Runs with the other scores")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--by-top-percent", type=float, help="Level of the other scores; default --top-percent")
    parser.add_argument("--prepared-dir", type=Path, required=True, help="Prepared ERA5: the 2 m temperature and the rain")
    parser.add_argument("--unseen-dir", type=Path, required=True, help="Monthly files of the gust and the CAPE")
    parser.add_argument("--present-ratio", type=float, default=2.0, help="A signal is in an event at this many times its share")
    parser.add_argument("--csv", type=Path, help="Also write the shares here")
    args = parser.parse_args(argv)
    by_top = args.top_percent if args.by_top_percent is None else args.by_top_percent
    from physiq_pv.era5.data import load_prepared

    paths = [args.score_dir, *args.by_dir]
    manifests = [json.loads((path / "metadata.json").read_text(encoding="utf-8")) for path in paths]
    timestamps = pd.DatetimeIndex(np.load(args.score_dir / "test_timestamps.npy"))
    for path in args.by_dir:
        if not pd.DatetimeIndex(np.load(path / "test_timestamps.npy")).equals(timestamps):
            raise ValueError(f"The scores are not over the same timestamps: {path}")
    grid = manifests[0]["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    latitudes = np.asarray(grid["latitudes"], dtype=float)[rows]
    longitudes = np.asarray(grid["longitudes"], dtype=float)[cols]
    band_of = np.searchsorted(-np.array([low for low, _ in BANDS[:-1]]), -latitudes, side="left")
    season_of = {month: index for index, months in enumerate(SEASONS.values()) for month in months}

    cubes = load_prepared(args.prepared_dir)[0]
    arrays = [np.load(path / score_file(manifest, "combined"), mmap_mode="r") for path, manifest in zip(paths, manifests)]
    try:
        names = list(cubes.feature_names)
        t2m, tp = names.index("2m_temperature"), names.index("total_precipitation")
        sources = [(cubes.validation, cubes.validation_timestamps), (cubes.test, cubes.test_timestamps),
                   (cubes.train, cubes.train_timestamps)]
        limits = [score_limit(array, args.top_percent if index == 0 else by_top) for index, array in enumerate(arrays)]
        # Cells counted [population, season, band] and their types and signals: populations "event cells" and
        # "edge, first score alone".
        counts = np.zeros((2, len(SEASONS), len(BANDS)))
        typed = np.zeros((2, len(SEASONS), len(BANDS), len(TYPES)))
        signalled = np.zeros((2, len(SEASONS), len(BANDS), len(SIGNALS)))
        for month in range(1, 13):
            positions = np.nonzero(timestamps.month == month)[0]
            if not len(positions):
                continue
            times = timestamps[positions]
            # The gust and the CAPE of the month, in the order of the scored timestamps.
            parts = [read_month(args.unseen_dir, year, month, latitudes, longitudes) for year in sorted(set(times.year))]
            unseen_names = parts[0][2]
            found = parts[0][0].append([part[0] for part in parts[1:]])
            index = found.get_indexer(times)
            if (index < 0).any():
                raise ValueError(f"The unseen variables do not cover every timestamp of month {month}")
            unseen = np.concatenate([part[1] for part in parts])[index]
            del parts
            gust, cape = unseen[..., unseen_names.index("i10fg")], unseen[..., unseen_names.index("cape")]
            del unseen
            # The temperature and the rain, each timestamp from the part of the prepared data that holds it.
            block = np.full((len(times), len(rows), 2), np.nan, dtype=np.float32)
            covered = np.zeros(len(times), dtype=bool)
            for data, data_times in sources:
                rows_of = pd.DatetimeIndex(data_times).get_indexer(times)
                found_here = (rows_of >= 0) & ~covered
                if found_here.any():
                    block[found_here] = rows_of_features(data, rows_of[found_here], [t2m, tp])
                    covered |= found_here
            if not covered.all():
                raise ValueError(f"The prepared data do not cover every timestamp of month {month}")
            present = signals(gust, cape, block[..., 0], block[..., 1], times.hour.to_numpy())
            del block, gust, cape
            base = present.reshape(-1, len(SIGNALS)).mean(axis=0)
            season = season_of[month]
            flags = [np.asarray(array[positions]) > limit for array, limit in zip(arrays, limits)]
            model, others = flags[0], np.logical_or.reduce(flags[1:])
            for step in range(len(positions)):
                area = others[step]
                if not area.any():
                    continue
                lattice = np.zeros(shape, dtype=bool)
                lattice[rows[area], cols[area]] = True
                labels_lattice, _ = ndimage.label(lattice, structure=np.ones((3, 3), dtype=bool))
                labels = labels_lattice[rows, cols]
                types, shares = event_types(labels, present[step], base, args.present_ratio)
                # All the cells of the events of the other scores.
                inside = labels > 0
                kind = types[labels[inside] - 1]
                np.add.at(counts[0, season], band_of[inside], 1)
                np.add.at(typed[0, season], (band_of[inside], kind), 1)
                np.add.at(signalled[0, season], band_of[inside], shares[labels[inside] - 1])
                # The edge cells of the first score alone.
                touched = edge_events(labels_lattice, rows, cols, shape)
                edge = model[step] & ~area & (touched > 0)
                if edge.any():
                    kind = types[touched[edge] - 1]
                    np.add.at(counts[1, season], band_of[edge], 1)
                    np.add.at(typed[1, season], (band_of[edge], kind), 1)
                    np.add.at(signalled[1, season], band_of[edge], shares[touched[edge] - 1])
            del present, flags
    finally:
        cubes.close()
        for array in arrays:
            array._mmap.close()

    common = len(os.path.commonprefix([path.name for path in paths]))
    short = [path.name[common:] or path.name for path in paths]
    print(f"[edge-types] {timestamps[0].date()} to {timestamps[-1].date()} | flagged: above the threshold of the highest "
          f"{args.top_percent:g}% of {short[0]}, {by_top:g}% of {' or '.join(short[1:])} | a signal is in an event at "
          f"{args.present_ratio:g} times its share in the month")
    populations = ("event cells of the others", f"edge, {short[0]} alone")
    rows_out = []

    def table(title, selector):
        print(f"\n{title}")
        print(f"{'':<{NAME}}{'cells':>12}" + "".join(f"{name:>{WIDTH}}" for name in TYPES)
              + "".join(f"{'s:' + name[:6]:>{WIDTH}}" for name in SIGNALS))
        for p, population in enumerate(populations):
            total = counts[p][selector].sum()
            types = typed[p][selector].reshape(-1, len(TYPES)).sum(axis=0)
            mean_signals = signalled[p][selector].reshape(-1, len(SIGNALS)).sum(axis=0)
            values = [100 * value / max(total, 1) for value in (*types, *mean_signals)]
            rows_out.append({"selection": title, "population": population, "cells": int(total),
                             **dict(zip([*TYPES, *("signal_" + name for name in SIGNALS)], values))})
            print(f"{population:<{NAME}}{int(total):>12d}" + "".join(f"{value:>{WIDTH}.1f}" for value in values))

    everything = (slice(None), slice(None))
    table("% of the cells by type of their event (types), and mean share of the event cells with each signal (s:)",
          everything)
    for s, season in enumerate(SEASONS):
        table(f"season {season}", (slice(s, s + 1), slice(None)))
    labels = [f"{low}-{high}N" if high < 90 else f">{low}N" for low, high in BANDS[:-1]] + [f"<{BANDS[-1][1]}N"]
    for b, label in enumerate(labels):
        table(f"latitude {label}", (slice(None), slice(b, b + 1)))
    print(f"\ntypes: one signal at {args.present_ratio:g} times its share in the month gives the type, more than one "
          f"mixed, none other; signals: wind gust > p98, heat t2m > p90, cold t2m < p10, rain > p95 of wet hours "
          f"(>= {WET_M * 1000:g} mm/h), convective CAPE > p95 on a wet hour; percentiles by cell, month and hour")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(args.csv, index=False)
        print(f"written {args.csv}")
    return rows_out


if __name__ == "__main__":
    main()
