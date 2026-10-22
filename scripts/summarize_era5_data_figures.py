"""The numbers behind the three ERA5 data figures of the thesis, from the training years, with no model.

1. Seasonal and daily cycle: in three cells far apart (northern Europe, Po valley, Sahara), the mean
   and the standard deviation of the 2 m temperature and of the total cloud cover by calendar month,
   and their mean by time of day in January and in July.
2. Change over 24 hours: for every variable, the mean squared difference between a value and the one
   24 hours before, after the min-max of the variable over the training years, and its share of the
   sum over the variables. Read on every timestamp and every cell.
3. Correlation against distance: for a fixed random set of cells, the correlation between the
   anomalies of two cells (value minus the mean of its cell, calendar month and time of day), against
   the great-circle distance between them, by 100 km bins: median and quartiles.

The result is a JSON file for figures/genera_figure_dati.py of the thesis; the main numbers are printed.

    python scripts/summarize_era5_data_figures.py --prepared-dir data/era5_prepared \
        --train-end-year 1999 --output era5_dati.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from physiq_pv.era5.data import load_prepared

CELLS = {"Europa settentrionale": (55.0, 10.0), "Pianura Padana": (45.0, 9.0), "Sahara": (25.0, 10.0)}
CYCLE_FEATURES = ("2m_temperature", "total_cloud_cover")
CORRELATION_FEATURES = ("2m_temperature", "mean_sea_level_pressure", "total_cloud_cover", "total_precipitation")
LAG = 8  # timestamps in 24 hours
CHUNK = 256
BIN_KM = 100
MAX_KM = 3000
EARTH_KM = 6371.0


def nearest(latitudes, longitudes, latitude, longitude):
    return int(np.argmin((latitudes - latitude) ** 2 + (longitudes - longitude) ** 2))


def great_circle_km(latitudes, longitudes):
    """Distance [cells, cells] in km."""
    lat, lon = np.radians(latitudes), np.radians(longitudes)
    dlat, dlon = lat[:, None] - lat[None, :], lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def anomalies(values, timestamps):
    """Value minus the mean of its cell, calendar month and time of day [timestamps, cells]."""
    keys = (timestamps.month.to_numpy() - 1) * 24 + timestamps.hour.to_numpy()
    out = values.astype(np.float64)
    for key in np.unique(keys):
        out[keys == key] -= out[keys == key].mean(axis=0)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--train-end-year", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correlation-cells", type=int, default=300)
    parser.add_argument("--correlation-stride", type=int, default=9, help="One timestamp every this many (odd)")
    args = parser.parse_args(argv)
    if args.correlation_stride < 1 or args.correlation_stride % 2 == 0:
        raise ValueError("correlation-stride must be odd, so that every time of day is read")
    cubes = load_prepared(args.prepared_dir, args.train_end_year)[0]
    try:
        array, times = cubes.train, cubes.train_timestamps
        if not (np.diff(times.asi8) == 3 * 3600 * 10**9).all():
            raise ValueError("The training timestamps are not every 3 hours: the 24-hour lag would be wrong")
        names = list(cubes.feature_names)
        latitudes, longitudes = np.asarray(cubes.latitudes, float), np.asarray(cubes.longitudes, float)
        n_cells, n_features = array.shape[1], array.shape[2]
        cycle_cells = {name: nearest(latitudes, longitudes, *where) for name, where in CELLS.items()}
        cycle_columns = [names.index(name) for name in CYCLE_FEATURES]
        rng = np.random.default_rng(0)
        corr_cells = np.sort(rng.choice(n_cells, size=min(args.correlation_cells, n_cells), replace=False))
        corr_columns = [names.index(name) for name in CORRELATION_FEATURES]
        corr_positions = np.arange(0, len(array), args.correlation_stride)

        minimum = np.full(n_features, np.inf)
        maximum = np.full(n_features, -np.inf)
        squares = np.zeros(n_features)
        pairs = 0
        cycle = np.empty((len(array), len(cycle_cells), len(cycle_columns)), dtype=np.float32)
        corr = np.empty((len(corr_positions), len(corr_cells), len(corr_columns)), dtype=np.float32)
        tail = None
        for start in range(0, len(array), CHUNK):
            block = np.asarray(array[start:start + CHUNK], dtype=np.float32)
            minimum = np.minimum(minimum, block.min(axis=(0, 1)))
            maximum = np.maximum(maximum, block.max(axis=(0, 1)))
            joined = block if tail is None else np.concatenate([tail, block])
            change = (joined[LAG:] - joined[:-LAG]).astype(np.float64)
            squares += (change ** 2).sum(axis=(0, 1))
            pairs += change.shape[0] * n_cells
            tail = joined[-LAG:]
            cycle[start:start + len(block)] = block[:, list(cycle_cells.values())][:, :, cycle_columns]
            inside = corr_positions[(corr_positions >= start) & (corr_positions < start + len(block))]
            first = np.searchsorted(corr_positions, start)
            corr[first:first + len(inside)] = block[inside - start][:, corr_cells][:, :, corr_columns]

        # 2. Change over 24 hours after the min-max of each variable.
        span = np.where(maximum > minimum, maximum - minimum, 1.0)
        mean_square = squares / pairs / span ** 2
        share = mean_square / mean_square.sum()
        print(f"[data-figures] training {times[0].year}-{times[-1].year}, {len(array)} timestamps, {n_cells} cells")
        print("\nchange over 24 hours after the min-max: share of the sum over the variables (%)")
        for index in np.argsort(-share):
            print(f"  {names[index]:36s} {100 * share[index]:6.2f}")

        # 1. Cycles.
        months = times.month.to_numpy()
        hours = times.hour.to_numpy()
        cycles = {}
        print("\nseasonal cycle: monthly means (min, max) and mean standard deviation within a month")
        for c, (cell, index) in enumerate(cycle_cells.items()):
            cycles[cell] = {"latitude": float(latitudes[index]), "longitude": float(longitudes[index])}
            for f, feature in enumerate(CYCLE_FEATURES):
                values = cycle[:, c, f].astype(np.float64)
                by_month = [values[months == m] for m in range(1, 13)]
                monthly_mean = [float(v.mean()) for v in by_month]
                monthly_std = [float(v.std()) for v in by_month]
                daily = {str(m): [float(values[(months == m) & (hours == h)].mean()) for h in range(0, 24, 3)]
                         for m in (1, 7)}
                cycles[cell][feature] = {"monthly_mean": monthly_mean, "monthly_std": monthly_std, "hourly_mean": daily}
                print(f"  {cell:22s} {feature:20s} mean {min(monthly_mean):9.3f} to {max(monthly_mean):9.3f}, "
                      f"std within month {np.mean(monthly_std):7.3f}")

        # 3. Correlation against distance.
        distance = great_circle_km(latitudes[corr_cells], longitudes[corr_cells])
        upper = np.triu_indices(len(corr_cells), k=1)
        edges = np.arange(0, MAX_KM + BIN_KM, BIN_KM)
        which = np.digitize(distance[upper], edges) - 1
        correlation = {}
        print("\ncorrelation of the anomalies against distance: median at 100-200 km, 500-600 km, "
              "1000-1100 km; first bin with median below 0.5")
        corr_times = times[corr_positions]
        for f, feature in enumerate(CORRELATION_FEATURES):
            values = anomalies(corr[:, :, f], corr_times)
            keep = values.std(axis=0) > 0
            matrix = np.full((len(corr_cells), len(corr_cells)), np.nan)
            matrix[np.ix_(keep, keep)] = np.corrcoef(values[:, keep].T)
            pair = matrix[upper]
            rows = []
            for b in range(len(edges) - 1):
                inside = pair[(which == b) & np.isfinite(pair)]
                rows.append([float(np.median(inside)), float(np.percentile(inside, 25)), float(np.percentile(inside, 75)),
                             int(inside.size)] if inside.size else [None, None, None, 0])
            correlation[feature] = rows
            medians = [row[0] for row in rows]
            below = next((edges[b] for b, m in enumerate(medians) if m is not None and m < 0.5), None)
            shown = " ".join("  -  " if medians[b] is None else f"{medians[b]:+.2f}" for b in (1, 5, 10))
            print(f"  {feature:28s} {shown} | below 0.5 from {below} km")

        result = {
            "training": [int(times[0].year), int(times[-1].year)], "timestamps": int(len(array)), "cells": int(n_cells),
            "change_24h": {"features": names, "share": share.tolist(), "mean_square_minmax": mean_square.tolist()},
            "cycles": {"features": list(CYCLE_FEATURES), "hours": list(range(0, 24, 3)), "cells": cycles},
            "correlation": {"features": list(CORRELATION_FEATURES), "bin_km": BIN_KM, "edges_km": edges.tolist(),
                            "cells": int(len(corr_cells)), "timestamp_stride": args.correlation_stride,
                            "bins": correlation},
        }
        args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"\nwritten {args.output}")
    finally:
        cubes.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
