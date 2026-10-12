"""How far are the scored years from the climate of the training years?

No model is used. For every cell and variable, the mean and the standard deviation of the
training years by calendar month and time of day are the normal. Every value of every year is
then its distance from that normal in standard deviations, and the mean of those distances by
year says how much the climate has moved: it is 0 over the training years as a whole, by
construction, and it stays near 0 in the scored years only if they have not drifted. A detector
that standardises its inputs with the training normal reads such a drift as an anomaly of every
day.

The part of the seasonal cycle inside a calendar month counts as spread, so the distances are a
little smaller than those of a normal by day of the year.

One timestamp every `--time-stride` is read (an odd number, so that every time of day is read)
and one cell every `--cell-stride`.

    python scripts/check_era5_climate_drift.py --prepared-dir data/era5_prepared
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from physiq_pv.era5.data import load_prepared

FEATURES = ("2m_temperature", "temperature", "specific_humidity", "mean_sea_level_pressure", "total_precipitation",
            "volumetric_soil_water_layer_1", "surface_solar_radiation_downwards", "total_cloud_cover")
SEASONS = {"all months": range(1, 13), "June to August": (6, 7, 8), "December to February": (12, 1, 2)}
STD_FLOOR = 0.01  # share of the overall standard deviation of a variable below which a bin's own is not taken
WIDTH = 13


def sample(array, timestamps, time_stride, cells, columns):
    """Values [timestamps, cells, variables] of one timestamp every `time_stride`, and those timestamps."""
    positions = np.arange(0, len(array), time_stride)
    values = np.empty((len(positions), len(cells), len(columns)), dtype=np.float32)
    for index, position in enumerate(positions):
        values[index] = np.asarray(array[position:position + 1])[0][np.ix_(cells, columns)]
    return values, timestamps[positions]


def bins(timestamps):
    """Calendar month and time of day of every timestamp, as one number."""
    return (timestamps.month.to_numpy() - 1) * 24 + timestamps.hour.to_numpy()


def fit_normal(values, timestamps):
    """Per calendar month and time of day: mean and standard deviation [cells, variables] of the training values."""
    floor = STD_FLOOR * values.std(axis=(0, 1), dtype=np.float64)
    floor[floor <= 0] = 1.0
    keys = bins(timestamps)
    return {key: (values[keys == key].mean(axis=0, dtype=np.float64),
                  np.maximum(values[keys == key].std(axis=0, dtype=np.float64), floor)) for key in np.unique(keys)}


def distances(values, timestamps, normal):
    """Mean over the cells of the distance from the normal, in standard deviations [timestamps, variables]."""
    keys = bins(timestamps)
    result = np.empty((len(values), values.shape[2]))
    for key in np.unique(keys):
        if key not in normal:
            raise ValueError("A month or a time of day of the scored years is not in the training sample")
        mean, std = normal[key]
        result[keys == key] = ((values[keys == key] - mean) / std).mean(axis=1)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--features", nargs="+", default=list(FEATURES))
    parser.add_argument("--time-stride", type=int, default=9)
    parser.add_argument("--cell-stride", type=int, default=5)
    args = parser.parse_args(argv)
    if args.time_stride < 1 or args.time_stride % 2 == 0 or args.cell_stride < 1:
        raise ValueError("time-stride must be odd and both strides at least 1")
    cubes = load_prepared(args.prepared_dir)[0]
    try:
        unknown = [name for name in args.features if name not in cubes.feature_names]
        if unknown:
            raise ValueError(f"Unknown variables: {unknown}; the cache has {list(cubes.feature_names)}")
        columns = [cubes.feature_names.index(name) for name in args.features]
        cells = np.arange(0, cubes.train.shape[1], args.cell_stride)
        train, train_times = sample(cubes.train, cubes.train_timestamps, args.time_stride, cells, columns)
        normal = fit_normal(train, train_times)
        parts = [pd.DataFrame(distances(train, train_times, normal), index=train_times, columns=args.features)]
        del train
        for array, timestamps in ((cubes.validation, cubes.validation_timestamps), (cubes.test, cubes.test_timestamps)):
            values, times = sample(array, timestamps, args.time_stride, cells, columns)
            parts.append(pd.DataFrame(distances(values, times, normal), index=times, columns=args.features))
    finally:
        cubes.close()
    table = pd.concat(parts)
    last_training_year = int(train_times[-1].year)
    print(f"[drift] normal: {train_times[0].year}-{last_training_year}, by calendar month and time of day | cells "
          f"{len(cells)} | mean distance from the normal in its standard deviations, by year")
    for season, months in SEASONS.items():
        chosen = table[table.index.month.isin(list(months))]
        by_year = chosen.groupby(chosen.index.year).mean()
        print(f"\n{season}\n{'year':<6}" + "".join(f"{name[:WIDTH - 1]:>{WIDTH}}" for name in args.features))
        for year, row in by_year.iterrows():
            print(f"{year:<6}" + "".join(f"{value:>+{WIDTH}.2f}" for value in row)
                  + ("" if year <= last_training_year else "  scored"))
        late = by_year[by_year.index > last_training_year]
        print(f"{'last 5':<6}" + "".join(f"{value:>+{WIDTH}.2f}" for value in late.tail(5).mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
