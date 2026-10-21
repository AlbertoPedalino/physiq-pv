"""Where does the step of the total precipitation in 2000 come from?

The drift check finds the total precipitation near 0 standard deviations from the training normal
over 1980-1999 and between +0.1 and +0.26 in every year from 2000. This check reads where the
step is, with no model:

- by year, in the values themselves (prepared cache, metres in one hour): the mean, the share of
  exact zeros, of tiny values (above 0 and below 0.01 mm) and of wet hours (at least 0.1 mm), the
  smallest positive value and the 99th percentile of the wet hours;
- by year, the mean distance from the training normal (as in the drift check) in the driest
  quarter of the cells and in the others: a step made of tiny values weighs most where the normal
  is almost always 0 and its standard deviation is small;
- by month around the step, to date it, and by time of day, before and after it;
- in the raw monthly files of the months given (by default December 1999 and January 2000): how
  the precipitation is stored (type, scale, offset, smallest positive value, share of zeros), and
  whether the prepared cache holds exactly the values of the files.

    python scripts/check_era5_precipitation_step.py --prepared-dir data/era5_prepared \
        --raw-dir data/era5 --train-end-year 1999
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import xarray as xr

from physiq_pv.era5.data import load_prepared, month_files, monthly_blocks

FEATURE = "total_precipitation"
TINY = 1e-5  # m in one hour: 0.01 mm
WET = 1e-4  # m in one hour: 0.1 mm
STD_FLOOR = 0.01  # as in the drift check
AROUND = (1998, 2001)  # years of the monthly table


def sample(array, timestamps, column, time_stride, cells):
    """Precipitation [timestamps, cells] of one timestamp every `time_stride`, and those timestamps."""
    positions = np.arange(0, len(array), time_stride)
    values = np.empty((len(positions), len(cells)), dtype=np.float32)
    for index, position in enumerate(positions):
        values[index] = np.asarray(array[position:position + 1])[0][cells, column]
    return values, timestamps[positions]


def bins(timestamps):
    return (timestamps.month.to_numpy() - 1) * 24 + timestamps.hour.to_numpy()


def distances(values, timestamps, train, train_times):
    """Distance from the training normal by calendar month and time of day, in its standard deviations."""
    floor = STD_FLOOR * train.std(dtype=np.float64)
    out = np.empty(values.shape, dtype=np.float64)
    train_keys, keys = bins(train_times), bins(timestamps)
    for key in np.unique(keys):
        reference = train[train_keys == key]
        mean, std = reference.mean(axis=0, dtype=np.float64), np.maximum(reference.std(axis=0, dtype=np.float64), floor)
        out[keys == key] = (values[keys == key] - mean) / std
    return out


def value_table(values, timestamps, last_training_year):
    print("\nvalues by year (all sampled cells; mm in one hour)")
    print(f"{'year':>6} {'mean':>8} {'zero %':>8} {'tiny %':>8} {'wet %':>8} {'min>0':>10} {'p99 wet':>8}")
    for year in np.unique(timestamps.year):
        v = values[timestamps.year == year].ravel()
        positive = v[v > 0]
        wet = v[v >= WET]
        print(f"{year:>6} {v.mean() * 1000:>8.4f} {100 * (v == 0).mean():>8.2f} "
              f"{100 * ((v > 0) & (v < TINY)).mean():>8.2f} {100 * (v >= WET).mean():>8.2f} "
              f"{(positive.min() * 1000 if positive.size else np.nan):>10.2e} "
              f"{(np.percentile(wet, 99) * 1000 if wet.size else np.nan):>8.3f}"
              + ("" if year <= last_training_year else "  scored"))


def distance_table(distance, timestamps, dry, last_training_year):
    print("\nmean distance from the training normal by year: driest quarter of the cells | the others")
    for year in np.unique(timestamps.year):
        d = distance[timestamps.year == year]
        print(f"{year:>6} {d[:, dry].mean():>+8.2f} {d[:, ~dry].mean():>+8.2f}"
              + ("" if year <= last_training_year else "  scored"))


def month_table(distance, values, timestamps):
    first, last = AROUND
    print(f"\nby month {first}-{last}: mean distance (all cells) | share of tiny values %")
    inside = (timestamps.year >= first) & (timestamps.year <= last)
    frame = pd.DataFrame({"distance": distance[inside].mean(axis=1),
                          "tiny": 100 * ((values[inside] > 0) & (values[inside] < TINY)).mean(axis=1)},
                         index=timestamps[inside])
    for (year, month), group in frame.groupby([frame.index.year, frame.index.month]):
        print(f"{year}-{month:02d} {group['distance'].mean():>+8.2f} {group['tiny'].mean():>8.2f}")


def hour_table(distance, timestamps, last_training_year):
    print("\nby time of day (UTC): mean distance, last 5 training years | first 5 years after them")
    before = (timestamps.year > last_training_year - 5) & (timestamps.year <= last_training_year)
    after = (timestamps.year > last_training_year) & (timestamps.year <= last_training_year + 5)
    for hour in range(0, 24, 3):
        at = timestamps.hour == hour
        print(f"{hour:>02d}:00 {distance[before & at].mean():>+8.2f} {distance[after & at].mean():>+8.2f}")


def raw_check(raw_dir, months, cubes, grid, column):
    print("\nraw monthly files")
    partitions = ((cubes.train, cubes.train_timestamps), (cubes.validation, cubes.validation_timestamps),
                  (cubes.test, cubes.test_timestamps))
    for text in months:
        year, month = (int(part) for part in text.split("-"))
        for path in month_files(raw_dir, year, month):
            with xr.open_dataset(path, engine="netcdf4", cache=False) as ds:
                name = next((n for n in ("tp", FEATURE) if n in ds.data_vars), None)
                if name is None:
                    continue
                encoding = ds[name].encoding
                v = ds[name].values.ravel()
                positive = v[v > 0]
                print(f"  {path.name}: stored {encoding.get('dtype')}, scale {encoding.get('scale_factor')}, "
                      f"offset {encoding.get('add_offset')}, fill {encoding.get('_FillValue')}, "
                      f"format {ds.encoding.get('format', '?')} | zero {100 * (v == 0).mean():.2f}%, "
                      f"tiny {100 * ((v > 0) & (v < TINY)).mean():.2f}%, "
                      f"min>0 {(positive.min() * 1000 if positive.size else np.nan):.2e} mm, "
                      f"negative {100 * (v < 0).mean():.2f}%")
                attributes = {k: str(val)[:80] for k, val in ds.attrs.items() if k.lower() in ("history", "institution", "conventions", "source")}
                if attributes:
                    print(f"    {attributes}")
        # Prepared cache against the raw files, over the cells it keeps.
        largest = 0.0
        for block_times, block in monthly_blocks(raw_dir, year, month):
            raw = block[:, grid.rows, grid.cols, column]
            for array, times in partitions:
                where = times.get_indexer(block_times)
                if (where >= 0).all():
                    prepared = np.stack([np.asarray(array[int(i):int(i) + 1])[0][:, column] for i in where])
                    largest = max(largest, float(np.nanmax(np.abs(prepared - raw))))
                    break
        print(f"  {year}-{month:02d}: largest difference raw - prepared {largest:.3e} m")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=None, help="Monthly NetCDF root; without it the raw check is skipped")
    parser.add_argument("--raw-months", nargs="+", default=["1999-12", "2000-01"])
    parser.add_argument("--train-end-year", type=int, default=None)
    parser.add_argument("--time-stride", type=int, default=9)
    parser.add_argument("--cell-stride", type=int, default=5)
    args = parser.parse_args(argv)
    cubes, grid, _ = load_prepared(args.prepared_dir, args.train_end_year)
    try:
        column = cubes.feature_names.index(FEATURE)
        cells = np.arange(0, cubes.train.shape[1], args.cell_stride)
        train, train_times = sample(cubes.train, cubes.train_timestamps, column, args.time_stride, cells)
        last_training_year = int(train_times[-1].year)
        parts = [(train, train_times)] + [sample(array, times, column, args.time_stride, cells)
                                          for array, times in ((cubes.validation, cubes.validation_timestamps),
                                                               (cubes.test, cubes.test_timestamps))]
        values = np.concatenate([part[0] for part in parts])
        timestamps = parts[0][1].append([part[1] for part in parts[1:]])
        print(f"[precipitation-step] training {train_times[0].year}-{last_training_year} | {len(cells)} cells, "
              f"one timestamp every {args.time_stride}")
        value_table(values, timestamps, last_training_year)
        distance = distances(values, timestamps, train, train_times)
        dry = train.mean(axis=0) <= np.quantile(train.mean(axis=0), 0.25)
        distance_table(distance, timestamps, dry, last_training_year)
        month_table(distance, values, timestamps)
        hour_table(distance, timestamps, last_training_year)
        if args.raw_dir is not None:
            raw_check(args.raw_dir, args.raw_months, cubes, grid, column)
    finally:
        cubes.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
