"""Download the dust of the CAMS reanalysis every 3 hours, on a 0.5 degree grid.

The reanalysis (EAC4) has a field every 3 hours from 2003. The ADS serves it on a regular
0.75 x 0.75 degree grid, so the 0.5 degree fields saved here are a bilinear interpolation of
those: they sit on the cells of the ERA5 domain but hold no detail finer than 0.75 degrees. The
fields as downloaded are saved next to them, so that nothing has to be downloaded again to
interpolate them in another way.

One request per year. `--workers` years are asked for at the same time, each with its own
client; the ADS decides how many of them it runs at once and queues the others. A year already
saved is not asked for again, and a year that fails is reported at the end without stopping the
others: run the command again to retry it.

Each eac4_<year>.npz holds `dust` [time, latitude, longitude] with `times`, `latitudes` (north to
south) and `longitudes`, and `native_dust` with `native_latitudes` and `native_longitudes`.

    python scripts/download_cams_dust.py --output-dir data/cams_dust_3h
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import sys
import tempfile
import threading
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts.check_era5_dust_lead import DATA_FORMATS, DATASET, VARIABLE, create_client

AREA = (60.0, -15.0, 20.0, 50.0)  # north, west, south, east: the ERA5 domain
NATIVE_STEP = 0.75  # degrees of the grid the ADS serves
HOURS = tuple(range(0, 24, 3))
_files = threading.Lock()  # one downloaded file is read at a time


def target_axes(area, step):
    """Latitudes (north to south) and longitudes (west to east) of the grid of `step` degrees over `area`."""
    north, west, south, east = area
    count = lambda first, last: int(round((last - first) / step)) + 1
    latitudes, longitudes = north - step * np.arange(count(south, north)), west + step * np.arange(count(west, east))
    if min(len(latitudes), len(longitudes)) < 2 or abs(latitudes[-1] - south) > 1e-6 or abs(longitudes[-1] - east) > 1e-6:
        raise ValueError("The area must be a whole number of grid steps, north above south and east beyond west")
    return latitudes, longitudes


def request_area(area):
    """North, west, south, east asked of the ADS: the lines of its grid just outside `area` on every side."""
    outside = lambda value, sign: float(sign * NATIVE_STEP * (np.floor(sign * value / NATIVE_STEP + 1e-9) + 1))
    north, west, south, east = area
    return [outside(north, 1), outside(west, -1), outside(south, -1), outside(east, 1)]


def read_fields(path):
    """Fields of one downloaded file in time order: (values [instants, lat, lon], times, latitudes, longitudes).

    The ADS delivers a NetCDF or a zip holding one, with the time either as a `time` dimension or
    as `valid_time` over (forecast_period, forecast_reference_time).
    """
    import xarray as xr
    with tempfile.TemporaryDirectory() as temp:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                path = archive.extract(next(name for name in archive.namelist() if name.endswith(".nc")), temp)
        with xr.open_dataset(path) as dataset:
            field = dataset[next(name for name in dataset.data_vars if dataset[name].ndim >= 3)]
            time_name = "valid_time" if "valid_time" in field.coords else "time"
            field = field.stack(instant=[name for name in field.dims if name not in ("latitude", "longitude")])
            field = field.transpose("instant", "latitude", "longitude")
            times = pd.DatetimeIndex(np.asarray(field[time_name].values).reshape(-1))
            values = np.asarray(field.values, dtype=np.float32)
            latitudes = np.asarray(dataset["latitude"].values, dtype=float)
            longitudes = (np.asarray(dataset["longitude"].values, dtype=float) + 180) % 360 - 180
    usable = np.flatnonzero(~times.isna() & np.isfinite(values).all(axis=(1, 2)))
    order = usable[np.argsort(times[usable].asi8, kind="stable")]
    return values[order], times[order], latitudes, longitudes


def to_grid(values, latitudes, longitudes, target_latitudes, target_longitudes):
    """Fields [instants, lat, lon] on the target axes, bilinear between the four points around each target point."""
    def weights(axis, target):
        order = np.argsort(axis)
        axis, target = np.asarray(axis, dtype=float)[order], np.asarray(target, dtype=float)
        if target.min() < axis[0] - 1e-9 or target.max() > axis[-1] + 1e-9:
            raise ValueError("The downloaded fields do not cover the grid asked for")
        low = np.clip(np.searchsorted(axis, target, side="right") - 1, 0, len(axis) - 2)
        return order[low], order[low + 1], np.clip((target - axis[low]) / (axis[low + 1] - axis[low]), 0, 1)

    first_lat, second_lat, along_lat = weights(latitudes, target_latitudes)
    first_lon, second_lon, along_lon = weights(longitudes, target_longitudes)
    rows = values[:, first_lat] * (1 - along_lat)[None, :, None] + values[:, second_lat] * along_lat[None, :, None]
    return (rows[:, :, first_lon] * (1 - along_lon) + rows[:, :, second_lon] * along_lon).astype(np.float32)


def download_year(directory, year, *, area=AREA, step=0.5):
    """Download one year and save eac4_<year>.npz; nothing is left behind if it fails."""
    directory = Path(directory)
    target, raw, partial = (directory / name for name in (f"eac4_{year}.npz", f"raw_{year}.tmp", f"eac4_{year}.partial"))
    latitudes, longitudes = target_axes(area, step)
    client = create_client()
    try:
        for data_format in DATA_FORMATS:
            request = {"variable": [VARIABLE], "date": [f"{year}-01-01/{year}-12-31"],
                       "time": [f"{hour:02d}:00" for hour in HOURS], "data_format": data_format,
                       "area": request_area(area)}
            try:
                client.retrieve(DATASET, request, str(raw))
            except Exception as error:
                print(f"[cams] year={year} data_format={data_format} refused: {str(error)[:400]}", flush=True)
                continue
            break
        else:
            raise RuntimeError(f"Dust download failed for {year}")
        with _files:
            values, times, native_latitudes, native_longitudes = read_fields(raw)
            expected = pd.date_range(f"{year}-01-01", f"{year}-12-31 21:00", freq=f"{HOURS[1]}h")
            if len(times) != len(expected) or (times.as_unit("ns").asi8 != expected.as_unit("ns").asi8).any():
                raise ValueError(f"{year}: {len(times)} fields instead of one every {HOURS[1]} hours ({len(expected)})")
            dust = to_grid(values, native_latitudes, native_longitudes, latitudes, longitudes)
            with open(partial, "wb") as handle:
                np.savez(handle, dust=dust, times=expected.values.astype("datetime64[h]"), latitudes=latitudes,
                         longitudes=longitudes, native_dust=values, native_latitudes=native_latitudes,
                         native_longitudes=native_longitudes)
            os.replace(partial, target)
    finally:
        raw.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    print(f"[cams] year={year} saved: fields={len(dust)} grid={len(latitudes)}x{len(longitudes)} "
          f"from {len(native_latitudes)}x{len(native_longitudes)} mean_aod={dust.mean():.4f}", flush=True)


def download(directory, years, *, workers=8, area=AREA, step=0.5):
    """Save eac4_<year>.npz for every year not already in `directory`; return the years that failed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    target_axes(area, step)  # a wrong area is refused before anything is asked for
    missing = [year for year in years if not (directory / f"eac4_{year}.npz").exists()]
    for year in years:
        if year not in missing:
            print(f"[cams] year={year} already saved", flush=True)
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_year, directory, year, area=area, step=step): year for year in missing}
        for future in as_completed(futures):
            error = future.exception()
            if error is not None:
                failed.append(futures[future])
                print(f"[cams] year={futures[future]} failed: {str(error)[:400]}", flush=True)
    return sorted(failed)


def load_fields(directory):
    """All saved years as (dust [times, lat, lon], times, latitudes, longitudes)."""
    files = sorted(Path(directory).glob("eac4_*.npz"))
    if not files:
        raise ValueError(f"No eac4_<year>.npz in {directory}")
    years = [np.load(file) for file in files]
    times = pd.DatetimeIndex(np.concatenate([year["times"] for year in years]))
    values = np.empty((len(times), *years[0]["dust"].shape[1:]), dtype=np.float32)
    start = 0
    for year in years:  # one year at a time: the years together are several gigabytes
        field = year["dust"]
        values[start:start + len(field)] = field
        start += len(field)
    return values, times, years[0]["latitudes"], years[0]["longitudes"]


def parser():
    root = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    root.add_argument("--output-dir", type=Path, required=True, help="Where the yearly files are kept")
    root.add_argument("--first-year", type=int, default=2003)
    root.add_argument("--last-year", type=int, default=2025)
    root.add_argument("--workers", type=int, default=8, help="Years asked for at the same time")
    root.add_argument("--area", type=float, nargs=4, default=list(AREA), metavar=("NORTH", "WEST", "SOUTH", "EAST"))
    root.add_argument("--grid", type=float, default=0.5, help="Degrees of the grid the fields are interpolated to")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    years = list(range(args.first_year, args.last_year + 1))
    failed = download(args.output_dir, years, workers=args.workers, area=tuple(args.area), step=args.grid)
    print(f"[cams] saved {len(years) - len(failed)} of {len(years)} years in {args.output_dir}"
          + (f" | failed: {' '.join(map(str, failed))} (run the command again to retry them)" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
