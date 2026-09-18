"""Read already-downloaded ERA5 monthly NetCDFs into bounded-memory model cubes."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .cube import CubeGrid
from .features import FEATURE_NAMES, SINGLE, PRESSURE
from ..anomaly_detection.stgan.data import AlignedCubes


def month_files(root, year, month, pressure=False):
    directory, prefix = ("pressure_850", "era5_850") if pressure else ("single_levels", "era5_single")
    target = Path(root)/directory/str(year)/f"{prefix}_{year}_{month:02d}.nc"
    receipt = target.with_suffix(".json")
    if receipt.is_file():
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        paths = []
        for entry in saved["files"]:
            name = entry["name"]
            if Path(name).name != name or not (name == target.name or name.startswith(target.stem+"__")):
                raise ValueError("Unsafe ERA5 receipt path")
            paths.append(target.parent/name)
    elif target.is_file():
        paths = [target]
    else:
        paths = sorted(target.parent.glob(target.stem+"__*.nc"))
    if not paths or not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"Missing ERA5 month: {target}")
    return paths


def latest_local_year(root, start=2005):
    """Latest consecutive complete calendar year present locally, no network."""
    years = [int(p.name) for p in (Path(root)/"single_levels").glob("*") if p.is_dir() and p.name.isdigit()]
    latest = start-1
    for year in range(start, min(max(years, default=start-1), pd.Timestamp.now(tz="UTC").year-1)+1):
        try:
            for month in range(1, 13):
                for pressure in (False, True):
                    month_files(root, year, month, pressure)
        except FileNotFoundError:
            break
        latest = year
    if latest < start:
        raise ValueError(f"No complete contiguous scoring year from {start} in local ERA5 files")
    return latest


def monthly_blocks(root, year, month, *, area=(60,-15,20,50), chunk_size=32):
    """Yield (timestamp, H,W,F) blocks, preserving tp/ssrd units and valid times."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    north, west, south, east = area
    lat = np.arange(round(north*2), round(south*2)-1, -1)/2
    lon = np.arange(round(west*2), round(east*2)+1)/2
    expected = pd.date_range(f"{year}-{month:02d}-01", periods=pd.Period(f"{year}-{month:02d}").days_in_month*8, freq="3h").as_unit("ns")
    with ExitStack() as stack:
        fields = {}
        for pressure, variables in ((False, SINGLE), (True, PRESSURE)):
            for path in month_files(root, year, month, pressure):
                ds = stack.enter_context(xr.open_dataset(path, engine="netcdf4", cache=False))
                time_name = "valid_time" if "valid_time" in ds.coords else "time"
                times = pd.DatetimeIndex(ds[time_name].values).as_unit("ns")
                if not times.equals(expected):
                    raise ValueError(f"Wrong/incomplete 3h valid times in {path}")
                if time_name != "time":
                    ds = ds.rename({time_name: "time"})
                ds = ds.assign_coords(longitude=(ds.longitude+180)%360-180).sortby("longitude").sortby("latitude", ascending=False)
                if ds.latitude.shape != lat.shape or ds.longitude.shape != lon.shape or not np.allclose(ds.latitude, lat) or not np.allclose(ds.longitude, lon):
                    raise ValueError(f"Wrong 0.5 degree area in {path}")
                if pressure:
                    level = next((n for n in ("pressure_level", "level", "isobaricInhPa") if n in ds.coords), None)
                    if level is None or ds[level].size != 1 or float(ds[level].values.reshape(-1)[0]) != 850:
                        raise ValueError(f"Require exclusively 850 hPa in {path}")
                for feature, aliases in variables.items():
                    name = next((name for name in (*aliases, feature) if name in ds.data_vars), None)
                    if name is None:
                        continue
                    if feature in fields:
                        raise ValueError(f"Duplicate ERA5 variable {feature}")
                    field = ds[name]
                    for dim in set(field.dims)-{"time","latitude","longitude"}:
                        if field.sizes[dim] != 1:
                            raise ValueError(f"Unresolved {dim} in {path}; do not silently mix expver or pressure levels")
                        field = field.isel({dim: 0}, drop=True)
                    fields[feature] = field.transpose("time", "latitude", "longitude")
        if set(fields) != set(FEATURE_NAMES):
            raise ValueError(f"Missing ERA5 features: {set(FEATURE_NAMES)-set(fields)}")
        for start in range(0, len(expected), chunk_size):
            yield expected[start:start+chunk_size], np.stack([
                fields[name].isel(time=slice(start,start+chunk_size)).values for name in FEATURE_NAMES
            ], axis=-1).astype(np.float32)


def prepare_era5(root, cache_dir, *, start_year=1980, train_end_year=2004,
                 score_end_year="latest", area=(60,-15,20,50), chunk_size=32,
                 missing_policy="static_mask"):
    """Disk cache; no feature conversion/imputation or full in-memory concatenation.

    Static missing sites (typically soil water over sea) are excluded according
    to the FIRST TRAINING frame only. Any change of this complete-case mask at a
    later time fails. This avoids test-derived selection or silent imputation.
    `missing_policy='error'` requires all 15 features everywhere.
    """
    score_start = train_end_year+1
    end = latest_local_year(root, score_start) if score_end_year == "latest" else int(score_end_year)
    if not 1940 <= start_year <= train_end_year < end or missing_policy not in ("static_mask", "error"):
        raise ValueError("Invalid years or missing policy")
    north, west, south, east = area
    if not (-90 <= south < north <= 90 and -180 <= west < east <= 180) or any(x*2 != round(x*2) for x in area):
        raise ValueError("Require a 0.5 degree aligned non-wrapping area")
    output = Path(cache_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty prepared data directory")
    # Preflight all monthly paths before allocating multi-GB caches.
    for year in range(start_year, end+1):
        for month in range(1,13):
            for pressure in (False,True):
                month_files(root, year, month, pressure)
    output.mkdir(parents=True, exist_ok=True)
    times = {"train": pd.date_range(f"{start_year}-01-01", f"{train_end_year}-12-31 21:00", freq="3h"),
             "test": pd.date_range(f"{score_start}-01-01", f"{end}-12-31 21:00", freq="3h")}
    times = {name: value.as_unit("ns") for name, value in times.items()}
    arrays, offsets, valid, grid = {}, {"train":0,"test":0}, None, None
    try:
        for year in range(start_year,end+1):
            split = "train" if year <= train_end_year else "test"
            for month in range(1,13):
                for block_times, block in monthly_blocks(root,year,month,area=area,chunk_size=chunk_size):
                    complete = np.isfinite(block).all(axis=-1)
                    if valid is None:
                        valid = complete[0]
                        if not valid.any() or (missing_policy == "error" and not valid.all()):
                            raise ValueError("ERA5 has missing features; choose static_mask explicitly or supply complete data")
                        rows, cols = np.nonzero(valid)
                        grid = CubeGrid(north-np.arange(valid.shape[0])*.5,
                                        west+np.arange(valid.shape[1])*.5, rows, cols)
                        for name in times:
                            arrays[name] = np.lib.format.open_memmap(output/(name+".npy"), mode="w+",
                                dtype=np.float32, shape=(len(times[name]), len(rows), len(FEATURE_NAMES)))
                    if not np.array_equal(complete, np.broadcast_to(valid, complete.shape)):
                        raise ValueError(f"Time-varying missing features at {block_times[0]}; no imputation applied")
                    offset = offsets[split]
                    if not times[split][offset:offset+len(block)].equals(block_times):
                        raise ValueError("ERA5 split alignment failed")
                    arrays[split][offset:offset+len(block)] = block[:,valid,:]
                    offsets[split] += len(block)
        for name, array in arrays.items():
            if offsets[name] != len(times[name]):
                raise ValueError("Incomplete ERA5 split")
            array.flush()
            np.save(output/(name+"_timestamps.npy"), times[name].asi8)
        metadata = {"status":"complete", "grid":grid.to_dict(), "features":list(FEATURE_NAMES),
                    "start_year":start_year,"train_end_year":train_end_year,"score_end_year":end,
                    "timestep_hours":3,"trend_steps_for_168_hours":56,
                    "missing_policy":missing_policy,"n_excluded_cells":int((~valid).sum()),
                    "feature_transform":"none; tp/ssrd sampled one-hour accumulations retained"}
        (output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    finally:
        for array in arrays.values():
            array._mmap.close()
    return load_prepared(output)


def load_prepared(directory):
    directory = Path(directory)
    metadata = json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or tuple(metadata["features"]) != FEATURE_NAMES:
        raise ValueError("Incomplete/incompatible ERA5 cache")
    grid = CubeGrid(**metadata["grid"])
    cubes = AlignedCubes(
        train=np.load(directory/"train.npy",mmap_mode="r"), test=np.load(directory/"test.npy",mmap_mode="r"),
        train_timestamps=pd.DatetimeIndex(np.load(directory/"train_timestamps.npy")),
        test_timestamps=pd.DatetimeIndex(np.load(directory/"test_timestamps.npy")),
        location_names=tuple(f"r{r}_c{c}" for r,c in zip(grid.rows,grid.cols)), feature_names=FEATURE_NAMES,
        latitudes=grid.latitudes[grid.rows], longitudes=grid.longitudes[grid.cols])
    return cubes, grid, metadata
