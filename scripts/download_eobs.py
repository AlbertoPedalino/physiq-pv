"""Daily precipitation measured by stations (E-OBS), on the cells of an ERA5 run.

E-OBS is a gridded data set of daily weather over European land, built from station observations
and not from ERA5: a judge of the scores that saw none of it. This script downloads the ensemble
mean of the daily precipitation at 0.25 degrees from the Copernicus Climate Data Store
(`insitu-gridded-observations-europe`), keeps the raw files, and writes for every cell of the run
the mean of the E-OBS points inside it (four at 0.25 degrees in a cell of 0.5): one value per day
and cell, in mm, NaN where no point of the cell is on land or measured. The E-OBS day is a total
over 24 hours that for many countries runs from 06 UTC to 06 UTC of the next day.

What is written in `--output-dir`: `raw/` with the downloaded files, `rain.npy` [days, cells] float32,
`days.npy` (datetime64[D] as int64 days) and `metadata.json`. A file already downloaded is not asked
again; `--skip-download` only reads `raw/`.

Credentials: ~/.cdsapirc, with the licence of the data set accepted on the CDS website.

    python scripts/download_eobs.py --grid-dir outputs/run_clim --output-dir data/eobs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

DATASET = "insitu-gridded-observations-europe"
VARIABLE = "precipitation_amount"
NAME = "rr"  # the precipitation in the E-OBS files


def request(period, version, resolution):
    return {"product_type": "ensemble_mean", "variable": VARIABLE, "grid_resolution": resolution,
            "period": period, "version": version}


def download(raw_dir, periods, version, resolution):
    """The files of every period, under `raw_dir`; a zip from the CDS is unpacked there."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = None
    for period in periods:
        target = raw_dir / f"eobs_{VARIABLE}_{resolution}_{version}_{period}.download"
        if any(raw_dir.glob(f"eobs_{VARIABLE}_{resolution}_{version}_{period}*.nc")):
            print(f"[eobs] period={period} already downloaded", flush=True)
            continue
        if client is None:
            try:
                import cdsapi
                client = cdsapi.Client()
            except Exception:
                # Do not echo the error: it can contain configuration text.
                raise RuntimeError("Cannot initialise the CDS client. Check url and key in ~/.cdsapirc") from None
        print(f"[eobs] asking period={period} version={version} resolution={resolution}", flush=True)
        client.retrieve(DATASET, request(period, version, resolution), str(target))
        stem = target.with_suffix("")
        if zipfile.is_zipfile(target):
            with zipfile.ZipFile(target) as archive:
                names = [name for name in archive.namelist() if name.endswith(".nc")]
                if not names:
                    raise ValueError(f"The archive of {period} holds no NetCDF file")
                for index, name in enumerate(names):
                    (Path(f"{stem}{'' if len(names) == 1 else f'_{index}'}.nc")).write_bytes(archive.read(name))
            target.unlink()
        else:
            target.rename(stem.with_suffix(".nc"))
        print(f"[eobs] period={period} saved", flush=True)


def cell_points(axis, centres, spacing):
    """For every cell centre, the indices of the points of `axis` inside the cell (at most two at half the
    spacing), padded with -1."""
    inside = np.abs(axis[None, :] - centres[:, None]) < spacing / 2 - 1e-6
    count = inside.sum(axis=1)
    if count.max(initial=0) > 2:
        raise ValueError("More than two E-OBS points per cell on one axis: the grid is finer than expected")
    points = np.full((len(centres), 2), -1)
    for index, found in enumerate(inside):
        chosen = np.nonzero(found)[0]
        points[index, :len(chosen)] = chosen
    return points


def rain_on_cells(paths, latitudes, longitudes, spacing, first_day, last_day):
    """Daily E-OBS precipitation [days, cells] averaged over the points inside each cell, and the days."""
    import xarray as xr

    pieces, days = [], []
    for path in sorted(paths):
        with xr.open_dataset(path) as dataset:
            if NAME not in dataset:
                raise ValueError(f"{path.name} holds no '{NAME}'")
            field = dataset[NAME].transpose("time", "latitude", "longitude")
            time = pd.DatetimeIndex(field["time"].values).normalize()
            kept = (time >= first_day) & (time <= last_day)
            if not kept.any():
                continue
            rows = cell_points(field["latitude"].values.astype(float), latitudes, spacing)
            cols = cell_points(field["longitude"].values.astype(float), longitudes, spacing)
            values = field.values[kept]  # [days, lat, lon], NaN over the sea
            block = np.full((len(values), len(latitudes), 4), np.nan, dtype=np.float32)
            for slot, (i, j) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                usable = (rows[:, i] >= 0) & (cols[:, j] >= 0)
                block[:, usable, slot] = values[:, rows[usable, i], cols[usable, j]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pieces.append(np.nanmean(block, axis=2).astype(np.float32))
            days.append(time[kept])
    if not pieces:
        raise ValueError("No E-OBS day in the period asked")
    rain, days = np.concatenate(pieces), days[0].append(days[1:]) if len(days) > 1 else days[0]
    order = np.argsort(days.values, kind="stable")
    rain, days = rain[order], days[order]
    if days.has_duplicates:
        keep = ~days.duplicated(keep="last")  # periods that overlap: the later file wins
        rain, days = rain[keep], days[keep]
    return rain, days


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid-dir", type=Path, required=True, help="A run whose metadata.json has the grid of the cells")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--periods", nargs="+", default=["1995_2010", "2011_2025"])
    parser.add_argument("--version", default="33_0e")
    parser.add_argument("--resolution", default="0_25deg")
    parser.add_argument("--years", type=int, nargs=2, default=(2005, 2025), metavar=("FIRST", "LAST"))
    parser.add_argument("--skip-download", action="store_true", help="Only read the files already in raw/")
    args = parser.parse_args(argv)
    raw = args.output_dir / "raw"
    if not args.skip_download:
        download(raw, args.periods, args.version, args.resolution)
    paths = sorted(raw.glob("*.nc"))
    if not paths:
        raise ValueError(f"No E-OBS file in {raw}")
    grid = json.loads((args.grid_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"]), np.asarray(grid["cols"])
    latitudes = np.asarray(grid["latitudes"], dtype=float)[rows]
    longitudes = (np.asarray(grid["longitudes"], dtype=float)[cols] + 180) % 360 - 180
    spacing = float(grid.get("spacing", 0.5))
    rain, days = rain_on_cells(paths, latitudes, longitudes, spacing,
                               pd.Timestamp(f"{args.years[0]}-01-01"), pd.Timestamp(f"{args.years[1]}-12-31"))
    np.save(args.output_dir / "rain.npy", rain)
    np.save(args.output_dir / "days.npy", days.values.astype("datetime64[D]").astype(np.int64))
    measured = np.isfinite(rain).any(axis=0)
    metadata = {"dataset": DATASET, "variable": VARIABLE, "product_type": "ensemble_mean",
                "version": args.version, "resolution": args.resolution, "periods": args.periods,
                "files": [path.name for path in paths], "grid_dir": str(args.grid_dir),
                "first_day": str(days[0].date()), "last_day": str(days[-1].date()), "days": len(days),
                "cells": len(rows), "cells_measured": int(measured.sum()), "units": "mm per day",
                "day": "24 h total; for many countries 06 UTC to 06 UTC of the next day"}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[eobs] {days[0].date()} to {days[-1].date()}: {len(days)} days, {int(measured.sum())} of {len(rows)} "
          f"cells with stations ({100 * measured.mean():.1f}%), mean {np.nanmean(rain):.2f} mm/day", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
