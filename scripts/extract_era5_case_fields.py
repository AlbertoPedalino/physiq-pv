"""Raw ERA5 fields of a few timestamps over the whole domain, for the maps of the thesis cases.

The maps of the three cases of Section 6.4.1 show standardised values. This script reads, at the timestamps of those
maps, the values that the standardisation hides: by default the mean sea-level pressure (for isobars) and the total
precipitation (to mask the dry cells, where the fourth root of zero standardises to a negative value). It reads the
test cube of a prepared ERA5 directory ([timestamps, cells, features], physical units) and keeps the cells in the order
of the grid of a scored run, so that the output lines up with the case files (rows, cols of the grid).

Usage (on the server):
    python scripts/extract_era5_case_fields.py --run outputs/stgan_optuna27_scored --prepared-dir data/era5_prepared \
        --output outputs/casi_campi_grezzi.npz

Standalone on purpose (numpy and pandas only): it can be run from a `git archive` of this one file.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CASE_TIMESTAMPS = ("2017-11-23 00:00", "2006-05-19 03:00", "2024-06-15 12:00")  # the maps of Figures 6.7, 6.8, 6.9
FEATURES = ("mean_sea_level_pressure", "total_precipitation")


def case_fields(run_dir, prepared_dir, timestamps, features):
    """Arrays of the output: the grid of the run, the timestamps, the features and values [timestamps, cells, features]."""
    run_dir, prepared_dir = Path(run_dir), Path(prepared_dir)
    grid = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    names = list(json.loads((prepared_dir / "metadata.json").read_text(encoding="utf-8"))["features"])
    missing = [name for name in features if name not in names]
    if missing:
        raise ValueError(f"Features not in the prepared directory: {missing}")
    columns = [names.index(name) for name in features]
    prepared_times = pd.DatetimeIndex(np.load(prepared_dir / "test_timestamps.npy"))
    wanted = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
    positions = prepared_times.get_indexer(wanted)
    if (positions < 0).any():
        raise ValueError(f"Timestamps not in the prepared test cube: {list(wanted[positions < 0].astype(str))}")
    cube = np.load(prepared_dir / "test.npy", mmap_mode="r")
    try:
        if cube.shape[1] != len(rows):
            raise ValueError(f"The prepared cube has {cube.shape[1]} cells, the grid of the run {len(rows)}")
        values = np.stack([np.asarray(cube[position][:, columns], dtype=np.float32) for position in positions])
    finally:
        cube._mmap.close()
    return {"rows": rows, "cols": cols, "latitudes": np.asarray(grid["latitudes"]),
            "longitudes": np.asarray(grid["longitudes"]), "timestamps": wanted.as_unit("ns").asi8,
            "features": np.asarray(features), "values": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="Scored run: its metadata.json gives the grid")
    parser.add_argument("--prepared-dir", type=Path, required=True, help="Prepared ERA5 directory (test.npy)")
    parser.add_argument("--timestamps", nargs="+", default=list(CASE_TIMESTAMPS))
    parser.add_argument("--features", nargs="+", default=list(FEATURES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays = case_fields(args.run, args.prepared_dir, args.timestamps, args.features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"[fields] {args.output}: {len(arrays['timestamps'])} timestamps x {arrays['values'].shape[1]} cells")
    for k, moment in enumerate(pd.to_datetime(arrays["timestamps"])):
        for j, name in enumerate(arrays["features"]):
            column = arrays["values"][k, :, j]
            print(f"  {moment}  {name}: min {np.nanmin(column):.6g}, median {np.nanmedian(column):.6g}, "
                  f"max {np.nanmax(column):.6g}, zeros {100 * np.mean(column == 0):.1f}%")


if __name__ == "__main__":
    main()
