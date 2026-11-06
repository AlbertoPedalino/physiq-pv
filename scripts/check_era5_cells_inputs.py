"""What the input and the error of one variable do, year by year, in the cells of some patches against the domain.

Written on 6 November 2026 for the striking days by intensity (list_era5_striking_days.py --rank-by intensity): small
patches, with the soil water first among the variables, in the same few places, with scores hundreds of times their
threshold, while the spread of the score climatology there is ordinary (check_era5_climatology_spread.py). Is the input
itself unusual there (out of the range of the training years, after a given year), or does the model fail on an
ordinary input?

For the cells inside the boxes of the patches of `--cases` (casi.csv of list_era5_striking_days.py; boxes that are the
same are read once) and for all the cells of the domain, year by year over the scored period, for `--variable`:
- the mean standardised input (distance from the seasonal normal of the training years, in its standard deviations);
- the share of the values out of the range of the training years, as the model sees them after its min-max
  ((value - minimum) / scale out of [0, 1], from the shared cache of the run);
- the mean squared reconstruction error of the generator for that variable (scores/feature_scores.npy of the run).
One timestamp every `--stride` is read.

    python scripts/check_era5_cells_inputs.py --run-dir outputs/stgan_optuna27_scored \
        --cases outputs/optuna27/giorni_eclatanti_intensita/casi.csv --variable volumetric_soil_water_layer_1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path.cwd()))  # after the copy next to the script: a copy run from the checkout imports itself
import numpy as np
import pandas as pd

from physiq_pv.anomaly_detection.stgan.shared_cache import MANIFEST
from scripts.check_era5_event_zones import inputs_of

CHUNK = 64


def places(cases, latitudes, longitudes):
    """One row per distinct box of the cases: its name and the indices of the cells inside it."""
    boxes = cases[["lat_min", "lat_max", "lon_min", "lon_max"]].drop_duplicates()
    rows = []
    for _, box in boxes.iterrows():
        inside = np.flatnonzero((latitudes >= box["lat_min"]) & (latitudes <= box["lat_max"])
                                & (longitudes >= box["lon_min"]) & (longitudes <= box["lon_max"]))
        ranks = cases[(cases[["lat_min", "lat_max", "lon_min", "lon_max"]] == box.values).all(axis=1)]["rango"]
        name = f"{box['lat_min']:g}-{box['lat_max']:g}N {box['lon_min']:g}-{box['lon_max']:g}E"
        rows.append((name, ",".join(str(rank) for rank in ranks), inside))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Scored run with scores/feature_scores.npy")
    parser.add_argument("--cases", type=Path, required=True, help="casi.csv of list_era5_striking_days.py")
    parser.add_argument("--variable", default="volumetric_soil_water_layer_1")
    parser.add_argument("--stride", type=int, default=8, help="One timestamp every this many (8: one a day)")
    args = parser.parse_args(argv)
    manifest = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    grid = manifest["grid"]
    latitudes = np.asarray(grid["latitudes"])[np.asarray(grid["rows"])]
    longitudes = np.asarray(grid["longitudes"])[np.asarray(grid["cols"])]
    timestamps = pd.DatetimeIndex(np.load(args.run_dir / "test_timestamps.npy"))
    inputs, first, names = inputs_of(args.run_dir, timestamps)
    built = json.loads((Path(manifest["backend"]["seasonal_normalization"]["mean_file"]).parent / MANIFEST)
                       .read_text(encoding="utf-8"))["built"]
    column = names.index(args.variable)
    minimum, scale = float(built["minimum"][column]), float(built["scale"][column])
    errors = np.load(args.run_dir / "scores" / "feature_scores.npy", mmap_mode="r")
    groups = [("dominio", "", np.arange(len(latitudes)))] + places(pd.read_csv(args.cases), latitudes, longitudes)
    read = np.arange(0, len(timestamps), args.stride)
    years = timestamps[read].year.to_numpy()
    sums = {(name, year): np.zeros(4) for name, _, _ in groups for year in np.unique(years)}
    try:
        for start in range(0, len(read), CHUNK):
            steps = read[start:start + CHUNK]
            value = np.asarray(inputs[first + steps][:, :, column], dtype=np.float64)
            error = np.asarray(errors[steps][:, :, column], dtype=np.float64)
            position = (value - minimum) / scale
            outside = (position < 0) | (position > 1)
            for name, _, cells in groups:
                for year in np.unique(years[start:start + CHUNK]):
                    at = years[start:start + CHUNK] == year
                    block = (slice(None), cells)
                    sums[name, year] += (value[at][block].sum(), outside[at][block].sum(),
                                         np.nansum(error[at][block]), at.sum() * len(cells))
    finally:
        inputs._mmap.close()
        errors._mmap.close()
    print(f"[cells] {args.variable}: training range as the model sees it, minimum {minimum:.4g}, scale {scale:.4g} "
          f"(standardised units); one timestamp every {args.stride}")
    for name, ranks, cells in groups:
        table = pd.DataFrame([{"anno": year, "input medio": total[0] / total[3], "fuori intervallo %": 100 * total[1] / total[3],
                               "errore G": total[2] / total[3]} for (group, year), total in sums.items() if group == name])
        print(f"\n{name}" + (f" (giorni {ranks}, {len(cells)} celle)" if ranks else f" ({len(cells)} celle)"))
        print(table.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
