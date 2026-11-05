"""The maps of G, D and D(real) at two timestamps of a scored run, for the figures of the behaviour of the scores.

The two timestamps are fixed by rule: the typical one, whose spatial Spearman correlation between G and D (the
`spatial_correlation.csv` written by check_era5_score_correlation.py) is the closest to its median over the period,
and the one with the highest mean G over the cells; the earliest when two are as close or as high. The maps go to
`--output` (an .npz with `times`, `G`, `D`, `Dreale` [2, cells] and the `grid` of the run as JSON), in this order.

    python scripts/save_era5_instant_maps.py --run-dir outputs/stgan_optuna27_validation \
        --correlation outputs/comportamento_optuna27/correlazioni_g_d/spatial_correlation.csv \
        --output outputs/comportamento_optuna27/mappe_istanti.npz
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

from scripts.check_era5_event_zones import open_score, release
from scripts.check_era5_unreached import CHUNK

COMPONENTS = {"G": "generator", "D": "discriminator", "Dreale": "real"}


def typical_index(correlation):
    """Position of the value closest to the median (the earliest of equally close ones)."""
    values = np.asarray(correlation, dtype=np.float64)
    distance = np.abs(values - np.nanmedian(values))
    return int(np.nanargmin(distance))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Scored run (score_era5_stgan_run.py)")
    parser.add_argument("--correlation", type=Path, required=True, help="spatial_correlation.csv of the same run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    timestamps = pd.DatetimeIndex(np.load(args.run_dir / "test_timestamps.npy"))
    spatial = pd.read_csv(args.correlation, index_col=0, parse_dates=True)["G - D spearman"]
    typical = timestamps.get_loc(spatial.index[typical_index(spatial.to_numpy())])
    manifest, scores, maps = open_score(("G", args.run_dir, "generator"))
    try:
        means = np.empty(len(timestamps))
        for start in range(0, len(timestamps), CHUNK):
            with np.errstate(invalid="ignore"):
                means[start:start + CHUNK] = np.nanmean(np.asarray(scores[start:start + CHUNK], dtype=np.float64), axis=1)
            release(scores)
        highest = int(np.nanargmax(means))
    finally:
        for array in maps:
            array._mmap.close()
    chosen = [typical, highest]
    arrays = {"times": np.array([str(timestamps[index]) for index in chosen]), "grid": json.dumps(manifest["grid"])}
    for name, component in COMPONENTS.items():
        _, values, opened = open_score((name, args.run_dir, component))
        try:
            arrays[name] = np.stack([np.asarray(values[index], dtype=np.float32) for index in chosen])
        finally:
            for array in opened:
                array._mmap.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"[mappe] typical {arrays['times'][0]} (G-D Spearman {spatial.iloc[typical_index(spatial.to_numpy())]:.3f}, "
          f"median {spatial.median():.3f}) | highest mean G {arrays['times'][1]} ({means[highest]:.5f}) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
