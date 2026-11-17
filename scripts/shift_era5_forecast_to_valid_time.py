"""The forecast anomaly at the timestamp it is valid for, not at the one it is known at.

score_era5_forecast_anomaly.py writes the forecast anomaly of T + 1 at T (row i holds the forecast of timestamp i + 1,
the last row is NaN), so that a score that looks ahead is known where it is set. To compare it cell by cell with the
other methods, whose values describe the field of their own timestamp, it is moved one step later: row i of the new run
holds the forecast of timestamp i, made from the history up to i - 1 (the first row is NaN). As in the verification of
forecasts, every value is read at its valid time; the forecast was available 3 h before. The move is done on the raw run,
before its climatology. Every score file of the run (the mean of the draws and, with the dropouts on, their spread) is
moved; the timestamps and the rest of the metadata stay.

    python scripts/shift_era5_forecast_to_valid_time.py --run-dir outputs/optuna27/previsione_mc/test/P \
        --output-dir outputs/optuna27/previsione_mc_valida/test/P
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

KNOWN_AT = "forecast_anomaly_of_the_next_timestamp"
VALID_AT = "forecast_anomaly_at_its_valid_timestamp"
CHUNK = 256
FILE_KEYS = ("anomaly_mean_file", "anomaly_std_file", "generator_mean_file", "generator_std_file")


def shift_file(source, target):
    """target[0] = NaN, target[i] = source[i - 1], read and written a block at a time."""
    values = np.load(source, mmap_mode="r")
    try:
        out = np.lib.format.open_memmap(target, mode="w+", dtype=values.dtype, shape=values.shape)
        try:
            out[0] = np.nan
            for start in range(1, len(values), CHUNK):
                stop = min(start + CHUNK, len(values))
                out[start:stop] = values[start - 1:stop - 1]
            out.flush()
        finally:
            out._mmap.close()
    finally:
        values._mmap.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Run written by score_era5_forecast_anomaly.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="New/empty directory")
    args = parser.parse_args(argv)
    run = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    if run.get("status") != "complete":
        raise ValueError(f"The run is not complete: {args.run_dir}")
    if run.get("score_kind") != KNOWN_AT:
        raise ValueError(f"Not a forecast anomaly known one step before (score_kind {run.get('score_kind')!r})")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Use a new/empty directory: {args.output_dir}")
    (args.output_dir / "scores").mkdir(parents=True, exist_ok=True)
    for name in sorted({run[key] for key in FILE_KEYS if run.get(key)}):
        shift_file(args.run_dir / name, args.output_dir / name)
        print(f"[shift] {name}: moved one timestamp later", flush=True)
    shutil.copyfile(args.run_dir / "test_timestamps.npy", args.output_dir / "test_timestamps.npy")
    shifted = {**run, "score_kind": VALID_AT, "shifted_from": str(args.run_dir),
               "shift": "row i = forecast of timestamp i, from the history up to i - 1; first row NaN"}
    (args.output_dir / "metadata.json").write_text(json.dumps(shifted, indent=2), encoding="utf-8")
    print(f"[shift] {args.run_dir} -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
