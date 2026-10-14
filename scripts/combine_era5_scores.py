"""One score out of several: the highest or the mean of them, written as a run the checks read.

The scores must be on the same scale, as those written by the `climatology` step are (the distance of
the score from the usual of its cell, calendar month and time of day, in its standard deviations).
`--how max` flags a value when any of the scores is high, with the alarms of one score; `--how mean`
when they are high together. Where a score is not finite the others decide; where none is, the
combined score is not finite either.

    python scripts/combine_era5_scores.py --run-dirs outputs/run_no_model_clim outputs/run_previous_clim \
        --output-dir outputs/run_simple_max --how max
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import numpy as np
import pandas as pd

from scripts.check_era5_dust_lead import score_file

CHUNK = 256  # timestamps read at a time


def combine(run_dirs, output_dir, *, how="max"):
    """Write the combined score of the runs `run_dirs` as a complete run in `output_dir`; return its manifest."""
    if how not in ("max", "mean"):
        raise ValueError("how must be max or mean")
    if len(run_dirs) < 2:
        raise ValueError("Give at least two scores to combine")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty output directory")
    manifests = [json.loads((Path(path) / "metadata.json").read_text(encoding="utf-8")) for path in run_dirs]
    timestamps = pd.DatetimeIndex(np.load(Path(run_dirs[0]) / "test_timestamps.npy"))
    opened = []
    try:
        for path, manifest in zip(run_dirs, manifests):
            if not pd.DatetimeIndex(np.load(Path(path) / "test_timestamps.npy")).equals(timestamps):
                raise ValueError(f"The score is not over the same timestamps: {path}")
            opened.append(np.load(Path(path) / score_file(manifest, "combined"), mmap_mode="r"))
            if opened[-1].shape != opened[0].shape:
                raise ValueError(f"The score is not over the same cells: {path}")
        (output / "scores").mkdir(parents=True)
        result = np.lib.format.open_memmap(output / "scores" / "anomaly_mean.npy", mode="w+", dtype=np.float32,
                                           shape=opened[0].shape)
        reduce = np.nanmax if how == "max" else np.nanmean
        for start in range(0, len(timestamps), CHUNK):
            stack = np.stack([np.asarray(array[start:start + CHUNK], dtype=np.float32) for array in opened])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells stay NaN
                result[start:start + CHUNK] = reduce(stack, axis=0)
        result.flush()
        result._mmap.close()
    finally:
        for array in opened:
            array._mmap.close()
    np.save(output / "test_timestamps.npy", timestamps.as_unit("ns").asi8)
    manifest = {"status": "complete", "grid": manifests[0]["grid"], "score_kind": f"{how}_of_scores",
                "source_run_dirs": [str(path) for path in run_dirs], "anomaly_mean_file": "scores/anomaly_mean.npy",
                "anomaly_std_file": None}
    (output / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True, help="Runs with scores on the same scale")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--how", choices=("max", "mean"), default="max")
    args = parser.parse_args(argv)
    combine(args.run_dirs, args.output_dir, how=args.how)
    print(f"[combine] {args.how} of {len(args.run_dirs)} scores -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
