"""The score of no model: the error of answering the seasonal normal at every cell and time.

A run with the seasonal normalization scores a cell by how far the reconstruction of the generator
is from the data, in the units of the model. This script writes the same score for the answer that
needs no model: the seasonal normal itself, which is zero in the standardised data (the trend of
the training years, where one is removed, is already taken away there). In the units of the model
the error of that answer for a variable is 2 z / scale, with z the standardised value and scale
the min-max range of the training years.

With `--previous-value N` the answer is instead the value of N timestamps before: the weather of
a moment ago, still as far from its normal as it was. Its error is 2 (z - z before) / scale. A model
that reads the last hours has to do better than that answer to be worth its training.

What a trained model adds is what its score finds that this one does not. The result is a run
with the squared error by variable (scores/feature_scores.npy) and their mean with the weights of
the run (scores/anomaly_mean.npy), so that `climatology`, `rebalance` and the checks read it as
they read the run. It reads the standardised test period from the shared cache of the run, which
has to be there still; nothing in the run or in the cache is changed.

    python scripts/score_era5_seasonal_normal.py --run-dir outputs/run --output-dir outputs/run_no_model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import numpy as np

from physiq_pv.anomaly_detection.stgan.shared_cache import MANIFEST

CHUNK = 256  # timestamps read at a time


def score(run_dir, output_dir, *, cache_dir=None, previous=0):
    """Write the error of the seasonal normal over the scored period of `run_dir` as a run in `output_dir`;
    with `previous`, the error of the value that many timestamps before."""
    run_dir, output = Path(run_dir), Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty output directory")
    manifest = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("The run must be complete")
    backend = manifest.get("backend", {})
    seasonal = backend.get("seasonal_normalization")
    if not seasonal:
        raise ValueError("The run has no seasonal normalization: there is no seasonal normal to answer")
    cache = Path(cache_dir) if cache_dir is not None else Path(seasonal["mean_file"]).parent
    if not (cache / MANIFEST).is_file() or not (cache / "test.npy").is_file():
        raise ValueError(f"The shared cache of the run is not there: {cache}")
    scale = np.asarray(json.loads((cache / MANIFEST).read_text(encoding="utf-8"))["built"]["scale"], dtype=np.float64)
    timestamps = np.load(run_dir / "test_timestamps.npy")
    standardised = np.load(cache / "test.npy", mmap_mode="r")
    try:
        names = list(manifest.get("preparation", {}).get("features") or
                     [f"feature_{index}" for index in range(standardised.shape[2])])
        # The cache holds a few timestamps of context before the scored period.
        first = len(standardised) - len(timestamps)
        if first < 0 or standardised.shape[2] != len(names) or len(scale) != len(names):
            raise ValueError("The cache and the run are not over the same period and variables")
        if not 0 <= previous <= first:
            raise ValueError(f"The cache holds {first} timestamps before the scored period: previous must be within them")
        saved = backend.get("reconstruction_feature_weights")
        weights = np.array([saved[name] for name in names]) if saved else np.ones(len(names))
        (output / "scores").mkdir(parents=True)
        errors = np.lib.format.open_memmap(output / "scores" / "feature_scores.npy", mode="w+", dtype=np.float32,
                                           shape=(len(timestamps),) + standardised.shape[1:])
        mean = np.lib.format.open_memmap(output / "scores" / "anomaly_mean.npy", mode="w+", dtype=np.float32,
                                         shape=(len(timestamps), standardised.shape[1]))
        for start in range(0, len(timestamps), CHUNK):
            rows = slice(first + start, first + min(start + CHUNK, len(timestamps)))
            block = np.asarray(standardised[rows], dtype=np.float64)
            if previous:
                block = block - np.asarray(standardised[rows.start - previous:rows.stop - previous], dtype=np.float64)
            squared = (2.0 * block / scale) ** 2
            errors[start:start + CHUNK] = squared
            mean[start:start + CHUNK] = (squared * weights).mean(axis=2)
        for array in (errors, mean):
            array.flush()
            array._mmap.close()
    finally:
        standardised._mmap.close()
    np.save(output / "test_timestamps.npy", timestamps)
    kind = "error_of_the_previous_value_with_no_model" if previous else "error_of_the_seasonal_normal_with_no_model"
    result = {"status": "complete", "grid": manifest["grid"], "score_kind": kind, "previous_timestamps": previous,
              "source_run_dir": str(run_dir), "cache_dir": str(cache), "preparation": {"features": names},
              "backend": {"reconstruction_feature_weights": saved}, "anomaly_mean_file": "scores/anomaly_mean.npy",
              "anomaly_std_file": None}
    (output / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Complete run with the seasonal normalization")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="Shared cache of the run; default: the one the run names")
    parser.add_argument("--previous-value", type=int, default=0,
                        help="Answer the value this many timestamps before instead of the seasonal normal")
    args = parser.parse_args(argv)
    result = score(args.run_dir, args.output_dir, cache_dir=args.cache_dir, previous=args.previous_value)
    print(f"[{'previous-value' if args.previous_value else 'seasonal-normal'}] variables={len(result['preparation']['features'])} cache={result['cache_dir']} "
          f"output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
