"""Average the anomaly score of several finished ERA5 STGAN runs into a new run for `climatology` and `events`.

The ranking of the highest scores changes from one epoch checkpoint to the next. This script
writes the mean of the combined score (scores/anomaly_mean.npy) of runs scored on the same test
period and grid, for example two epoch checkpoints of one training. Each run keeps its own
min-max scale, so the runs do not weigh exactly the same. Nothing in the source runs is changed.

    python scripts/average_era5_stgan_scores.py \
        --run-dirs outputs/run_epoch4 outputs/run_epoch6 --output-dir outputs/run_mean46
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CHUNK = 512  # timestamps read at a time


def average_scores(run_dirs, output_dir):
    """Write the mean score of `run_dirs` as a complete run in `output_dir` and return its manifest."""
    run_dirs, output = [Path(run) for run in run_dirs], Path(output_dir)
    if len(run_dirs) < 2:
        raise ValueError("Averaging needs at least two runs")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty output directory")
    manifests = [json.loads((run / "metadata.json").read_text(encoding="utf-8")) for run in run_dirs]
    if any(manifest.get("status") != "complete" for manifest in manifests):
        raise ValueError("Every run must be complete")
    timestamps = np.load(run_dirs[0] / "test_timestamps.npy")
    scores = [np.load(run / manifest.get("anomaly_mean_file", manifest.get("scores_file")), mmap_mode="r")
              for run, manifest in zip(run_dirs, manifests)]
    try:
        if (any(array.shape != scores[0].shape for array in scores)
                or any(manifest["grid"] != manifests[0]["grid"] for manifest in manifests)
                or any(not np.array_equal(timestamps, np.load(run / "test_timestamps.npy")) for run in run_dirs[1:])):
            raise ValueError("The runs must share timestamps and grid")
        (output / "scores").mkdir(parents=True)
        mean = np.lib.format.open_memmap(output / "scores" / "anomaly_mean.npy", mode="w+", dtype=np.float32,
                                         shape=scores[0].shape)
        for start in range(0, len(mean), CHUNK):
            mean[start:start + CHUNK] = np.mean(
                [np.asarray(array[start:start + CHUNK], dtype=np.float64) for array in scores], axis=0)
        mean.flush()
        mean._mmap.close()
    finally:
        for array in scores:
            array._mmap.close()
    np.save(output / "test_timestamps.npy", timestamps)
    manifest = {"status": "complete", "grid": manifests[0]["grid"], "score_kind": "mean_of_runs",
                "source_run_dirs": [str(run) for run in run_dirs], "anomaly_mean_file": "scores/anomaly_mean.npy",
                "anomaly_std_file": None}
    (output / "metadata.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parser():
    root = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    root.add_argument("--run-dirs", type=Path, nargs="+", required=True, help="Two or more complete runs")
    root.add_argument("--output-dir", type=Path, required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    manifest = average_scores(args.run_dirs, args.output_dir)
    print(f"[average-scores] runs={len(args.run_dirs)} output={args.output_dir}")
    print(json.dumps({key: value for key, value in manifest.items() if key != "grid"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
