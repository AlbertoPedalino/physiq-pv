"""How much the score of every cell has grown since `--steps` timestamps before, written as a run.

The score of a run says how unusual a value is; its rise says how fast it is becoming unusual. A
weather event that starts makes the score grow at its first timestamps, before the score is at its
highest: flagging the highest rises flags the starts. The rise of a cell at a timestamp is its
score minus the score of the same cell `--steps` timestamps before (not finite at the first
timestamps, or where either score is not finite).

The result is a complete run that `climatology`, `events` and the checks accept. Nothing in the
source run is changed.

    python scripts/rise_era5_scores.py --run-dir outputs/run_gen_clim --output-dir outputs/run_gen_clim_rise
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import numpy as np

from scripts.check_era5_dust_lead import score_file

CHUNK = 256  # timestamps read at a time


def rise(run_dir, output_dir, *, steps=1):
    """Write the rise of the score of `run_dir` over `steps` timestamps as a complete run in `output_dir`."""
    run_dir, output = Path(run_dir), Path(output_dir)
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new/empty output directory")
    manifest = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("The run must be complete")
    source = score_file(manifest, "combined")
    scores = np.load(run_dir / source, mmap_mode="r")
    try:
        (output / "scores").mkdir(parents=True)
        result = np.lib.format.open_memmap(output / "scores" / "anomaly_mean.npy", mode="w+", dtype=np.float32,
                                           shape=scores.shape)
        result[:steps] = np.nan
        for start in range(steps, len(scores), CHUNK):
            stop = min(start + CHUNK, len(scores))
            result[start:stop] = (np.asarray(scores[start:stop], dtype=np.float32)
                                  - np.asarray(scores[start - steps:stop - steps], dtype=np.float32))
        result.flush()
        result._mmap.close()
    finally:
        scores._mmap.close()
    np.save(output / "test_timestamps.npy", np.load(run_dir / "test_timestamps.npy"))
    result = {"status": "complete", "grid": manifest["grid"], "score_kind": "rise_of_score", "steps": steps,
              "source_run_dir": str(run_dir), "source_score_file": source,
              "anomaly_mean_file": "scores/anomaly_mean.npy", "anomaly_std_file": None}
    (output / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Complete run with the score")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1, help="Timestamps back the score is compared with")
    args = parser.parse_args(argv)
    rise(args.run_dir, args.output_dir, steps=args.steps)
    print(f"[rise] score minus the score {args.steps} timestamps before -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
