"""Make the checks flag a score above the threshold of the same score over the validation years.

The checks flag the values of a score in its highest `--top-percent`, a percentile of the score itself.
When the threshold is chosen on the validation years and the score is then read on the test, the
percentile must come from the validation: this script writes in the metadata.json of the run of the
test score the run of the same score over the validation years (`threshold_reference`), and every
check that reads the threshold with `score_limit` takes it from there. On the test the share of flagged
values is then whatever lies above that threshold, about but not exactly the percent asked.

The two runs must be the same kind of score on the same cells, the reference before the scored run.
`--clear` removes the reference. Nothing else in the run is changed.

    python scripts/set_era5_threshold_reference.py --run-dir outputs/test_gen_clim \
        --reference-run-dir outputs/val_gen_clim
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KIND = ("score_kind", "source_score_component", "score_mode")  # what the two runs must share when they have it


def set_reference(run_dir, reference_run_dir=None):
    """Write (or with None remove) the threshold reference of `run_dir`; return its metadata."""
    run_dir = Path(run_dir)
    path = run_dir / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if reference_run_dir is None:
        metadata.pop("threshold_reference", None)
    else:
        reference_dir = Path(reference_run_dir).resolve()
        reference = json.loads((reference_dir / "metadata.json").read_text(encoding="utf-8"))
        for manifest, where in ((metadata, run_dir), (reference, reference_dir)):
            if manifest.get("status") != "complete":
                raise ValueError(f"The run is not complete: {where}")
        if reference.get("threshold_reference"):
            raise ValueError("The reference has a threshold reference of its own")
        if metadata["grid"] != reference["grid"]:
            raise ValueError("The two runs are not on the same cells")
        differing = [key for key in KIND if metadata.get(key) != reference.get(key)]
        if differing:
            raise ValueError(f"The two runs are not the same kind of score: {differing}")
        scored = pd.DatetimeIndex(np.load(run_dir / "test_timestamps.npy"))
        before = pd.DatetimeIndex(np.load(reference_dir / "test_timestamps.npy"))
        if before[-1] >= scored[0]:
            raise ValueError("The reference must cover years before the scored run")
        metadata["threshold_reference"] = str(reference_dir)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Run of the score over the test")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--reference-run-dir", type=Path, help="Run of the same score over the validation years")
    target.add_argument("--clear", action="store_true", help="Remove the threshold reference")
    args = parser.parse_args(argv)
    metadata = set_reference(args.run_dir, None if args.clear else args.reference_run_dir)
    print(f"[threshold] {args.run_dir}: {metadata.get('threshold_reference') or 'its own percentile'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
