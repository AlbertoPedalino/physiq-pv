"""The usual spread of the dropout draws of G and D at every cell, for each calendar month and hour of the day.

A spread is larger where the weather is more variable (convective places and seasons): divided by its usual
value it says whether the draws disagree more than they do there at that time of year and day. The normals
come from a run scored with Monte Carlo dropout over the validation years and are applied to any period with
`--spread-normals` in the alarm, direction and E-OBS checks.

    python scripts/build_era5_spread_normals.py --components-dir outputs/run_validation \
        --output outputs/spread_normals_validation.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import numpy as np
import pandas as pd

from scripts.check_era5_components_by_group import build_spread_normals, open_components


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--components-dir", type=Path, required=True, help="Run scored with dropout draws")
    parser.add_argument("--output", type=Path, required=True, help="The .npz file of the normals")
    args = parser.parse_args(argv)
    timestamps = pd.DatetimeIndex(np.load(args.components_dir / "test_timestamps.npy"))
    normals = build_spread_normals(open_components(args.components_dir), timestamps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **normals)
    print(f"[spread-normals] {timestamps[0].date()} to {timestamps[-1].date()}: {len(normals['keys'])} month-hours, "
          f"{normals['values'].shape[1]} cells -> {args.output}")
    return normals


if __name__ == "__main__":
    main()
