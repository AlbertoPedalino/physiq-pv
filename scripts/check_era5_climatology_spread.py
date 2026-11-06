"""How many groups of the score climatology have almost no spread, and how many flags fall in them.

A climatology run (run_era5_stgan.py climatology --reference validation) divides every value of a score by the spread
of the same score at the same cell, day of the year and hour over the reference years. Groups with zero spread are
left out, but a spread that is tiny and not zero is used as it is, and the score there can grow without bound. The
seasonal normalization of the inputs has a floor for this (--seasonal-std-floor, a share of the overall standard
deviation of the variable); the climatology of the score has none.

The check rebuilds the mean and the spread of every group exactly as the climatology run did (same reference run,
same masks, same period and window), takes as floor `--floor-share` times the standard deviation of all the reference
values (the rule of the inputs), and writes:
- the groups under the floor, how many cells have at least one, and the cells with the most;
- of the values of the climatology run flagged (above its threshold, from the validation when it names one), the share
  that falls in a group under the floor, and the share of values above `--extreme` with the same split.

Rule fixed on 6 November 2026, before it was run: if under 1% of the flags fall in groups under the floor the defect
is a limit to state; at 1% or more the climatology has to be computed again with the floor.

    python scripts/check_era5_climatology_spread.py --run-dir outputs/optuna27/test_generator_clim_day
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

from physiq_pv.era5.seasonality import _moments, _slots
from scripts.check_era5_dust_lead import score_limit

CHUNK = 64


def reference_moments(reference, reference_timestamps, excluded, period, window_days):
    """Count, mean and spread of every slot x cell of the reference, as reference_zscores computes them; and the
    sum, sum of squares and count of all the values kept, for their overall standard deviation."""
    hours = np.unique(reference_timestamps.hour)
    slot, n_slots, _ = _slots(reference_timestamps, hours, period)
    shape = (n_slots, *reference.shape[1:])
    count, total, square = np.zeros(shape, np.int64), np.zeros(shape), np.zeros(shape)
    for start in range(0, len(reference), CHUNK):
        block = np.asarray(reference[start:start + CHUNK], dtype=np.float64)
        valid = np.isfinite(block)
        if excluded is not None:
            valid &= ~np.asarray(excluded[start:start + CHUNK], dtype=bool)
        values = np.where(valid, block, 0.0)
        where = slot[start:start + CHUNK]
        np.add.at(count, where, valid)
        np.add.at(total, where, values)
        np.add.at(square, where, values * values)
    pooled = (int(count.sum()), float(total.sum()), float(square.sum()))
    n, mean, deviation = _moments(count, total, square, len(hours), period, window_days)
    return hours, n, mean, deviation, pooled


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Climatology run with --reference validation and a --reference-run-dir")
    parser.add_argument("--floor-share", type=float, default=0.01,
                        help="The floor: this share of the standard deviation of all the reference values")
    parser.add_argument("--extreme", type=float, default=100.0, help="Values above this are counted apart")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--cells", type=int, default=15, help="How many of the cells with most groups to list")
    args = parser.parse_args(argv)
    run = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    normalization = run.get("normalization") or {}
    if run.get("reference") != "validation":
        raise ValueError("The run must be a climatology against the validation years")
    from scripts.run_era5_stgan import score_files
    from physiq_pv.anomaly_detection.stgan.normal_masks import load_normal_masks
    if normalization.get("reference_run_dir"):
        reference_dir = Path(normalization["reference_run_dir"])
        reference_run = json.loads((reference_dir / "metadata.json").read_text(encoding="utf-8"))
        reference_file = score_files(reference_run, run["source_score_component"])[0]
        reference_times_file = reference_dir / "test_timestamps.npy"
    else:  # the validation errors of the generator saved by the scored run itself
        reference_dir, reference_file = Path(run["source_run_dir"]), "validation_generator_mean.npy"
        reference_times_file = reference_dir / "validation_timestamps.npy"
    period, window = normalization["period"], normalization.get("window_days") or 0
    masks = normalization.get("normal_masks")
    excluded = None if masks is None else load_normal_masks(masks).validation_mask
    reference = np.load(reference_dir / reference_file, mmap_mode="r")
    reference_timestamps = pd.DatetimeIndex(np.load(reference_times_file))
    try:
        hours, n, mean, deviation, (count, total, square) = reference_moments(
            reference, reference_timestamps, excluded, period, window)
    finally:
        reference._mmap.close()
    overall = np.sqrt(max(square / count - (total / count) ** 2, 0.0))
    floor = args.floor_share * overall
    usable = (n >= normalization.get("min_samples", 30)) & (deviation > 0)
    under = usable & (deviation < floor)
    grid = run["grid"]
    latitudes = np.asarray(grid["latitudes"])[np.asarray(grid["rows"])]
    longitudes = np.asarray(grid["longitudes"])[np.asarray(grid["cols"])]
    print(f"[spread] reference {reference_dir} ({reference_file}), {period} +-{window} days x {len(hours)} hours | "
          f"overall std of the reference {overall:.4g}, floor {args.floor_share:g} x = {floor:.4g}")
    print(f"groups usable {int(usable.sum())}, under the floor {int(under.sum())} "
          f"({100 * under.sum() / max(int(usable.sum()), 1):.3f}%); cells with at least one group under it "
          f"{int(under.any(axis=0).sum())} of {under.shape[1]}")
    ratio = np.where(usable, deviation / overall, np.nan)
    print("spread of the groups over the overall std, percentiles 0.1 / 1 / 10 / 50: " +
          " / ".join(f"{np.nanpercentile(ratio, q):.4g}" for q in (0.1, 1, 10, 50)))
    per_cell = under.sum(axis=0)
    worst = np.argsort(-per_cell)[:args.cells]
    table = pd.DataFrame({"cella": worst, "lat": latitudes[worst], "lon": longitudes[worst],
                          "gruppi sotto il minimo": per_cell[worst],
                          "spread minimo / std complessiva": np.nanmin(ratio[:, worst], axis=0)})
    print(table[table["gruppi sotto il minimo"] > 0].round(5).to_string(index=False))

    scores = np.load(args.run_dir / run["anomaly_mean_file"], mmap_mode="r")
    timestamps = pd.DatetimeIndex(np.load(args.run_dir / "test_timestamps.npy"))
    slot, _, _ = _slots(timestamps, hours, period)
    flagged = flagged_under = extreme = extreme_under = 0
    try:
        limit = score_limit(scores, args.top_percent)
        for start in range(0, len(scores), CHUNK):
            block = np.asarray(scores[start:start + CHUNK], dtype=np.float64)
            low = under[slot[start:start + CHUNK]]
            with np.errstate(invalid="ignore"):
                marked, high = block > limit, block > args.extreme
            flagged += int(marked.sum())
            flagged_under += int((marked & low).sum())
            extreme += int(high.sum())
            extreme_under += int((high & low).sum())
    finally:
        scores._mmap.close()
    share = 100 * flagged_under / max(flagged, 1)
    print(f"\n[flags] threshold {limit:.4g} (highest {args.top_percent:g}%): flagged {flagged}, in groups under the floor "
          f"{flagged_under} ({share:.3f}%)")
    print(f"values above {args.extreme:g}: {extreme} ({100 * extreme / max(flagged, 1):.3f}% of the flagged), in groups "
          f"under the floor {extreme_under} ({100 * extreme_under / max(extreme, 1):.1f}% of them)")
    print(f"rule fixed before: under 1% of the flags in groups under the floor = a limit to state; 1% or more = "
          f"recompute the climatology with the floor -> {'limit' if share < 1 else 'recompute'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
