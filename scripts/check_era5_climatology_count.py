"""How many reference values every group of the score climatology keeps, and what a minimum would change.

A climatology run (run_era5_stgan.py climatology --reference validation) gives a score only to the groups (cell, day of
the year, hour) with at least `min_samples` reference values left after the unusual values of the validation years are
left out (default 30). This check rebuilds the count of every group exactly as the run did (same reference run, same
masks, same period and window) and also the count without the masks, and writes:
- the size of the window (values per group before the masks) and the share the masks leave, percentiles;
- the groups under `min_samples` (no score today) and the groups that keep less than `--share` of their values (no
  score with a rule "at least that share of the window", like the 80% of WMO-No. 1203 for climate normals);
- of the values of the run flagged (above its threshold, from the validation when it names one), how many fall in
  those groups, i.e. would lose their score.

    python scripts/check_era5_climatology_count.py --run-dir outputs/optuna27/test_generator_clim_day
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

from physiq_pv.era5.seasonality import _slots
from scripts.check_era5_climatology_spread import CHUNK, reference_moments
from scripts.check_era5_dust_lead import score_limit

PERCENTILES = (0, 0.1, 1, 10, 50)


def summarize(kept, total, deviation, min_samples, share):
    """Masks of the groups without a score today and with the share rule, from the counts with and without the masks
    (groups with no reference value at all are left out of both)."""
    present = total > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        kept_share = np.where(present, kept / np.where(present, total, 1), np.nan)
    spread = deviation > 0
    no_score_now = present & ~((kept >= min_samples) & spread)
    no_score_share = present & ~((kept >= share * total) & (kept >= 2) & spread)
    return kept_share, no_score_now, no_score_share


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Climatology run with --reference validation")
    parser.add_argument("--share", type=float, default=0.8, help="Minimum share of the window kept (WMO: 0.8)")
    parser.add_argument("--top-percent", type=float, default=5)
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
    min_samples = normalization.get("min_samples", 30)
    masks = normalization.get("normal_masks")
    excluded = None if masks is None else load_normal_masks(masks).validation_mask
    reference = np.load(reference_dir / reference_file, mmap_mode="r")
    reference_timestamps = pd.DatetimeIndex(np.load(reference_times_file))
    try:
        hours, kept, _, deviation, _ = reference_moments(reference, reference_timestamps, excluded, period, window)
        _, total, _, _, _ = reference_moments(reference, reference_timestamps, None, period, window)
    finally:
        reference._mmap.close()
    kept_share, no_now, no_share = summarize(kept, total, deviation, min_samples, args.share)
    present = total > 0
    groups = int(present.sum())
    print(f"[count] reference {reference_dir} ({reference_file}), {period} +-{window} days x {len(hours)} hours, "
          f"masks {'yes' if excluded is not None else 'no'} | groups with reference values {groups}")
    print("values per group before the masks, percentiles " + "/".join(map(str, PERCENTILES)) + ": " +
          " / ".join(f"{np.percentile(total[present], q):.0f}" for q in PERCENTILES) + f", max {total.max():.0f}")
    print("values kept after the masks, same percentiles: " +
          " / ".join(f"{np.percentile(kept[present], q):.0f}" for q in PERCENTILES))
    print("share kept, same percentiles: " +
          " / ".join(f"{np.nanpercentile(kept_share, q):.3f}" for q in PERCENTILES))
    for name, mask in ((f"under {min_samples} values or no spread (no score today)", no_now),
                       (f"keeping less than {args.share:g} of the window or no spread", no_share)):
        print(f"groups {name}: {int(mask.sum())} ({100 * mask.sum() / max(groups, 1):.4f}%), "
              f"cells with at least one {int(mask.any(axis=0).sum())} of {mask.shape[1]}")

    scores = np.load(args.run_dir / run["anomaly_mean_file"], mmap_mode="r")
    timestamps = pd.DatetimeIndex(np.load(args.run_dir / "test_timestamps.npy"))
    slot, _, _ = _slots(timestamps, hours, period)
    flagged = flagged_lost = scored = scored_lost = 0
    try:
        limit = score_limit(scores, args.top_percent)
        for start in range(0, len(scores), CHUNK):
            block = np.asarray(scores[start:start + CHUNK], dtype=np.float64)
            lost = no_share[slot[start:start + CHUNK]] & ~no_now[slot[start:start + CHUNK]]
            finite = np.isfinite(block)
            with np.errstate(invalid="ignore"):
                marked = block > limit
            scored += int(finite.sum())
            scored_lost += int((finite & lost).sum())
            flagged += int(marked.sum())
            flagged_lost += int((marked & lost).sum())
    finally:
        scores._mmap.close()
    print(f"\n[flags] threshold {limit:.4g} (highest {args.top_percent:g}%): scored values {scored}, flagged {flagged}")
    print(f"with the rule 'at least {args.share:g} of the window' these would lose their score: scored "
          f"{scored_lost} ({100 * scored_lost / max(scored, 1):.4f}%), flagged {flagged_lost} "
          f"({100 * flagged_lost / max(flagged, 1):.4f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
