"""Two checks on scores already computed: the share of flagged values year by year, and the mean error of the model
against the two answers that need no model.

1. `yearly`: for every score (NAME=DIR[@component], as in the other checks), the percentage of the values of each test
   year above its threshold, the highest `--top-percent` taken from the validation years exactly as the thesis does
   (`score_limit`: the `threshold_reference` of the run when it names one). With thresholds fixed on five validation
   years and a drift in the inputs, a share that grows over the years would show it.
2. `skill`: the mean over the validation values, without the excluded ones (`validation_mask.npy` of the normal masks),
   of three errors in the same units and with the same weights of the variables: G, the reconstruction error of the
   generator (mean over the 3x3 patch and the Monte Carlo draws), the error of the seasonal normal and that of the value
   three hours before (`score_era5_seasonal_normal.py`). The weights of the runs are compared first. G is the mean of
   the errors of the draws with dropout, not the error of the mean reconstruction: it is higher by the variance of the
   draws (Section 4.4 of the thesis), so it bounds the error of the model from above.

Usage (on the server, from the checkout):
    python scripts/check_era5_yearly_share_and_skill.py yearly --output outputs/anni_segnalazioni.csv \
        --scores Gn=outputs/optuna27/test_generator_clim_day normal=outputs/confronti_good2/climatology/test/normal \
                 change_3h=outputs/confronti_good2/climatology/test/change_3h
    python scripts/check_era5_yearly_share_and_skill.py skill --masks outputs/era5_sweep_1999/normal_masks \
        --model outputs/stgan_optuna27_validation@generator \
        --normal outputs/confronti_good2/validation/simple/normal \
        --change outputs/confronti_good2/validation/simple/change_3h
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from scripts.check_era5_dust_lead import score_file, score_limit
from scripts.check_era5_score_correlation import open_component, parse_score

CHUNK = 256  # timestamps read at a time


def timestamps_of(directory):
    return pd.DatetimeIndex(np.load(Path(directory) / "test_timestamps.npy"))


def yearly_share(scores, timestamps, limit):
    """Percentage of the finite values of every year above `limit`, and the number of finite values."""
    years = np.unique(timestamps.year)
    above, count = dict.fromkeys(years, 0), dict.fromkeys(years, 0)
    for start in range(0, len(timestamps), CHUNK):
        block = np.asarray(scores[start:start + CHUNK], dtype=np.float64)
        finite = np.isfinite(block)
        for year in np.unique(timestamps.year[start:start + CHUNK]):
            rows = timestamps.year[start:start + CHUNK] == year
            above[year] += int((finite[rows] & (block[rows] > limit)).sum())
            count[year] += int(finite[rows].sum())
    return pd.DataFrame({"year": years, "share_pct": [100 * above[y] / count[y] if count[y] else np.nan for y in years],
                         "values": [count[y] for y in years]})


def run_yearly(args):
    tables = []
    for text in args.scores:
        name, directory, component = parse_score(text)
        manifest = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        scores, maps = open_component(directory, manifest, component)
        try:
            limit = score_limit(scores, args.top_percent)
            table = yearly_share(scores, timestamps_of(directory), limit)
        finally:
            for array in maps:
                array._mmap.close()
        table.insert(0, "score", name)
        table["limit"] = limit
        tables.append(table)
        print(f"[yearly] {name}: threshold {limit:.6g}; share {table.share_pct.min():.2f}-{table.share_pct.max():.2f}%")
    frame = pd.concat(tables, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(frame.pivot(index="year", columns="score", values="share_pct").round(2).to_string())


def weights_of(directory):
    manifest = json.loads((Path(directory) / "metadata.json").read_text(encoding="utf-8"))
    return manifest.get("backend", {}).get("reconstruction_feature_weights")


def mask_rows(mask, count):
    """The rows of the validation mask that belong to the `count` scored timestamps (the mask may hold the context
    rows before them)."""
    if len(mask) == count:
        return mask
    if len(mask) > count:
        return mask[len(mask) - count:]
    raise ValueError(f"The validation mask has {len(mask)} rows, the scores {count}")


def masked_mean(scores, keep):
    """Mean of `scores` [time, cell] over the finite values where `keep`, and their number."""
    total, count = 0.0, 0
    for start in range(0, len(keep), CHUNK):
        block = np.asarray(scores[start:start + CHUNK], dtype=np.float64)
        use = np.asarray(keep[start:start + CHUNK]) & np.isfinite(block)
        total += float(block[use].sum())
        count += int(use.sum())
    return total / count, count


def run_skill(args):
    model_dir, _, component = str(args.model).partition("@")
    sources = {"G": (Path(model_dir), component or "generator"), "distanza dal normale": (args.normal, "combined"),
               "variazione in tre ore": (args.change, "combined")}
    weights = {name: weights_of(directory) for name, (directory, _) in sources.items()}
    if any(w != weights["G"] for w in weights.values()):
        raise ValueError(f"The runs do not share the weights of the variables: {weights}")
    times = {name: timestamps_of(directory) for name, (directory, _) in sources.items()}
    if any(not t.equals(times["G"]) for t in times.values()):
        raise ValueError("The runs are not over the same validation timestamps")
    excluded = np.load(Path(args.masks) / "validation_mask.npy", mmap_mode="r")
    keep = ~mask_rows(np.asarray(excluded, dtype=bool), len(times["G"]))
    rows = []
    for name, (directory, part) in sources.items():
        manifest = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        scores, maps = open_component(directory, manifest, part)
        try:
            if scores.shape != keep.shape:
                raise ValueError(f"{name}: scores {scores.shape}, mask {keep.shape}")
            mean, count = masked_mean(scores, keep)
        finally:
            for array in maps:
                array._mmap.close()
        rows.append({"error": name, "file": str(directory / score_file(manifest, part)), "mean": mean, "values": count})
    frame = pd.DataFrame(rows)
    frame["ratio_to_G"] = frame["mean"] / frame.loc[0, "mean"]
    print(f"[skill] validation {times['G'][0]} to {times['G'][-1]}, excluded values left out "
          f"({100 * (1 - keep.mean()):.2f}% of the values)")
    with pd.option_context("display.width", 200, "display.max_colwidth", 80):
        print(frame.to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    yearly = commands.add_parser("yearly")
    yearly.add_argument("--scores", nargs="+", required=True, help="NAME=DIR[@component]")
    yearly.add_argument("--top-percent", type=float, default=5)
    yearly.add_argument("--output", type=Path, required=True)
    skill = commands.add_parser("skill")
    skill.add_argument("--model", required=True, help="Validation run of the model, DIR[@component]; generator by default")
    skill.add_argument("--normal", type=Path, required=True, help="Validation run of the error of the seasonal normal")
    skill.add_argument("--change", type=Path, required=True, help="Validation run of the error of the value 3 h before")
    skill.add_argument("--masks", type=Path, required=True, help="Normal masks directory (validation_mask.npy)")
    args = parser.parse_args(argv)
    run_yearly(args) if args.command == "yearly" else run_skill(args)


if __name__ == "__main__":
    main()
