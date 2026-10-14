"""How much do the Monte Carlo dropout draws move the score, and do they move the flags?

A run scored with Monte Carlo dropout keeps, for every cell and timestamp, the mean of the score over
the draws and their standard deviation (the spread). The score that flags is the mean. The check reads
both, on one timestamp every `--stride`, and says

- by band of the score, from its lower half to its highest 0.1%: the mean score, the mean spread and
  their ratio, and the rank correlation of score and spread over all the values read;
- for the flagged values, those in the highest `--top-percent` of the score: how many stay above the
  threshold one and two spreads lower, and for the others how many would pass it one spread higher.
  A flag the draws can undo is a flag the dropout decides, not the weather.

`--score-component generator` reads the reconstruction error of a training run; a run written by the
`climatology` step keeps the spread in the units of its own score.

    python scripts/check_era5_dropout_spread.py --run-dir outputs/run --score-component generator
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import numpy as np

from scripts.check_era5_dust_lead import COMPONENTS, THRESHOLD_STRIDE, score_file

CHUNK = 64  # timestamps read at a time
BANDS = ((0, 50), (50, 90), (90, 95), (95, 99), (99, 99.9), (99.9, 100))  # percentiles of the score
RANKED = 2_000_000  # values kept for the rank correlation
NAME, WIDTH = 44, 16


def std_file(manifest, component):
    """Path, relative to the run, of the spread of `component` over the draws, or None if it was not kept."""
    if component == "combined":
        return manifest.get("anomaly_std_file")
    return manifest.get(f"{component}_std_file")


def ranks(values):
    order = np.empty(len(values))
    order[np.argsort(values, kind="stable")] = np.arange(len(values))
    return order


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--score-component", choices=COMPONENTS, default="combined")
    parser.add_argument("--top-percent", type=float, default=5, help="Flagged: this highest share of the score")
    parser.add_argument("--stride", type=int, default=9, help="Timestamps read: one every this many")
    args = parser.parse_args(argv)
    if args.stride < 1 or args.stride % 2 == 0:
        raise ValueError("stride must be odd and at least 1")
    manifest = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    spread_path = std_file(manifest, args.score_component)
    if not spread_path:
        raise ValueError(f"The run kept no spread of the draws for {args.score_component}: it was scored without them")
    scores = np.load(args.run_dir / score_file(manifest, args.score_component), mmap_mode="r")
    spreads = np.load(args.run_dir / spread_path, mmap_mode="r")
    try:
        if scores.shape != spreads.shape:
            raise ValueError("The score and its spread are not over the same timestamps and cells")
        mean = np.asarray(scores[::args.stride], dtype=np.float64)
        spread = np.asarray(spreads[::args.stride], dtype=np.float64)
        threshold = float(np.nanpercentile(np.asarray(scores[::THRESHOLD_STRIDE]), 100 - args.top_percent))
    finally:
        scores._mmap.close()
        spreads._mmap.close()
    finite = np.isfinite(mean) & np.isfinite(spread)
    mean, spread = mean[finite], spread[finite]
    if not len(mean):
        raise ValueError("No value has a finite score and spread")
    if (spread == 0).all():
        print("[dropout-spread] every spread is zero: this score does not depend on the dropout draws")
        return 0
    edges = np.percentile(mean, sorted({edge for band in BANDS for edge in band}))
    edge = dict(zip(sorted({edge for band in BANDS for edge in band}), edges))
    keep = np.linspace(0, len(mean) - 1, min(RANKED, len(mean))).astype(np.int64)
    correlation = float(np.corrcoef(ranks(mean[keep]), ranks(spread[keep]))[0, 1])
    flagged = mean > threshold

    def row(label, values, form="{:.4g}"):
        print(f"{label:<{NAME}}" + "".join(f"{form.format(value):>{WIDTH}}" for value in values))

    print(f"[dropout-spread] {args.run_dir.name}, {args.score_component} | one timestamp every {args.stride} | "
          f"{len(mean)} values | spread: standard deviation of the score over the dropout draws")
    print(f"\n{'score percentiles':<{NAME}}" + "".join(f"{name:>{WIDTH}}" for name in ("mean score", "mean spread",
                                                                                    "spread / score")))
    for low, high in BANDS:
        inside = (mean >= edge[low]) & (mean <= edge[high])
        row(f"{low:g}-{high:g}", [mean[inside].mean(), spread[inside].mean(), spread[inside].mean() / mean[inside].mean()])
    print(f"rank correlation between score and spread: {correlation:+.3f}")
    print(f"\nflagged: the highest {args.top_percent:g}% of the score, {flagged.mean() * 100:.1f}% of the values read")
    row("flagged, still flagged one spread lower", [100 * (mean - spread > threshold)[flagged].mean()], "{:.1f}%")
    row("flagged, still flagged two spreads lower", [100 * (mean - 2 * spread > threshold)[flagged].mean()], "{:.1f}%")
    row("not flagged, flagged one spread higher", [100 * (mean + spread > threshold)[~flagged].mean()], "{:.1f}%")
    row("spread of the flagged, over the threshold", [np.median(spread[flagged]) / abs(threshold)], "{:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
