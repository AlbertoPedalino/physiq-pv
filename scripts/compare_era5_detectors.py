"""Do the detectors flag the same values, and what does their majority flag?

Every detector, named on the command line (`--runs NAME=DIR`, or `NAME=DIR@component` for the generator or
discriminator mean of a scored run), flags the values in its highest `--top-percent`, with the threshold of
`score_limit` (that of the validation years when the run names one), so that all flag about as many values.
Only the values every detector scores are read. The check says:
- for every detector, the share of the values it flags, of its flags the majority also flags, and of its
  flags no other detector flags;
- for every pair, the share of the flags of the row also flagged by the column, and the Jaccard index of
  the two sets of flags (common flags over flags of either);
- how many detectors flag each value: the share of the values flagged by none, one, ..., all.

The majority vote flags a value when more than half of the detectors flag it (`--votes` for another number).
With `--vote-dir` the number of detectors that flag each value is written as a run, for the other checks:
a value is flagged by the vote when that number reaches the votes, which the run records.

    python scripts/compare_era5_detectors.py --runs G+clim=outputs/gen_clim simple=outputs/simple_max \
        pca=outputs/shallow/pca chen=outputs/spatial/chen_multivariate --vote-dir outputs/vote --figure vote.png
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

from physiq_pv.anomaly_detection.stgan.loading import release_pages
from scripts.check_era5_dust_lead import score_file, score_limit
from scripts.check_era5_score_correlation import parse_score

CHUNK = 256  # timestamps read at a time
NAME = 16


def agreement(flags, finite):
    """Counts of one block: flags [detectors, values] of the values every detector scores (`finite` [values]).
    Returns (values, per detector flags, pair counts [detectors, detectors], votes histogram [detectors + 1])."""
    chosen = flags[:, finite].astype(np.int64)
    votes = chosen.sum(axis=0)
    return (int(finite.sum()), chosen.sum(axis=1), chosen @ chosen.T,
            np.bincount(votes, minlength=len(flags) + 1), votes)


def compare(sources, top_percent, votes_needed, vote_dir=None):
    """Read every detector over the period; return the counts and, with `vote_dir`, write the vote run."""
    names = [name for name, _, _ in sources]
    manifests = {name: json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
                 for name, directory, _ in sources}
    grid = manifests[names[0]]["grid"]
    timestamps = pd.DatetimeIndex(np.load(sources[0][1] / "test_timestamps.npy"))
    for name, directory, _ in sources[1:]:
        if manifests[name]["grid"] != grid or not timestamps.equals(pd.DatetimeIndex(np.load(directory / "test_timestamps.npy"))):
            raise ValueError(f"{name} is not over the grid and timestamps of {names[0]}")
    arrays = {name: np.load(directory / score_file(manifests[name], component), mmap_mode="r")
              for name, directory, component in sources}
    count = len(names)
    totals = dict(values=0, flagged=np.zeros(count, dtype=np.int64), pairs=np.zeros((count, count), dtype=np.int64),
                  histogram=np.zeros(count + 1, dtype=np.int64), with_majority=np.zeros(count, dtype=np.int64),
                  alone=np.zeros(count, dtype=np.int64))
    written = None
    try:
        limits = {name: score_limit(array, top_percent) for name, array in arrays.items()}
        if vote_dir is not None:
            (vote_dir / "scores").mkdir(parents=True)
            written = np.lib.format.open_memmap(vote_dir / "scores" / "anomaly_mean.npy", mode="w+", dtype=np.float32,
                                                shape=(len(timestamps), len(grid["rows"])))
        for start in range(0, len(timestamps), CHUNK):
            blocks = np.stack([np.asarray(arrays[name][start:start + CHUNK], dtype=np.float64) for name in names])
            finite = np.isfinite(blocks).all(axis=0)
            flags = blocks > np.array([limits[name] for name in names])[:, None, None]
            values, flagged, pairs, histogram, votes = agreement(flags.reshape(count, -1), finite.ravel())
            totals["values"] += values
            totals["flagged"] += flagged
            totals["pairs"] += pairs
            totals["histogram"] += histogram
            chosen = flags.reshape(count, -1)[:, finite.ravel()]
            totals["with_majority"] += (chosen & (votes >= votes_needed)).sum(axis=1)
            totals["alone"] += (chosen & (votes == 1)).sum(axis=1)
            if written is not None:
                block = np.full(finite.shape, np.nan, dtype=np.float32)
                block[finite] = flags.sum(axis=0)[finite]
                written[start:start + CHUNK] = block
            release_pages(*arrays.values(), *([] if written is None else [written]))
    finally:
        for array in arrays.values():
            array._mmap.close()
        if written is not None:
            written.flush()
            written._mmap.close()
    if vote_dir is not None:
        np.save(vote_dir / "test_timestamps.npy", timestamps.as_unit("ns").asi8)
        (vote_dir / "metadata.json").write_text(json.dumps(
            {"status": "complete", "grid": grid, "score_kind": "majority_vote_count",
             "vote": {"detectors": {name: f"{directory}@{component}" for name, directory, component in sources},
                      "top_percent": top_percent, "thresholds": limits, "votes_needed": votes_needed},
             "anomaly_mean_file": "scores/anomaly_mean.npy", "anomaly_std_file": None}, indent=2), encoding="utf-8")
    return names, totals, timestamps


def plot(path, names, jaccard):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(1.0 + 0.62 * len(names), 0.6 + 0.55 * len(names)), constrained_layout=True)
    image = axis.imshow(np.where(np.eye(len(names), dtype=bool), np.nan, jaccard), cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
    axis.set_yticks(range(len(names)), names)
    for row in range(len(names)):
        for column in range(len(names)):
            if row != column:
                axis.text(column, row, f"{jaccard[row, column]:.2f}", ha="center", va="center", fontsize=8,
                          color="white" if jaccard[row, column] > 0.6 else "#0b0b0b")
    figure.colorbar(image, ax=axis, shrink=0.8, label="indice di Jaccard delle segnalazioni")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", required=True,
                        help="NAME=DIR (the score of a run) or NAME=DIR@generator|discriminator|combined")
    parser.add_argument("--top-percent", type=float, default=5, help="Every detector flags this highest share")
    parser.add_argument("--votes", type=int, help="Detectors needed to flag a value; default more than half")
    parser.add_argument("--vote-dir", type=Path, help="New directory: the number of detectors flagging each value, as a run")
    parser.add_argument("--figure", type=Path, help="PNG of the Jaccard indices of the pairs")
    args = parser.parse_args(argv)
    sources = [parse_score(text) for text in args.runs]
    names = [name for name, _, _ in sources]
    if len(names) < 3 or len(set(names)) != len(names):
        raise ValueError("Give at least three detectors with different names")
    votes_needed = len(names) // 2 + 1 if args.votes is None else args.votes
    if not 1 <= votes_needed <= len(names):
        raise ValueError(f"votes must be between 1 and {len(names)}")
    if args.vote_dir is not None and args.vote_dir.exists() and any(args.vote_dir.iterdir()):
        raise ValueError(f"Use a new/empty vote directory: {args.vote_dir}")
    names, totals, timestamps = compare(sources, args.top_percent, votes_needed, args.vote_dir)
    values, flagged, pairs = totals["values"], totals["flagged"], totals["pairs"]
    with np.errstate(invalid="ignore", divide="ignore"):
        overlap = pairs / flagged[:, None]  # share of the flags of the row also flagged by the column
        jaccard = pairs / (flagged[:, None] + flagged[None, :] - pairs)
    print(f"[compare] {timestamps[0].date()} to {timestamps[-1].date()} | {len(names)} detectors, each its highest "
          f"{args.top_percent:g}% | {values} values scored by all | majority: {votes_needed} of {len(names)}")
    print(f"\n{'':<{NAME}}{'flagged %':>11}{'with majority %':>17}{'alone %':>10}")
    for index, name in enumerate(names):
        print(f"{name:<{NAME}}{100 * flagged[index] / values:>11.2f}"
              f"{100 * totals['with_majority'][index] / max(flagged[index], 1):>17.1f}"
              f"{100 * totals['alone'][index] / max(flagged[index], 1):>10.1f}")
    for title, matrix in (("share of the flags of the row also flagged by the column, %", 100 * overlap),
                          ("Jaccard index of the flags, %", 100 * jaccard)):
        print(f"\n{title}")
        print(f"{'':<{NAME}}" + "".join(f"{name[:9]:>10}" for name in names))
        for index, name in enumerate(names):
            print(f"{name:<{NAME}}" + "".join("         -" if index == other else f"{matrix[index, other]:>10.1f}"
                                              for other in range(len(names))))
    histogram = totals["histogram"]
    print("\nvalues flagged by that many detectors, %")
    print("".join(f"{count:>8}" for count in range(len(names) + 1)))
    print("".join(f"{100 * share / values:>8.2f}" for share in histogram))
    print(f"majority vote (at least {votes_needed}): {100 * histogram[votes_needed:].sum() / values:.2f}% of the values")
    if args.figure is not None:
        plot(args.figure, names, jaccard)
    return 0


if __name__ == "__main__":
    sys.exit(main())
