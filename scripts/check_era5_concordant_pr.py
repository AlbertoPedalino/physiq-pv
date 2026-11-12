"""Precision and recall of every detector against the concordant cube, as its own threshold varies (thesis, Appendix A).

The cube (`--cube`, the output of build_era5_concordant_cube.py) stays fixed. Every detector of the cube (or those of
`--detectors`) flags, at each of `--percentiles`, the values above that percentile of its score, taken as in the cube
(score_limit: over the validation years when the run names a threshold reference); its flags are closed as in the
cube (only when the cube closed every detector before the vote) and kept where every detector scores. Against the
concordant values C:

    precision = |M & C| / |M|,    recall = |M & C| / |C|,

and the same against the cube without the detector's own vote: the values that at least the votes needed of the other
detectors flag, its own vote being its flags at the threshold of the cube (flags/NAME.npy). The percentile of the cube
is always added; there the flags must equal flags/NAME.npy, or the script stops (not the same runs, threshold or
closing). The flags are nested (a higher threshold flags a subset, and the closing keeps the order), so the recall
never grows with the percentile.

    python scripts/check_era5_concordant_pr.py --cube outputs/concordate_5pc_regola_tesi \\
        --csv outputs/concordate_5pc_regola_tesi/pr.csv
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
from scripts.build_era5_concordant_cube import CHUNK, close, segments
from scripts.check_era5_dust_lead import score_file, score_limit
from scripts.check_era5_score_correlation import parse_score

PERCENTILES = (80, 85, 90, 92.5, 95, 97, 98, 99, 99.5, 99.9)


def exceeded(array, limits, valid):
    """Per value, how many of the increasing `limits` the score exceeds (0 where not every detector scores)."""
    times = len(valid)
    count = np.zeros(valid.shape, dtype=np.uint8)
    bounds = np.asarray(limits, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        for start in range(0, times, CHUNK):
            block = np.asarray(array[start:start + CHUNK], dtype=np.float64)
            count[start:start + CHUNK] = (block[..., None] > bounds).sum(axis=-1)
    release_pages(array)
    count[~valid] = 0
    return count


def measures(flagged, reference):
    """Flagged, reference and shared values; precision and recall (NaN when nothing is flagged)."""
    shared = int(np.count_nonzero(flagged & reference))
    n_flagged, n_reference = int(np.count_nonzero(flagged)), int(np.count_nonzero(reference))
    return {"flagged": n_flagged, "reference": n_reference, "shared": shared,
            "precision": shared / n_flagged if n_flagged else np.nan,
            "recall": shared / n_reference if n_reference else np.nan}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cube", type=Path, required=True, help="Output of build_era5_concordant_cube.py")
    parser.add_argument("--percentiles", type=float, nargs="+", default=list(PERCENTILES),
                        help="Percentiles of the threshold of every detector (the one of the cube is always added)")
    parser.add_argument("--detectors", nargs="+", help="Only these detectors of the cube (default: all)")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    summary = json.loads((args.cube / "summary.json").read_text(encoding="utf-8"))
    vote = summary["vote"]
    names = list(vote["detectors"])
    chosen = args.detectors or names
    unknown = sorted(set(chosen) - set(names))
    if unknown:
        raise ValueError(f"{', '.join(unknown)} not among the detectors of the cube: {', '.join(names)}")
    needed = int(vote["votes_needed"])
    closing = vote["closure"]
    cube_percentile = 100 - float(vote["top_percent"])
    percentiles = sorted({float(p) for p in args.percentiles} | {cube_percentile})
    if not all(0 < p < 100 for p in percentiles):
        raise ValueError("percentiles must lie between 0 and 100")

    run = args.cube / "votes"
    grid = json.loads((run / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    timestamps = pd.DatetimeIndex(np.load(run / "test_timestamps.npy"))
    spans = segments(timestamps)
    times = len(timestamps)

    counted = np.load(run / "scores" / "anomaly_mean.npy", mmap_mode="r")
    votes = np.zeros((times, len(rows)), dtype=np.uint8)
    valid = np.zeros((times, len(rows)), dtype=bool)
    for start in range(0, times, CHUNK):
        block = np.asarray(counted[start:start + CHUNK])
        valid[start:start + CHUNK] = np.isfinite(block)
        votes[start:start + CHUNK] = np.where(np.isfinite(block), block, 0).astype(np.uint8)
    release_pages(counted)
    counted._mmap.close()
    concordant = valid & (votes >= needed)
    valid_lattice = np.zeros((times, *shape), dtype=bool)
    valid_lattice[:, rows, cols] = valid
    values = int(valid.sum())
    print(f"[pr] {times} timestamps, {len(rows)} cells, {values} values scored by every detector; concordant "
          f"{100 * concordant.sum() / values:.2f}% (at least {needed} of {len(names)}); percentiles "
          f"{', '.join(f'{p:g}' for p in percentiles)} (cube: {cube_percentile:g}); closing {closing['where']}",
          flush=True)

    out = []
    for name in chosen:
        source, directory, component = parse_score(f"{name}={vote['detectors'][name]}")
        manifest = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if not timestamps.equals(pd.DatetimeIndex(np.load(directory / "test_timestamps.npy"))):
            raise ValueError(f"{name}: not over the timestamps of the cube")
        array = np.load(directory / score_file(manifest, component), mmap_mode="r")
        limits = [score_limit(array, 100 - p) for p in percentiles]
        if not np.isclose(limits[percentiles.index(cube_percentile)], vote["thresholds"][name], rtol=1e-6, atol=0):
            raise ValueError(f"{name}: threshold {limits[percentiles.index(cube_percentile)]} differs from the cube's "
                             f"{vote['thresholds'][name]}")
        if any(b < a for a, b in zip(limits, limits[1:])):
            raise ValueError(f"{name}: thresholds do not grow with the percentile: {limits}")
        count = exceeded(array, limits, valid)
        array._mmap.close()
        own = np.unpackbits(np.load(args.cube / "flags" / f"{name}.npy"), axis=1, count=len(rows)).astype(bool)
        others = valid & ((votes.astype(np.int16) - own) >= needed)  # the cube without this detector's vote
        for k, (percentile, limit) in enumerate(zip(percentiles, limits)):
            lattice = np.zeros((times, *shape), dtype=bool)
            lattice[:, rows, cols] = count > k  # above limits[k], hence above every lower one
            if closing["where"] == "before":
                lattice = close(lattice, int(closing["steps"]), int(closing["cells"]), spans) & valid_lattice
            flagged = lattice[:, rows, cols]
            del lattice
            if percentile == cube_percentile and not np.array_equal(flagged, own):
                raise ValueError(f"{name}: the flags at the percentile of the cube differ from flags/{name}.npy")
            row = {"detector": name, "percentile": percentile, "threshold": limit,
                   "flagged_percent": 100 * np.count_nonzero(flagged) / values, "cube": percentile == cube_percentile}
            row.update(measures(flagged, concordant))
            row.update({f"{key}_without_own": value for key, value in measures(flagged, others).items()})
            out.append(row)
            print(f"[pr] {name} {percentile:g}: {row['flagged_percent']:.2f}% flagged, precision {row['precision']:.3f}, "
                  f"recall {row['recall']:.3f}; without its vote {row['precision_without_own']:.3f}, "
                  f"{row['recall_without_own']:.3f}", flush=True)
        del count, own, others

    table = pd.DataFrame(out)
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv, index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        for suffix, title in (("", "against the concordant cube"), ("_without_own", "against the cube without its vote")):
            print(f"\n[pr] precision / recall {title} (rows: detector, columns: percentile of its threshold)")
            print(table.pivot(index="detector", columns="percentile", values=f"precision{suffix}").loc[chosen].to_string())
            print(table.pivot(index="detector", columns="percentile", values=f"recall{suffix}").loc[chosen].to_string())
    if args.csv is not None:
        print(f"\nwritten {args.csv}")


if __name__ == "__main__":
    main()
