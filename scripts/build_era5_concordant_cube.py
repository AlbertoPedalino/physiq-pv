"""The concordant cube of the detectors and the list of its clusters (Simone's "gold anomalies", not a truth).

Every detector (`--runs NAME=DIR` or `NAME=DIR@component`) flags its highest `--top-percent` with the threshold of
compare_era5_detectors.py (that of the validation years when the run names one); only the values every detector scores
are read. The holes of the flags are closed with a binary closing, dilation then erosion, by an element of
(2 `--closure-steps` + 1) timestamps x (2 `--closure-cells` + 1) x (2 `--closure-cells` + 1) cells: on the cube of every
detector before the vote (`--closure before`, the rule of the plan), on the concordant cube after it (`after`) or not at
all (`none`). The element is cut at the edge of the domain and of the period (what lies outside counts as not flagged,
on a padding cut away afterwards), so the closing never removes a flag and adds none outside; it does not cross a gap in
the timestamps. A value is concordant when at
least `--votes` detectors flag it.

The clusters of the concordant cube join, as in the thesis, the concordant cells touching at a timestamp (diagonals
included) and, at consecutive timestamps, the patches sharing at least one cell (`--overlap 0`) or at least that share of
the smaller patch (`--overlap`; 0 by default, as in the plan; scripts/check_era5_cluster_rules.py). Each cluster gets its box [t0, t1] x rows x columns and the box
padded by `--pad-cells` and `--pad-hours`, cut at the edge of the domain and of the period; it is analysed when it holds
at least `--min-values` values and lasts at most `--max-hours` (t1 - t0), small when it holds fewer, long otherwise.

Written in OUTPUT (new or empty):
- votes/: the number of detectors flagging each value after the closing, as a run (scores/anomaly_mean.npy, NaN where a
  detector does not score), with the vote in metadata.json, as compare_era5_detectors.py --vote-dir;
- flags/NAME.npy: the flags of every detector after its closing, bits packed along the cells (np.unpackbits(..., axis=1,
  count=cells)), for the measures in the boxes;
- clusters.csv: one row per cluster with its values, timestamps, rows, columns, degrees, padded box and kind;
- summary.json and the printed summary: the growth of every detector with the closing, the share of concordant values
  with and without the closing (without: the majority vote of the thesis, Section 6.6), the share of the cube where each
  detector is the deciding vote (exactly `--votes` votes, one of them its own: the cube without its vote lacks them), the
  clusters by kind and the share of concordant values they hold.

    python scripts/build_era5_concordant_cube.py --output outputs/concordate_5pc --runs Gn=outputs/optuna27/test_generator_clim_day \\
        normal=outputs/confronti_good2/climatology/test/normal change_3h=outputs/confronti_good2/climatology/test/change_3h ...
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
from scipy import ndimage

from physiq_pv.anomaly_detection.stgan.loading import release_pages
from scripts.check_era5_cluster_rules import clusters, patches
from scripts.check_era5_dust_lead import score_file, score_limit
from scripts.check_era5_score_correlation import parse_score

CHUNK = 256  # timestamps read at a time


def segments(timestamps):
    """[start, stop) of the runs of regularly spaced timestamps (the closing does not cross a gap)."""
    steps = np.diff(timestamps.asi8)
    if len(steps) == 0:
        return [(0, len(timestamps))]
    step = np.median(steps)
    cuts = np.flatnonzero(steps != step) + 1
    bounds = [0, *cuts.tolist(), len(timestamps)]
    return list(zip(bounds[:-1], bounds[1:]))


def close(lattice, steps, cells, spans):
    """Binary closing of a cube [time, row, column] with a box element of +-steps x +-cells x +-cells, cut at the edges:
    each run of timestamps is padded with unflagged values as wide as the element, closed and cut back, so that the
    closing contains the input (every value lies inside the dilation of itself) and adds nothing outside the domain or
    across a gap in the timestamps."""
    if steps == 0 and cells == 0:
        return lattice.copy()
    element = np.ones((2 * steps + 1, 2 * cells + 1, 2 * cells + 1), dtype=bool)
    width = ((steps, steps), (cells, cells), (cells, cells))
    closed = np.zeros_like(lattice)
    for start, stop in spans:
        grown = ndimage.binary_dilation(np.pad(lattice[start:stop], width), structure=element)
        shrunk = ndimage.binary_erosion(grown, structure=element)
        del grown
        closed[start:stop] = shrunk[steps:shrunk.shape[0] - steps, cells:shrunk.shape[1] - cells,
                                    cells:shrunk.shape[2] - cells]
        del shrunk
    return closed


def boxes(ids, count, overlap, rows, cols):
    """Per cluster: values, first and last timestamp, lowest and highest row and column (indices)."""
    of_patch = clusters(ids, count, overlap)
    n = int(of_patch.max()) + 1 if count else 0
    big = np.iinfo(np.int64).max
    table = {"values": np.zeros(n, dtype=np.int64), "t0": np.full(n, big), "t1": np.full(n, -1),
             "row0": np.full(n, big), "row1": np.full(n, -1), "col0": np.full(n, big), "col1": np.full(n, -1)}
    for t in range(len(ids)):
        cells = np.flatnonzero(ids[t] >= 0)
        if len(cells) == 0:
            continue
        cluster = of_patch[ids[t, cells]]
        np.add.at(table["values"], cluster, 1)
        np.minimum.at(table["t0"], cluster, t)
        np.maximum.at(table["t1"], cluster, t)
        np.minimum.at(table["row0"], cluster, rows[cells])
        np.maximum.at(table["row1"], cluster, rows[cells])
        np.minimum.at(table["col0"], cluster, cols[cells])
        np.maximum.at(table["col1"], cluster, cols[cells])
    return pd.DataFrame(table)


def describe_clusters(table, timestamps, shape, latitudes, longitudes, pad_steps, pad_cells, step_hours, min_values,
                      max_hours):
    """The list of clusters with times, degrees, padded box and kind."""
    table = table.copy()
    table.insert(0, "cluster", np.arange(1, len(table) + 1))
    table["hours"] = (table["t1"] - table["t0"]) * step_hours
    table["start"] = timestamps[table["t0"]].strftime("%Y-%m-%d %H:%M")
    table["end"] = timestamps[table["t1"]].strftime("%Y-%m-%d %H:%M")
    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    table["lat_min"] = np.minimum(lat[table["row0"]], lat[table["row1"]])
    table["lat_max"] = np.maximum(lat[table["row0"]], lat[table["row1"]])
    table["lon_min"] = np.minimum(lon[table["col0"]], lon[table["col1"]])
    table["lon_max"] = np.maximum(lon[table["col0"]], lon[table["col1"]])
    table["pad_t0"] = np.maximum(table["t0"] - pad_steps, 0)
    table["pad_t1"] = np.minimum(table["t1"] + pad_steps, len(timestamps) - 1)
    table["pad_row0"] = np.maximum(table["row0"] - pad_cells, 0)
    table["pad_row1"] = np.minimum(table["row1"] + pad_cells, shape[0] - 1)
    table["pad_col0"] = np.maximum(table["col0"] - pad_cells, 0)
    table["pad_col1"] = np.minimum(table["col1"] + pad_cells, shape[1] - 1)
    table["kind"] = np.where(table["values"] < min_values, "small",
                             np.where(table["hours"] <= max_hours, "analysed", "long"))
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", required=True, help="NAME=DIR or NAME=DIR@component of every detector")
    parser.add_argument("--output", type=Path, required=True, help="New or empty directory")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--votes", type=int, help="Detectors needed; default more than half")
    parser.add_argument("--closure", choices=("before", "after", "none"), default="before")
    parser.add_argument("--closure-cells", type=int, default=1, help="Half width of the element in cells")
    parser.add_argument("--closure-steps", type=int, default=1, help="Half length of the element in timestamps")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="Share of the smaller patch shared to join consecutive timestamps (0, the plan: any cell, the "
                             "rule of Section 6.7 and the default of TempestExtremes, Ullrich et al. 2021)")
    parser.add_argument("--pad-cells", type=int, default=5)
    parser.add_argument("--pad-hours", type=float, default=24)
    parser.add_argument("--min-values", type=int, default=20)
    parser.add_argument("--max-hours", type=float, default=72)
    args = parser.parse_args(argv)
    sources = [parse_score(text) for text in args.runs]
    names = [name for name, _, _ in sources]
    votes_needed = len(names) // 2 + 1 if args.votes is None else args.votes
    if not 1 <= votes_needed <= len(names):
        raise ValueError(f"votes must be between 1 and {len(names)}")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Use a new/empty output directory: {args.output}")

    manifests = {name: json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
                 for name, directory, _ in sources}
    grid = manifests[names[0]]["grid"]
    timestamps = pd.DatetimeIndex(np.load(sources[0][1] / "test_timestamps.npy"))
    for name, directory, _ in sources[1:]:
        if manifests[name]["grid"] != grid or not timestamps.equals(pd.DatetimeIndex(np.load(directory / "test_timestamps.npy"))):
            raise ValueError(f"{name} is not over the grid and timestamps of {names[0]}")
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    spans = segments(timestamps)
    step_hours = float(np.median(np.diff(timestamps.asi8))) / 3.6e12
    pad_steps = int(round(args.pad_hours / step_hours))
    times = len(timestamps)
    print(f"[concordant] {times} timestamps every {step_hours:g} h in {len(spans)} run(s), {len(rows)} cells, "
          f"{len(names)} detectors, highest {args.top_percent:g}%, at least {votes_needed} votes, closing {args.closure} "
          f"(+-{args.closure_steps} timestamps, +-{args.closure_cells} cells)", flush=True)

    arrays = {name: np.load(directory / score_file(manifests[name], component), mmap_mode="r")
              for name, directory, component in sources}
    limits = {name: score_limit(array, args.top_percent) for name, array in arrays.items()}
    valid = np.ones((times, len(rows)), dtype=bool)  # the values every detector scores
    for array in arrays.values():
        for start in range(0, times, CHUNK):
            valid[start:start + CHUNK] &= np.isfinite(np.asarray(array[start:start + CHUNK]))
        release_pages(array)
    valid_lattice = np.zeros((times, *shape), dtype=bool)
    valid_lattice[:, rows, cols] = valid
    values = int(valid.sum())

    (args.output / "flags").mkdir(parents=True)
    votes = np.zeros((times, *shape), dtype=np.uint8)
    raw_votes = np.zeros((times, *shape), dtype=np.uint8)
    per_detector = {}
    try:
        for name in names:
            flagged = np.zeros((times, *shape), dtype=bool)
            with np.errstate(invalid="ignore"):
                for start in range(0, times, CHUNK):
                    block = np.asarray(arrays[name][start:start + CHUNK], dtype=np.float64) > limits[name]
                    flagged[start:start + CHUNK][:, rows, cols] = block & valid[start:start + CHUNK]
            release_pages(arrays[name])
            raw_votes += flagged
            raw = int(flagged.sum())
            if args.closure == "before":
                flagged = close(flagged, args.closure_steps, args.closure_cells, spans) & valid_lattice
            votes += flagged
            kept = int(flagged.sum())
            per_detector[name] = {"threshold": limits[name], "flagged_raw": raw, "flagged_closed": kept,
                                  "growth_percent": 100 * (kept - raw) / max(raw, 1)}
            np.save(args.output / "flags" / f"{name}.npy", np.packbits(flagged[:, rows, cols], axis=1))
            print(f"[concordant] {name}: {100 * raw / values:.2f}% flagged, {100 * kept / values:.2f}% after the closing",
                  flush=True)
            del flagged
    finally:
        for array in arrays.values():
            array._mmap.close()

    concordant = votes >= votes_needed
    raw_concordant = int((raw_votes >= votes_needed).sum())
    del raw_votes
    if args.closure == "after":
        concordant = close(concordant, args.closure_steps, args.closure_cells, spans) & valid_lattice
    total = int(concordant.sum())
    for name in names:  # deciding votes: concordant with exactly the votes needed, one of them this detector's
        packed = np.load(args.output / "flags" / f"{name}.npy", mmap_mode="r")
        deciding = 0
        for start in range(0, times, CHUNK):
            own = np.unpackbits(np.asarray(packed[start:start + CHUNK]), axis=1, count=len(rows)).astype(bool)
            exact = (votes[start:start + CHUNK][:, rows, cols] == votes_needed) & concordant[start:start + CHUNK][:, rows, cols]
            deciding += int((own & exact).sum())
        per_detector[name]["deciding_percent_of_cube"] = 100 * deciding / max(total, 1)

    run = args.output / "votes"
    (run / "scores").mkdir(parents=True)
    written = np.lib.format.open_memmap(run / "scores" / "anomaly_mean.npy", mode="w+", dtype=np.float32,
                                        shape=(times, len(rows)))
    for start in range(0, times, CHUNK):
        block = votes[start:start + CHUNK][:, rows, cols].astype(np.float32)
        block[~valid[start:start + CHUNK]] = np.nan
        written[start:start + CHUNK] = block
    written.flush()
    del written
    np.save(run / "test_timestamps.npy", timestamps.as_unit("ns").asi8)
    vote = {"detectors": {name: f"{directory}@{component}" for name, directory, component in sources},
            "top_percent": args.top_percent, "thresholds": limits, "votes_needed": votes_needed,
            "closure": {"where": args.closure, "steps": args.closure_steps, "cells": args.closure_cells}}
    (run / "metadata.json").write_text(json.dumps(
        {"status": "complete", "grid": grid, "score_kind": "majority_vote_count", "vote": vote,
         "anomaly_mean_file": "scores/anomaly_mean.npy", "anomaly_std_file": None}, indent=2), encoding="utf-8")
    del votes

    ids, patch_times, count = patches(concordant[:, rows, cols], rows, cols, shape, 8)
    del concordant
    table = describe_clusters(boxes(ids, count, args.overlap, rows, cols), timestamps, shape, grid["latitudes"],
                              grid["longitudes"], pad_steps, args.pad_cells, step_hours, args.min_values, args.max_hours)
    del ids
    table.to_csv(args.output / "clusters.csv", index=False)
    kinds = {kind: {"clusters": int((table["kind"] == kind).sum()),
                    "values_percent": 100 * float(table.loc[table["kind"] == kind, "values"].sum()) / max(total, 1)}
             for kind in ("analysed", "small", "long")}
    summary = {"values": values, "vote": vote, "overlap": args.overlap,
               "padding": {"cells": args.pad_cells, "hours": args.pad_hours, "timestamps": pad_steps},
               "selection": {"min_values": args.min_values, "max_hours": args.max_hours},
               "concordant_percent": 100 * total / values, "concordant_without_closing_percent": 100 * raw_concordant / values,
               "detectors": per_detector, "clusters": kinds,
               "longest_hours": float(table["hours"].max()) if len(table) else 0.0}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[concordant] concordant values: {summary['concordant_percent']:.2f}% after the closing, "
          f"{summary['concordant_without_closing_percent']:.2f}% without (majority vote of Section 6.6)")
    detectors = pd.DataFrame(per_detector).T[["flagged_raw", "flagged_closed", "growth_percent", "deciding_percent_of_cube"]]
    detectors[["flagged_raw", "flagged_closed"]] = 100 * detectors[["flagged_raw", "flagged_closed"]] / values
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print("\n[concordant] per detector: flagged % of the values before / after the closing, growth %, "
              "deciding vote % of the cube")
        print(detectors.to_string())
        print("\n[concordant] clusters")
        print(pd.DataFrame(kinds).T.to_string())
    print(f"longest cluster: {summary['longest_hours'] / 24:.1f} days")
    print(f"\nwritten {args.output}")


if __name__ == "__main__":
    main()
