"""How much of the chains of clusters comes from the rule that joins the flags?

Each score flags its highest `--top-percent` (threshold of the validation years when the run names it). At every
timestamp the flagged cells touching each other make patches, with 8 neighbours (diagonals included) or 4. Patches of
consecutive timestamps are joined when they share cells: any one cell (the rule of the thesis, with 8 neighbours), or
at least a share `--overlap` of the smaller patch, or never (patches of one timestamp only, a lower bound). A cluster
is a set of joined patches. For every rule and score the check gives the number of clusters, the share of the flagged
values in clusters of fewer than 20, at least 20 and at least 1000 values, the share in clusters lasting more than
`--long-days` days and the longest cluster. With 8 neighbours and any shared cell the clusters are those of
check_era5_event_agreement.events_of (ndimage.label with the same structure), so the first rule must give the numbers
of the thesis.

    python scripts/check_era5_cluster_rules.py --scores Gn=outputs/optuna27/test_generator_clim_day \\
        mcd=outputs/confronti_good2/climatology/test/mcd
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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from scripts.check_era5_lead import flags

RULES = (("8 neighbours, any shared cell (thesis)", 8, 0.0), ("8 neighbours, overlap >= 25%", 8, 0.25),
         ("8 neighbours, overlap >= 50%", 8, 0.5), ("8 neighbours, no link in time", 8, None),
         ("4 neighbours, any shared cell", 4, 0.0))  # grouped by neighbours: one patch map in memory at a time
STRUCTURES = {8: np.ones((3, 3), dtype=bool), 4: ndimage.generate_binary_structure(2, 1)}


def patches(flagged, rows, cols, shape, neighbours):
    """Patch id of every flagged value [time, cell] (-1 where not flagged), ids unique over the whole period, and the
    timestamp of every patch."""
    ids = np.full(flagged.shape, -1, dtype=np.int32)  # 4 bytes a value: the whole test period fits in a few GB
    times = []
    offset = 0
    lattice = np.zeros(shape, dtype=bool)
    for t in range(len(flagged)):
        lattice[:] = False
        lattice[rows[flagged[t]], cols[flagged[t]]] = True
        labels, count = ndimage.label(lattice, structure=STRUCTURES[neighbours])
        if count:
            cells = np.flatnonzero(flagged[t])
            ids[t, cells] = labels[rows[cells], cols[cells]] - 1 + offset
            times.append(np.full(count, t, dtype=np.int64))
            offset += count
    return ids, (np.concatenate(times) if times else np.zeros(0, dtype=np.int64)), offset


def clusters(ids, count, overlap):
    """Cluster of every patch, joining patches of consecutive timestamps that share cells: any (overlap 0), at least
    `overlap` of the smaller patch, or none (overlap None)."""
    if overlap is None or count == 0:
        return np.arange(count)
    sizes = np.bincount(ids[ids >= 0], minlength=count)
    sources, targets = [], []
    for t in range(len(ids) - 1):
        both = (ids[t] >= 0) & (ids[t + 1] >= 0)
        if not both.any():
            continue
        pairs, shared = np.unique(np.stack((ids[t][both], ids[t + 1][both])), axis=1, return_counts=True)
        keep = shared >= overlap * np.minimum(sizes[pairs[0]], sizes[pairs[1]]) if overlap > 0 else np.ones(len(shared), bool)
        sources.append(pairs[0][keep])
        targets.append(pairs[1][keep])
    sources = np.concatenate(sources) if sources else np.zeros(0, dtype=np.int64)
    targets = np.concatenate(targets) if targets else np.zeros(0, dtype=np.int64)
    graph = coo_matrix((np.ones(len(sources), dtype=np.int8), (sources, targets)), shape=(count, count))
    return connected_components(graph, directed=False)[1]


def describe(ids, times, count, overlap, step_hours, long_days):
    """Shares of the flagged values by size and duration of their cluster, for one rule."""
    of_patch = clusters(ids, count, overlap)
    patch_sizes = np.bincount(ids[ids >= 0], minlength=count)
    n = int(of_patch.max()) + 1 if count else 0
    values = np.bincount(of_patch, weights=patch_sizes, minlength=n)
    first = np.full(n, np.iinfo(np.int64).max)
    last = np.full(n, -1)
    np.minimum.at(first, of_patch, times)
    np.maximum.at(last, of_patch, times)
    days = (last - first + 1) * step_hours / 24
    total = max(values.sum(), 1)
    return {"clusters": n,
            "values in clusters < 20 %": 100 * values[values < 20].sum() / total,
            "in clusters >= 20 %": 100 * values[values >= 20].sum() / total,
            "in clusters >= 1000 %": 100 * values[values >= 1000].sum() / total,
            f"in clusters longer than {long_days:g} days %": 100 * values[days > long_days].sum() / total,
            "longest (days)": float(days.max()) if n else 0.0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scores", nargs="+", required=True, help="NAME=DIR of each score")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--long-days", type=float, default=3)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    rows_out = []
    for text in args.scores:
        name, _, directory = text.partition("=")
        directory = Path(directory)
        flagged, timestamps, _ = flags(directory, args.top_percent)
        grid = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))["grid"]
        rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
        shape = (len(grid["latitudes"]), len(grid["longitudes"]))
        step_hours = (timestamps[1] - timestamps[0]) / pd.Timedelta(hours=1)
        built = {}
        for label, neighbours, overlap in RULES:
            if neighbours not in built:
                built.clear()  # free the patch map of the other neighbourhood first
                built[neighbours] = patches(flagged, rows, cols, shape, neighbours)
            ids, times, count = built[neighbours]
            row = {"score": name, "rule": label, **describe(ids, times, count, overlap, step_hours, args.long_days)}
            rows_out.append(row)
            print(f"[cluster-rules] {name}: {label} done", flush=True)
        del built, flagged
    table = pd.DataFrame(rows_out)
    print(f"\n[cluster-rules] flagged: the highest {args.top_percent:g}% of each score; a cluster joins touching cells at a "
          f"timestamp and patches of consecutive timestamps by the rule")
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.1f}".format):
        print(table.to_string(index=False))
    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nwritten {args.csv}")


if __name__ == "__main__":
    main()
