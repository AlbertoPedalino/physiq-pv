"""How much of every concordant cluster each detector flags, inside the affiliation zone of its padded box.

Reads the output of build_era5_concordant_cube.py (`--cube`): the votes, the flags of every detector after its closing
and the list of clusters. The clusters are labelled again with the rule written in its summary (same function, so the
same numbers; checked against the list). For every cluster of `--kind` (the analysed ones by default) the zone is the
cluster dilated by the padding (cells at every timestamp, as the buffer of Lagerquist et al. 2019 and the radius at every
timestamp of Bourdin et al. 2022; and timestamps before and after), among the values every detector scores, keeping the
values nearer to it than to any other concordant cluster of any kind: the affiliation zones of Huet, Navarro & Rossi (KDD
2022, Sec. 3.2), there in time, here in time and space, with the distance in units of the padding (the padding in cells
and in timestamps both count 1). Each value is so counted for one cluster only; the concordant values inside a zone are
those of its cluster. The padded box only bounds the zone (the box is a descriptor of the cluster, as in Clark et al. 2014).

For every cluster and detector, with M the flags of the detector and C the concordant cube, inside the zone Z:
intersection I = |M & C & Z|, union U = |(M | C) & Z|, Jaccard I / U, coverage I / |C & Z|, excess |M & Z| - I; the same
against the cube without the vote of the detector (at least the votes needed among the others), which differs from C
only where the detector is the deciding vote. With `--inputs-run` (the scored run of the model, for its standardised
inputs) every cluster gets a type as in Section 6.7 (check_era5_event_zones.event_types: the input variable farthest from
its normal, on average over the values of the cluster).

Printed: per detector the median and quartiles over the clusters of every measure, with and without its vote; the median
Jaccard by type and by size of cluster; how much of the padded boxes is shared with other boxes and how much of each box is
left in its zone. `--csv` keeps one row per cluster and detector.

    python scripts/check_era5_concordant_boxes.py --cube outputs/concordate_5pc --inputs-run outputs/stgan_optuna27_scored \\
        --csv outputs/concordate_5pc/boxes.csv
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
from scripts.check_era5_event_zones import event_types, inputs_of

CHUNK = 256  # timestamps read at a time
BANDS = ((20, 99), (100, 999), (1000, 9999), (10000, None))  # sizes of cluster, as the bands of Section 6.7 from 20 on
MEASURES = ("intersection", "union", "jaccard", "coverage", "excess")


def zone(labels, cluster, pad_steps, pad_cells):
    """The zone of `cluster` in `labels` (a block [time, row, column] of cluster numbers, 0 where not concordant): the
    cluster dilated by +-pad_cells cells at every timestamp and by +-pad_steps timestamps (a box element: a buffer that
    follows the shape), keeping the cells nearer to it than to any other cluster, the distance being Euclidean with the
    timestamps divided by `pad_steps` and the cells by `pad_cells`. Ties go to the cluster whose cell comes first in the
    block (scipy's choice); the cells of a cluster are always its own."""
    own = (labels == cluster).astype(np.uint8)
    reach = ndimage.maximum_filter(own, size=(2 * pad_steps + 1, 2 * pad_cells + 1, 2 * pad_cells + 1), mode="constant",
                                   cval=0).astype(bool)  # separable, so fast also with a large element
    sampling = (1 / max(pad_steps, 1), 1 / max(pad_cells, 1), 1 / max(pad_cells, 1))
    nearest = ndimage.distance_transform_edt(labels == 0, sampling=sampling, return_distances=False, return_indices=True)
    return reach & (labels[tuple(nearest)] == cluster)


def measures(own, concordant, inside):
    """Intersection, union, Jaccard, coverage and excess of the flags `own` with `concordant` inside `inside`."""
    intersection = int((own & concordant & inside).sum())
    union = int(((own | concordant) & inside).sum())
    reference = int((concordant & inside).sum())
    flagged = int((own & inside).sum())
    return {"intersection": intersection, "union": union, "jaccard": intersection / union if union else np.nan,
            "coverage": intersection / reference if reference else np.nan, "excess": flagged - intersection,
            "concordant": reference, "flagged": flagged}


def band_of(values):
    for low, high in BANDS:
        if values >= low and (high is None or values <= high):
            return f"{low}-{high}" if high else f">={low}"
    return f"<{BANDS[0][0]}"


def label_clusters(concordant, rows, cols, shape, overlap):
    """Cluster number of every value [time, row, column] (0 where not concordant) and the values of every cluster."""
    ids, _, count = patches(concordant, rows, cols, shape, 8)
    of_patch = clusters(ids, count, overlap)
    n = int(of_patch.max()) + 1 if count else 0
    lattice = np.zeros((len(concordant), *shape), dtype=np.int32)
    sizes = np.zeros(n + 1, dtype=np.int64)
    for start in range(0, len(ids), CHUNK):
        block = ids[start:start + CHUNK]
        steps, cells = np.nonzero(block >= 0)
        numbers = of_patch[block[steps, cells]] + 1
        lattice[start + steps, rows[cells], cols[cells]] = numbers
        sizes += np.bincount(numbers, minlength=n + 1)
    return lattice, sizes[1:]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cube", type=Path, required=True, help="Output directory of build_era5_concordant_cube.py")
    parser.add_argument("--kind", choices=("analysed", "small", "long", "all"), default="analysed")
    parser.add_argument("--inputs-run", type=Path, help="Scored run of the model, for the type of every cluster")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    summary = json.loads((args.cube / "summary.json").read_text(encoding="utf-8"))
    names = list(summary["vote"]["detectors"])
    needed = summary["vote"]["votes_needed"]
    pad_steps, pad_cells = summary["padding"]["timestamps"], summary["padding"]["cells"]
    run = args.cube / "votes"
    grid = json.loads((run / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    timestamps = pd.DatetimeIndex(np.load(run / "test_timestamps.npy"))
    table = pd.read_csv(args.cube / "clusters.csv")
    votes = np.load(run / "scores" / "anomaly_mean.npy", mmap_mode="r")
    times = len(timestamps)

    concordant = np.zeros((times, len(rows)), dtype=bool)
    with np.errstate(invalid="ignore"):
        for start in range(0, times, CHUNK):
            concordant[start:start + CHUNK] = np.asarray(votes[start:start + CHUNK]) >= needed
    release_pages(votes)
    labels, sizes = label_clusters(concordant, rows, cols, shape, summary["overlap"])
    if len(sizes) != len(table) or not np.array_equal(sizes, table["values"].to_numpy()):
        raise ValueError("The clusters labelled again differ from clusters.csv: not the same cube or rule")
    print(f"[boxes] {len(table)} clusters labelled again, as in clusters.csv", flush=True)

    types = np.full(len(table), "", dtype=object)
    if args.inputs_run is not None:
        inputs, first, variables = inputs_of(args.inputs_run, timestamps)
        steps, cells = np.nonzero(concordant)
        events = labels[steps, rows[cells], cols[cells]].astype(np.int64) - 1
        kinds = event_types(steps, cells, events, len(table), inputs, first)
        types = np.array([variables[k] if variables else str(k) for k in kinds], dtype=object)
        del steps, cells, events
        inputs._mmap.close()
    del concordant
    table["type"] = types
    table["band"] = [band_of(v) for v in table["values"]]

    chosen = table if args.kind == "all" else table[table["kind"] == args.kind]
    flags = {name: np.load(args.cube / "flags" / f"{name}.npy", mmap_mode="r") for name in names}
    covered = np.zeros((times, *shape), dtype=bool)
    box_volume = zone_volume = 0
    out = []
    for done, row in enumerate(chosen.itertuples(index=False), 1):
        t0, t1, r0, r1, c0, c1 = row.pad_t0, row.pad_t1, row.pad_row0, row.pad_row1, row.pad_col0, row.pad_col1
        # The region searched for nearer clusters: the box padded once more (a value of the box can be nearer to a
        # cluster outside it, never farther than the padding beyond it).
        e = (max(t0 - pad_steps, 0), min(t1 + pad_steps, times - 1), max(r0 - pad_cells, 0),
             min(r1 + pad_cells, shape[0] - 1), max(c0 - pad_cells, 0), min(c1 + pad_cells, shape[1] - 1))
        near = zone(labels[e[0]:e[1] + 1, e[2]:e[3] + 1, e[4]:e[5] + 1], row.cluster, pad_steps, pad_cells)
        near = near[t0 - e[0]:t1 - e[0] + 1, r0 - e[2]:r1 - e[2] + 1, c0 - e[4]:c1 - e[4] + 1]
        block = np.full((t1 - t0 + 1, *shape), np.nan, dtype=np.float32)
        block[:, rows, cols] = np.asarray(votes[t0:t1 + 1])
        block = block[:, r0:r1 + 1, c0:c1 + 1]
        valid = np.isfinite(block)
        inside = near & valid
        reference = (block >= needed) & valid
        if int((reference & inside).sum()) != row.values:
            raise ValueError(f"cluster {row.cluster}: its zone does not hold exactly its values")
        covered[t0:t1 + 1, r0:r1 + 1, c0:c1 + 1] |= valid
        box_volume += int(valid.sum())
        zone_volume += int(inside.sum())
        for name in names:
            own = np.zeros((t1 - t0 + 1, *shape), dtype=bool)
            own[:, rows, cols] = np.unpackbits(np.asarray(flags[name][t0:t1 + 1]), axis=1, count=len(rows)).astype(bool)
            own = own[:, r0:r1 + 1, c0:c1 + 1]
            result = {"cluster": row.cluster, "detector": name, "type": row.type, "band": row.band, "values": row.values,
                      "hours": row.hours, "zone": int(inside.sum()), **measures(own, reference, inside)}
            others = ((block - own) >= needed) & valid  # the cube without this detector's vote
            result.update({f"{key}_without_own": value for key, value in measures(own, others, inside).items()})
            out.append(result)
        if done % 2000 == 0:
            release_pages(votes, *flags.values())
            print(f"[boxes] {done} of {len(chosen)} clusters", flush=True)
    for array in (votes, *flags.values()):
        array._mmap.close()
    results = pd.DataFrame(out)
    shared = 100 * (1 - covered.sum() / max(box_volume, 1))
    print(f"\n[boxes] {len(chosen)} clusters ({args.kind}), padding +-{pad_cells} cells, +-{pad_steps} timestamps; "
          f"{shared:.1f}% of the values of the padded boxes lie in more than one box; the zones (cluster dilated by the "
          f"padding, nearest cluster) hold {100 * zone_volume / max(box_volume, 1):.1f}% of the boxes")
    if results.empty:
        return
    quartiles = lambda s: f"{s.quantile(0.5):.3g} ({s.quantile(0.25):.3g}-{s.quantile(0.75):.3g})"
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for suffix, title in (("", "against the concordant cube"), ("_without_own", "against the cube without its vote")):
            print(f"\n[boxes] median (quartiles) over the clusters, {title}")
            print(results.groupby("detector", sort=False)[[m + suffix for m in MEASURES]].agg(quartiles).to_string())
        print("\n[boxes] median Jaccard by type of cluster (clusters in brackets in the first column)")
        by_type = results.pivot_table(index="type", columns="detector", values="jaccard", aggfunc="median")[names]
        by_type.insert(0, "clusters", results[results["detector"] == names[0]].groupby("type").size())
        print(by_type.round(3).to_string())
        print("\n[boxes] median Jaccard by size of cluster")
        by_band = results.pivot_table(index="band", columns="detector", values="jaccard", aggfunc="median")[names]
        by_band.insert(0, "clusters", results[results["detector"] == names[0]].groupby("band").size())
        print(by_band.round(3).to_string())
    if args.csv:
        results.to_csv(args.csv, index=False)
        print(f"\nwritten {args.csv}")


if __name__ == "__main__":
    main()
