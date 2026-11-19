"""Where are the exclusive flags of each detector around the concordant clusters, and do they belong to the cluster?

A curiosity check, not in the thesis. Reads the output of build_era5_concordant_cube.py (`--cube`). For every concordant
cluster of `--kind` (the long ones by default) the zone is the one of the thesis (scripts/era5_km_zone.py): the cells
less than `--radius-km` from the cluster at the same timestamp, nearest cluster. An exclusive flag of a detector is a
value of the zone that it alone flags (one vote, after the closing, as in check_era5_concordant_lead.py --exclusive).

For every cluster and detector:
- the great-circle distance of every exclusive flag from the nearest cell of the cluster at the same timestamp;
- whether the flag is attached to the cluster: its group of flags of the detector (inside the box of the zone, the rule
  of the clusters of the thesis: touching cells at a timestamp, diagonals included, the same cell at consecutive
  timestamps) holds a value of the cluster. A detached flag belongs to a group of the detector that does not touch the
  cluster inside the box, another patch in the zone;
- the along-track error of all its exclusive flags (as check_era5_concordant_lead.py: the control, it must give the
  along_alone_km of that script), of the attached ones and of the detached ones.

The groups are labelled inside the box of the cluster (its timestamps, its rows and columns widened by the radius): two
patches joined only outside the box count as detached.

    python scripts/check_era5_exclusive_flags.py --cube outputs/concordate_5pc_regola_tesi --radius-km 300 \\
        --csv outputs/concordate_5pc_regola_tesi/exclusive_r300_long.csv
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
from scripts.check_era5_concordant_boxes import label_clusters
from scripts.check_era5_concordant_lead import CHUNK, EARTH_KM, JOIN, track_errors
from scripts.era5_km_zone import owner_lattice, radius_box

BANDS_KM = (0, 55, 110, 165, 220, 300, np.inf)  # about one cell of latitude each, up to the radius of the thesis


def unit_vectors(lat_rad, lon_rad):
    """Points on the unit sphere [..., 3] from latitudes and longitudes in radians (arrays broadcast)."""
    return np.stack(np.broadcast_arrays(np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad),
                                        np.sin(lat_rad)), axis=-1)


def nearest_km(points, targets, chunk=4096):
    """Great-circle distance in km from every point [n, 3] to the nearest of `targets` [m, 3] (unit vectors)."""
    out = np.empty(len(points))
    for start in range(0, len(points), chunk):
        dots = points[start:start + chunk] @ targets.T
        out[start:start + chunk] = EARTH_KM * np.arccos(np.clip(dots.max(axis=1), -1.0, 1.0))
    return out


def attached(flags, own):
    """The values of `flags` [time, row, column] whose group (the rule of the clusters) holds a cell of `own`."""
    groups, count = ndimage.label(flags, structure=JOIN)
    if count == 0:
        return np.zeros_like(flags)
    touching = np.unique(groups[own & flags])
    touching = touching[touching > 0]
    return np.isin(groups, touching) & flags


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cube", type=Path, required=True, help="Output directory of build_era5_concordant_cube.py")
    parser.add_argument("--kind", choices=("analysed", "long", "all"), default="long")
    parser.add_argument("--radius-km", type=float, default=300.0)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    summary = json.loads((args.cube / "summary.json").read_text(encoding="utf-8"))
    names = list(summary["vote"]["detectors"])
    needed = summary["vote"]["votes_needed"]
    run = args.cube / "votes"
    grid = json.loads((run / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    latitudes = np.radians(np.asarray(grid["latitudes"], dtype=np.float64))
    longitudes = np.radians(np.asarray(grid["longitudes"], dtype=np.float64))
    sphere = unit_vectors(latitudes[:, None], longitudes[None, :])  # [row, column, 3]
    timestamps = pd.DatetimeIndex(np.load(run / "test_timestamps.npy"))
    table = pd.read_csv(args.cube / "clusters.csv")
    votes = np.load(run / "scores" / "anomaly_mean.npy", mmap_mode="r")
    times = len(timestamps)
    flags = {name: np.load(args.cube / "flags" / f"{name}.npy", mmap_mode="r") for name in names}

    concordant = np.zeros((times, len(rows)), dtype=bool)
    with np.errstate(invalid="ignore"):
        for start in range(0, times, CHUNK):
            concordant[start:start + CHUNK] = np.asarray(votes[start:start + CHUNK]) >= needed
    release_pages(votes)
    labels, sizes = label_clusters(concordant, rows, cols, shape, summary["overlap"])
    del concordant
    if len(sizes) != len(table) or not np.array_equal(sizes, table["values"].to_numpy()):
        raise ValueError("The clusters labelled again differ from clusters.csv: not the same cube or rule")
    owner, reach_rows, reach_cols = owner_lattice(labels, grid, args.radius_km, CHUNK)

    chosen = table if args.kind == "all" else table[table["kind"] == args.kind]
    out = []
    for done, row in enumerate(chosen.itertuples(index=False), 1):
        t0, t1, r0, r1, c0, c1 = radius_box(row, reach_rows, reach_cols, shape)
        near = owner[t0:t1 + 1, r0:r1 + 1, c0:c1 + 1] == row.cluster
        own = labels[t0:t1 + 1, r0:r1 + 1, c0:c1 + 1] == row.cluster
        first, last = row.t0 - t0, row.t1 - t0
        count = np.full((t1 - t0 + 1, *shape), np.nan, dtype=np.float32)
        count[:, rows, cols] = np.asarray(votes[t0:t1 + 1])
        alone = count[:, r0:r1 + 1, c0:c1 + 1] == 1
        y = np.repeat(EARTH_KM * latitudes[r0:r1 + 1][:, None], c1 - c0 + 1, axis=1)
        x = EARTH_KM * longitudes[c0:c1 + 1][None, :] * np.cos(latitudes[r0:r1 + 1][:, None])
        box_sphere = sphere[r0:r1 + 1, c0:c1 + 1]
        motion = None
        centre = lambda mask: np.array([x[mask].mean(), y[mask].mean()])
        start_c, end_c = centre(own[first]), centre(own[last])
        if np.hypot(*(end_c - start_c)) > 0:
            motion = (end_c - start_c) / np.hypot(*(end_c - start_c))
        for name in names:
            block = np.zeros((t1 - t0 + 1, *shape), dtype=bool)
            block[:, rows, cols] = np.unpackbits(np.asarray(flags[name][t0:t1 + 1]), axis=1, count=len(rows)).astype(bool)
            block = block[:, r0:r1 + 1, c0:c1 + 1]
            exclusive = block & near & alone
            result = {"cluster": row.cluster, "detector": name, "values": row.values, "hours": row.hours,
                      "zone_flags": int((block & near).sum()), "exclusive": int(exclusive.sum())}
            if exclusive.any():
                linked = attached(block, own) & exclusive
                detached = exclusive & ~linked
                distances, ahead = [], 0
                for t in np.flatnonzero(exclusive.any(axis=(1, 2))):
                    if not own[t].any():
                        continue
                    distances.append(nearest_km(box_sphere[exclusive[t]], box_sphere[own[t]]))
                    if motion is not None:
                        offset = np.stack([x[exclusive[t]], y[exclusive[t]]], axis=1) - centre(own[t])
                        ahead += int((offset @ motion > 0).sum())
                distances = np.concatenate(distances) if distances else np.array([])
                result.update({
                    "attached": int(linked.sum()), "detached": int(detached.sum()),
                    "ahead_values": ahead if motion is not None else np.nan,
                    "distance_median_km": float(np.median(distances)) if len(distances) else np.nan,
                    **{f"band_{int(a)}_{'inf' if np.isinf(b) else int(b)}": int(((distances >= a) & (distances < b)).sum())
                       for a, b in zip(BANDS_KM[:-1], BANDS_KM[1:])},
                    "along_alone_km": track_errors(own, exclusive, x, y, first, last)[0],
                    "along_attached_km": track_errors(own, linked, x, y, first, last)[0],
                    "along_detached_km": track_errors(own, detached, x, y, first, last)[0]})
            out.append(result)
        if done % 500 == 0:
            release_pages(votes, *flags.values())
            print(f"[exclusive] {done} of {len(chosen)} clusters", flush=True)
    for array in flags.values():
        array._mmap.close()
    votes._mmap.close()
    results = pd.DataFrame(out)
    if args.csv is not None:
        results.to_csv(args.csv, index=False)
        print(f"[exclusive] written {args.csv}")

    bands = [c for c in results.columns if c.startswith("band_")]
    lines = []
    for name, g in results.groupby("detector", sort=False):
        total = g["exclusive"].sum()
        row = {"detector": name, "zone flags": int(g["zone_flags"].sum()), "exclusive": int(total),
               "exclusive % of zone flags": round(100 * total / max(g["zone_flags"].sum(), 1), 1),
               "attached %": round(100 * g["attached"].sum() / max(total, 1), 1),
               "ahead % of values": round(100 * g["ahead_values"].sum() / max(total, 1), 1)}
        row.update({b.replace("band_", "km ").replace("_", "-") + " %": round(100 * g[b].sum() / max(total, 1), 1)
                    for b in bands})
        for col, label in (("along_alone_km", "along all"), ("along_attached_km", "along attached"),
                           ("along_detached_km", "along detached")):
            s = g[col].dropna() if col in g else pd.Series(dtype=float)
            row[f"{label} km (median)"] = round(s.median(), 1) if len(s) else np.nan
            row[f"{label} ahead %"] = round(100 * (s > 0).mean(), 1) if len(s) else np.nan
        lines.append(row)
    table_out = pd.DataFrame(lines).set_index("detector")
    print(f"\n[exclusive] {len(chosen)} clusters ({args.kind}), zone {args.radius_km:g} km at the same timestamp. "
          "Exclusive = flagged by this detector alone. Attached = in a group of its flags that touches the cluster "
          "(inside the box). Distance = from the nearest cell of the cluster at the same timestamp; ahead % of values "
          "= share of the exclusive values ahead of the centroid of the cluster along its motion.")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(table_out.T.to_string())


if __name__ == "__main__":
    main()
