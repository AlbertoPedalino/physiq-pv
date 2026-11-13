"""Which concordant clusters each detector catches, which of its groups have no counterpart, and who starts first.

Reads the output of build_era5_concordant_cube.py (`--cube`). The flags of every detector (after its closing) are joined
into groups with the rule of the clusters of the thesis (touching cells at a timestamp, diagonals included; the same cell
at consecutive timestamps) over the whole domain and test period; groups smaller than the minimum size of the analysed
clusters are left out, as the clusters are. The targets are the analysed and the long concordant clusters.

Matching, as Bourdin et al. (2022, GMD, Sect. 2.4) match detected and observed tracks:
- a group is near a cluster when at some timestamp a value of the group lies less than `--radius-km` (300) from a value of
  the cluster (great-circle distance between cell centres; scripts/era5_km_zone.py);
- a group near several targets goes to the one with the most associated values (the values of the cluster with a value
  of the group within the radius at the same timestamp; Bourdin's case 3), ties to the lowest cluster number;
- the groups that go to the same cluster are merged (Bourdin, p. 6764): the match starts at their first timestamp and
  ends at their last;
- a group near no target but near a small cluster is "near small clusters only"; near none, "without counterpart";
- a target with a group is caught (hit), otherwise missed; POD = H / (H + M) and FAR = FA / (H + FA) (Bourdin eqs. 3-4),
  FA counted as the groups without counterpart and, apart, also with those near small clusters only. A group without
  counterpart in the cube is not a false alarm: the cube is no truth.
Also given: how many targets are near a group that went to another cluster, and the share of the values of the domain
less than the radius from the cube (without it the POD cannot be read: a group placed at random would often be near).

Leads at the start and at the end (as Bourdin, Sect. 4.3, compares the first and final dates of matched tracks): on the
matches that also pass the criterion in time of Crawford et al. (2021, MWR, Sect. 2.4), 2 n_m / (n_1 + n_2) >= 0.6, with
n_1 the distinct timestamps of the merged groups, n_2 those of the cluster, n_m those where both exist. Start lead = first
timestamp of the match minus that of the cluster, in hours (negative: before); end lead = last minus last (positive:
after). Clusters or matches touching the first or the last timestamp of the period are left out of the leads
(counted). Why a match fails the criterion: the merged groups last longer than the cluster (n_1 > n_2), shorter
(n_1 < n_2), or as long but shifted in time. As a sensitivity: the leads without criterion, with the ratio of durations
of Davis et al. (2006b, MWR, p. 1789; 0.5 <= n_2 / n_1 <= 2, on distinct timestamps as Crawford), and cut at
`--cut-hours` (24) around the cluster, as the old zone did.

The groups without counterpart: the time from the nearest target in time whose first (or last) timestamp has a value less
than the radius from a value of the last (or first) timestamp of the group: negative when the group ends before the
cluster starts, positive when it starts after the cluster ends; none if no target is so near; and whether it coexists in
time with some target anywhere in the domain. One category each, in this order: "near a target in time" (a time is
found), "coexists, far" (no time, but some target exists at the same time, always beyond the radius), "none near".

`--save-lonely` writes, for every detector, the values of its groups without counterpart (against the cube) to
CUBE/lonely/NAME.npy, packed as the flags of the cube, with NAME.json (radius, groups, values), for
check_era5_concordant_excess.py --lonely-dir: what those values hold (standardised inputs, course, judges), as for the
excess around the clusters. Only against the cube, not against the cube without the detector's vote.

`--without-own` repeats everything against the cube without the vote of the detector (the values that at least the votes
needed of the others flag), with its clusters labelled and selected as those of the cube.

    python scripts/check_era5_concordant_match.py --cube outputs/concordate_5pc_regola_tesi --without-own \\
        --csv outputs/concordate_5pc_regola_tesi/match.csv
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
from scripts.check_era5_concordant_boxes import label_clusters
from scripts.era5_km_zone import nearest_within, row_offsets

CHUNK = 256  # timestamps read at a time
KEY = np.int64(1) << 32  # packs (cell of a block, label) into one integer


def unpack(packed, start, stop, cells):
    return np.unpackbits(np.asarray(packed[start:stop]), axis=1, count=cells).astype(bool)


def spans(lattice, count):
    """First and last timestamp of every label 1..count of a lattice [time, row, column]."""
    first = np.full(count + 1, np.iinfo(np.int64).max)
    last = np.full(count + 1, -1)
    for t in range(len(lattice)):
        present = np.unique(lattice[t])
        present = present[present > 0]
        np.minimum.at(first, present, t)
        np.maximum.at(last, present, t)
    return first, last


def keep_large(lattice, sizes, min_values):
    """The lattice with the labels of fewer than `min_values` values set to 0 and the others renumbered from 1."""
    keep = np.asarray(sizes) >= min_values
    remap = np.zeros(len(sizes) + 1, dtype=lattice.dtype)
    remap[1:][keep] = np.arange(1, int(keep.sum()) + 1, dtype=lattice.dtype)
    for start in range(0, len(lattice), CHUNK):
        lattice[start:start + CHUNK] = remap[lattice[start:start + CHUNK]]
    return lattice, int(keep.sum())


def near_counts(targets, groups, offsets):
    """For lattices of cluster and group labels [time, row, column], the pairs (cluster, group, n): n values of the
    cluster have a value of the group less than the radius of `offsets` away at the same timestamp."""
    times, height, width = targets.shape
    found = []
    for start in range(0, times, CHUNK):
        a = targets[start:start + CHUNK]
        b = groups[start:start + CHUNK]
        steps = len(a)
        has_group = (b > 0).any(axis=(0, 2))
        for r in range(height):
            tgt = a[:, r, :]
            if not (tgt > 0).any():
                continue
            keys = []
            for dr, dc, _ in offsets[r]:
                dr, dc = int(dr), int(dc)
                if not has_group[r + dr]:
                    continue
                source = b[:, r + dr, :]
                cand = np.zeros((steps, width), dtype=b.dtype)
                if dc >= 0:
                    cand[:, :width - dc] = source[:, dc:]
                else:
                    cand[:, -dc:] = source[:, :width + dc]
                m = (tgt > 0) & (cand > 0)
                if m.any():
                    t, c = np.nonzero(m)
                    keys.append((t.astype(np.int64) * width + c) * KEY + cand[m].astype(np.int64))
            if not keys:
                continue
            unique = np.unique(np.concatenate(keys))  # one per (value of the cluster, group)
            cell, group = unique // KEY, unique % KEY
            cluster = tgt.reshape(-1)[cell].astype(np.int64)
            pairs, n = np.unique(cluster * KEY + group, return_counts=True)
            found.append(np.stack((pairs // KEY, pairs % KEY, n), axis=1))
    if not found:
        return np.zeros((0, 3), dtype=np.int64)
    table = pd.DataFrame(np.concatenate(found), columns=["cluster", "group", "n"]).groupby(["cluster", "group"]).sum()
    return table.reset_index().to_numpy(dtype=np.int64)


def union_length(intervals):
    """Number of distinct timestamps covered by inclusive intervals [(a, b), ...]."""
    total, end = 0, -1
    for a, b in sorted(intervals):
        if b <= end:
            continue
        total += b - max(a, end + 1) + 1
        end = b
    return total


def boundary_cells(lattice, labels, at, width):
    """For every label of `labels`, the flat cells (row * width + column) it holds at its timestamp `at[label]`."""
    out = {}
    order = np.argsort(at[labels], kind="stable")
    labels = labels[order]
    times = at[labels]
    for t in np.unique(times):
        these = labels[times == t]
        grid = lattice[t]
        hit = np.isin(grid, these)
        r, c = np.nonzero(hit)
        owner = grid[r, c]
        flat = r * width + c
        for label in these:
            out[int(label)] = flat[owner == label]
    return out


def disc(cells, offsets, height, width):
    """The flat cells less than the radius of `offsets` from any of `cells` (flat indices)."""
    rows, cols = np.divmod(cells, width)
    out = []
    for r in np.unique(rows):
        here = cols[rows == r]
        offs = offsets[r]
        nr = r + offs[:, 0].astype(np.int64)
        nc = here[:, None] + offs[:, 1].astype(np.int64)[None, :]
        ok = (nc >= 0) & (nc < width)
        out.append((np.broadcast_to(nr, nc.shape)[ok]) * width + nc[ok])
    return np.unique(np.concatenate(out)) if out else np.zeros(0, dtype=np.int64)


def time_gaps(groups_lattice, group_first, group_last, unmatched, targets_lattice, target_ids, target_first, target_last,
              offsets, step_hours):
    """For every group without counterpart: hours from the nearest target in time with a value of its first (last)
    timestamp less than the radius from a value of the last (first) timestamp of the group; NaN when none."""
    height, width = groups_lattice.shape[1:]
    def by_cell(boundary, at):
        """flat cell -> (timestamps, clusters) sorted by timestamp, then cluster."""
        cells, times, ids = [], [], []
        for k, these in boundary.items():
            cells.append(these)
            times.append(np.full(len(these), int(at[k]), dtype=np.int64))
            ids.append(np.full(len(these), int(k), dtype=np.int64))
        if not cells:
            return {}
        cells, times, ids = np.concatenate(cells), np.concatenate(times), np.concatenate(ids)
        order = np.lexsort((ids, times, cells))
        cells, times, ids = cells[order], times[order], ids[order]
        bounds = np.flatnonzero(np.diff(cells)) + 1
        return {int(c[0]): (t, i) for c, t, i in zip(np.split(cells, bounds), np.split(times, bounds),
                                                       np.split(ids, bounds))}

    starts = by_cell(boundary_cells(targets_lattice, target_ids, target_first, width), target_first)
    ends = by_cell(boundary_cells(targets_lattice, target_ids, target_last, width), target_last)
    g_last_cells = boundary_cells(groups_lattice, unmatched, group_last, width)
    g_first_cells = boundary_cells(groups_lattice, unmatched, group_first, width)
    gaps = np.full(len(unmatched), np.nan)
    nearest = np.zeros(len(unmatched), dtype=np.int64)  # the cluster of the gap (0: none)
    for i, g in enumerate(unmatched):
        best = None  # (|gap| in timestamps, cluster, signed gap)
        te = int(group_last[g])
        for cell in disc(g_last_cells[int(g)], offsets, height, width):
            entry = starts.get(int(cell))
            if entry is None:
                continue
            t, k = entry
            j = np.searchsorted(t, te, side="right")  # the first start after the end of the group (lowest cluster)
            if j < len(t):
                cand = (int(t[j]) - te, int(k[j]), te - int(t[j]))
                best = cand if best is None or cand[:2] < best[:2] else best
        ts = int(group_first[g])
        for cell in disc(g_first_cells[int(g)], offsets, height, width):
            entry = ends.get(int(cell))
            if entry is None:
                continue
            t, k = entry
            j = np.searchsorted(t, ts, side="left") - 1  # the last end before the start of the group
            if j >= 0:
                j = np.searchsorted(t, t[j], side="left")  # the lowest cluster with that end
                cand = (ts - int(t[j]), int(k[j]), ts - int(t[j]))
                best = cand if best is None or cand[:2] < best[:2] else best
        if best is not None:
            gaps[i] = best[2] * step_hours
            nearest[i] = best[1]
    return gaps, nearest


def evaluate(name, targets_lattice, table, groups_lattice, n_groups, offsets, step_hours, cut_steps, crawford, davis,
             times):
    """Matching of the groups of one detector against one cube; returns (per cluster, per unmatched group, summary)."""
    target_ids = table.loc[table["kind"].isin(["analysed", "long"]), "cluster"].to_numpy()
    small_ids = set(table.loc[table["kind"] == "small", "cluster"].tolist())
    is_target = np.zeros(len(table) + 1, dtype=bool)
    is_target[target_ids] = True
    g_first, g_last = spans(groups_lattice, n_groups)
    k_first = np.zeros(len(table) + 1, dtype=np.int64)
    k_last = np.zeros(len(table) + 1, dtype=np.int64)
    k_first[table["cluster"].to_numpy()] = table["t0"].to_numpy()
    k_last[table["cluster"].to_numpy()] = table["t1"].to_numpy()

    pairs = near_counts(targets_lattice, groups_lattice, offsets)
    category = np.full(n_groups + 1, "without counterpart", dtype=object)
    assigned = {}
    touched = set()
    if len(pairs):
        frame = pd.DataFrame(pairs, columns=["cluster", "group", "n"])
        frame["target"] = is_target[frame["cluster"].to_numpy()]
        near_small = frame.loc[~frame["target"], "group"].unique()
        category[near_small] = "near small clusters only"
        tgt = frame[frame["target"]].sort_values(["group", "n", "cluster"], ascending=[True, False, True])
        touched = set(tgt["cluster"].tolist())
        best = tgt.drop_duplicates("group")  # the most associated values, ties to the lowest cluster
        category[best["group"].to_numpy()] = "matched"
        for k, g in zip(best["cluster"], best["group"]):
            assigned.setdefault(int(k), []).append(int(g))
    category = category[1:]

    kind_of = dict(zip(table["cluster"].tolist(), table["kind"].tolist()))
    rows = []
    for k in target_ids:
        k = int(k)
        groups = assigned.get(k, [])
        result = {"detector": name, "cluster": k, "kind": kind_of[k],
                  "caught": bool(groups), "groups": len(groups), "touched": k in touched,
                  "border": bool(k_first[k] == 0 or k_last[k] == times - 1)}  # the match may widen it, below
        if groups:
            intervals = [(int(g_first[g]), int(g_last[g])) for g in groups]
            start, end = min(a for a, _ in intervals), max(b for _, b in intervals)
            n1 = union_length(intervals)
            n2 = int(k_last[k] - k_first[k] + 1)
            nm = union_length([(max(a, int(k_first[k])), min(b, int(k_last[k]))) for a, b in intervals
                               if min(b, int(k_last[k])) >= max(a, int(k_first[k]))])
            ratio = 2 * nm / (n1 + n2)
            durations = n2 / n1  # Davis' ratio on distinct timestamps, as Crawford counts them
            result.update({"start": start, "end": end, "n1": n1, "n2": n2, "nm": nm, "crawford": ratio,
                           "passes_crawford": ratio >= crawford,
                           "reason": "longer" if n1 > n2 else "shorter" if n1 < n2 else "shifted",
                           "border": result["border"] or start == 0 or end == times - 1,
                           "passes_davis": davis[0] <= durations <= davis[1],
                           "start_lead_h": (start - k_first[k]) * step_hours, "end_lead_h": (end - k_last[k]) * step_hours,
                           "start_cut_h": (max(start, k_first[k] - cut_steps) - k_first[k]) * step_hours,
                           "end_cut_h": (min(end, k_last[k] + cut_steps) - k_last[k]) * step_hours})
        rows.append(result)
    clusters = pd.DataFrame(rows, columns=None if rows else ["detector", "cluster", "kind", "caught", "groups",
                                                             "touched", "border"])  # no target: an empty table
    for column in ("start", "end", "n1", "n2", "nm", "crawford", "passes_crawford", "reason", "passes_davis",
                   "start_lead_h", "end_lead_h", "start_cut_h", "end_cut_h"):
        if column not in clusters:
            clusters[column] = np.nan

    unmatched = np.flatnonzero(category == "without counterpart") + 1
    target_first = k_first
    target_last = k_last
    gaps, gap_cluster = time_gaps(groups_lattice, g_first, g_last, unmatched, targets_lattice, target_ids, target_first,
                                  target_last, offsets, step_hours) if len(unmatched) else (np.zeros(0), np.zeros(0, int))
    starts_sorted = np.sort(k_first[target_ids])
    ends_sorted = np.sort(k_last[target_ids])
    coexist = (np.searchsorted(starts_sorted, g_last[unmatched], side="right")
               - np.searchsorted(ends_sorted, g_first[unmatched], side="left")) > 0
    lonely = pd.DataFrame({"detector": name, "group": unmatched, "first": g_first[unmatched], "last": g_last[unmatched],
                           "gap_h": gaps, "gap_cluster": gap_cluster, "coexists": coexist})
    lonely["category"] = np.where(lonely["gap_h"].notna(), "near a target in time",
                                  np.where(lonely["coexists"], "coexists, far", "none near"))
    counts = pd.Series(category).value_counts()
    summary = {"detector": name, "groups": n_groups,
               "matched": int(counts.get("matched", 0)),
               "near_small_only": int(counts.get("near small clusters only", 0)),
               "without_counterpart": int(counts.get("without counterpart", 0))}
    return clusters, lonely, summary


def scores(clusters, summaries):
    """One row per detector and kind of target: POD, FAR, the criterion of Crawford and the leads."""
    out = []
    for s in summaries:
        name = s["detector"]
        c = clusters[clusters["detector"] == name]
        is_caught = c["caught"].astype(bool)  # an empty table concatenated makes it an object column
        hits_all = int(is_caught.sum())  # the false alarms are not split by kind: the rate is over every target
        for kind in ("analysed", "long"):
            part = c[c["kind"] == kind]
            got = part["caught"].astype(bool)
            hits, misses = int(got.sum()), int((~got).sum())
            fa, fa_all = s["without_counterpart"], s["without_counterpart"] + s["near_small_only"]
            caught = part[got]
            lead = caught[~caught["border"].astype(bool)]
            craw = lead[lead["passes_crawford"].astype(bool)]
            row = {"detector": name, "kind": kind, "targets": len(part), "POD": hits / max(hits + misses, 1),
                   "touched, went elsewhere": int((part["touched"].astype(bool) & ~got).sum()),
                   "pass Crawford %": 100 * caught["passes_crawford"].astype(bool).mean() if len(caught) else np.nan,
                   "failing: longer / shorter / shifted": "/".join(
                       str(int((caught.loc[~caught["passes_crawford"].astype(bool), "reason"] == why).sum()))
                       for why in ("longer", "shorter", "shifted")),
                   "border": int(caught["border"].astype(bool).sum()),
                   "start lead h (Crawford)": craw["start_lead_h"].median(),
                   "starts before %": 100 * (craw["start_lead_h"] < 0).mean() if len(craw) else np.nan,
                   "end lead h (Crawford)": craw["end_lead_h"].median(),
                   "ends after %": 100 * (craw["end_lead_h"] > 0).mean() if len(craw) else np.nan,
                   "start / end h, no criterion": f"{lead['start_lead_h'].median():.3g} / "
                                                    f"{lead['end_lead_h'].median():.3g}",
                   "start / end h, Davis": f"{lead.loc[lead['passes_davis'].astype(bool), 'start_lead_h'].median():.3g} / "
                                           f"{lead.loc[lead['passes_davis'].astype(bool), 'end_lead_h'].median():.3g}",
                   "start / end h, cut": f"{lead['start_cut_h'].median():.3g} / {lead['end_cut_h'].median():.3g}"}
            row["FAR (without counterpart)"] = fa / max(hits_all + fa, 1)
            row["FAR (+ near small only)"] = fa_all / max(hits_all + fa_all, 1)
            out.append(row)
    return pd.DataFrame(out)


def report(title, clusters, lonely, summaries, base_percent):
    print(f"\n[match] {title}; values of the domain less than the radius from the cube: {base_percent:.1f}%")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3g}".format):
        print(scores(clusters, summaries).to_string(index=False))
        print("\n[match] groups: matched / near small clusters only / without counterpart")
        print(pd.DataFrame(summaries).to_string(index=False))
        if len(lonely):
            g = lonely.groupby("detector")
            table = pd.DataFrame({
                "without counterpart": g.size(),
                "near a target in time %": g["category"].apply(lambda s: 100 * (s == "near a target in time").mean()),
                "coexists, far %": g["category"].apply(lambda s: 100 * (s == "coexists, far").mean()),
                "none near %": g["category"].apply(lambda s: 100 * (s == "none near").mean()),
                "ends before it %": g["gap_h"].apply(lambda s: 100 * (s < 0).mean()),
                "median gap h": g["gap_h"].median(),
                "quartiles gap h": g["gap_h"].apply(lambda s: f"{s.quantile(0.25):.3g} to {s.quantile(0.75):.3g}"),
            })
            print("\n[match] groups without counterpart: time from the nearest target within the radius (negative = "
                  "the group ends before it starts)")
            print(table.to_string())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cube", type=Path, required=True, help="Output directory of build_era5_concordant_cube.py")
    parser.add_argument("--detectors", nargs="+", help="Detectors of the cube (default: all)")
    parser.add_argument("--radius-km", type=float, default=300.0)
    parser.add_argument("--crawford", type=float, default=0.6)
    parser.add_argument("--davis", type=float, nargs=2, default=[0.5, 2.0])
    parser.add_argument("--cut-hours", type=float, default=24.0)
    parser.add_argument("--without-own", action="store_true")
    parser.add_argument("--save-lonely", action="store_true",
                        help="Write the values of the groups without counterpart to CUBE/lonely/NAME.npy")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    summary = json.loads((args.cube / "summary.json").read_text(encoding="utf-8"))
    names = args.detectors or list(summary["vote"]["detectors"])
    needed = summary["vote"]["votes_needed"]
    min_values = summary["selection"]["min_values"]
    max_hours = summary["selection"]["max_hours"]
    run = args.cube / "votes"
    grid = json.loads((run / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    timestamps = pd.DatetimeIndex(np.load(run / "test_timestamps.npy"))
    steps = np.diff(timestamps.asi8)
    if len(steps) and not (steps == steps[0]).all():
        raise ValueError("The timestamps are not evenly spaced: the leads in hours would be wrong")
    step_hours = float(steps[0]) / 3.6e12 if len(steps) else 3.0
    times = len(timestamps)
    table = pd.read_csv(args.cube / "clusters.csv")
    votes = np.load(run / "scores" / "anomaly_mean.npy", mmap_mode="r")
    offsets = row_offsets(np.asarray(grid["latitudes"], dtype=np.float64),
                          float(abs(grid["longitudes"][1] - grid["longitudes"][0])), args.radius_km)
    cut_steps = int(round(args.cut_hours / step_hours))

    concordant = np.zeros((times, len(rows)), dtype=bool)
    valid = np.zeros((times, len(rows)), dtype=bool)
    with np.errstate(invalid="ignore"):
        for start in range(0, times, CHUNK):
            block = np.asarray(votes[start:start + CHUNK])
            valid[start:start + CHUNK] = np.isfinite(block)
            concordant[start:start + CHUNK] = block >= needed
    release_pages(votes)
    labels, sizes = label_clusters(concordant, rows, cols, shape, summary["overlap"])
    if len(sizes) != len(table) or not np.array_equal(sizes, table["values"].to_numpy()):
        raise ValueError("The clusters labelled again differ from clusters.csv: not the same cube or rule")
    near_cube = 0
    for start in range(0, times, CHUNK):
        owner = nearest_within(labels[start:start + CHUNK], offsets)
        near_cube += int((owner[:, rows, cols] > 0)[valid[start:start + CHUNK]].sum())
    base_percent = 100 * near_cube / max(int(valid.sum()), 1)
    print(f"[match] {len(table)} clusters; targets: {int(table['kind'].isin(['analysed', 'long']).sum())} "
          f"(analysed and long); radius {args.radius_km:g} km at the same timestamp", flush=True)

    flags = {name: np.load(args.cube / "flags" / f"{name}.npy", mmap_mode="r") for name in names}
    results = {"cube": ([], [], []), "without own": ([], [], [])}
    for name in names:
        own = np.zeros((times, len(rows)), dtype=bool)
        for start in range(0, times, CHUNK):
            own[start:start + CHUNK] = unpack(flags[name], start, start + CHUNK, len(rows))
        groups, group_sizes = label_clusters(own, rows, cols, shape, summary["overlap"])
        groups, n_groups = keep_large(groups, group_sizes, min_values)
        print(f"[match] {name}: {n_groups} groups of at least {min_values} values", flush=True)
        out = evaluate(name, labels, table, groups, n_groups, offsets, step_hours, cut_steps, args.crawford, args.davis,
                       times)
        for store, part in zip(results["cube"], out):
            store.append(part)
        if args.save_lonely:  # the values of the groups without counterpart, packed as the flags of the cube
            ids = out[1]["group"].to_numpy()
            folder = args.cube / "lonely"
            folder.mkdir(exist_ok=True)
            packed = np.zeros((times, (len(rows) + 7) // 8), dtype=np.uint8)
            for start in range(0, times, CHUNK):
                lonely = np.isin(groups[start:start + CHUNK], ids)[:, rows, cols]
                packed[start:start + CHUNK] = np.packbits(lonely, axis=1)
            np.save(folder / f"{name}.npy", packed)
            (folder / f"{name}.json").write_text(json.dumps(
                {"detector": name, "radius_km": args.radius_km, "groups": int(len(ids)),
                 "values": int(np.unpackbits(packed, axis=1, count=len(rows)).sum())}), encoding="utf-8")
        if args.without_own:  # the cube without the vote of this detector, its clusters selected as those of the cube
            others = np.zeros((times, len(rows)), dtype=bool)
            for start in range(0, times, CHUNK):
                block = np.nan_to_num(np.asarray(votes[start:start + CHUNK]))
                others[start:start + CHUNK] = (block - own[start:start + CHUNK]) >= needed
            other_labels, other_sizes = label_clusters(others, rows, cols, shape, summary["overlap"])
            del others
            o_first, o_last = spans(other_labels, len(other_sizes))
            hours = (o_last[1:] - o_first[1:]) * step_hours
            other_table = pd.DataFrame({"cluster": np.arange(1, len(other_sizes) + 1), "values": other_sizes,
                                        "t0": o_first[1:], "t1": o_last[1:], "hours": hours})
            other_table["kind"] = np.where(other_table["values"] < min_values, "small",
                                           np.where(other_table["hours"] <= max_hours, "analysed", "long"))
            out = evaluate(name, other_labels, other_table, groups, n_groups, offsets, step_hours, cut_steps,
                           args.crawford, args.davis, times)
            for store, part in zip(results["without own"], out):
                store.append(part)
            del other_labels
        del groups, own
        release_pages(votes, flags[name])
    for array in (votes, *flags.values()):
        array._mmap.close()

    for label, (clusters, lonely, summaries) in results.items():
        if not clusters:
            continue
        clusters, lonely = pd.concat(clusters, ignore_index=True), pd.concat(lonely, ignore_index=True)
        report(f"against the {'cube' if label == 'cube' else 'cube without the vote of the detector'}", clusters, lonely,
               summaries, base_percent)
        if args.csv:
            suffix = "" if label == "cube" else "_without_own"
            path = args.csv.with_name(args.csv.stem + suffix + args.csv.suffix)
            clusters.to_csv(path, index=False)
            lonely.to_csv(path.with_name(path.stem + "_lonely" + path.suffix), index=False)
            print(f"\nwritten {path}")


if __name__ == "__main__":
    main()
