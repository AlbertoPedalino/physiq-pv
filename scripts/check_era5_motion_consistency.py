"""How coherent is the local motion of the concordant clusters? Mean translation (Davis et al. 2006b) or local bearing?

A check, not in the thesis. Reads the output of build_era5_concordant_cube.py (`--cube`). For every concordant cluster of
`--kind` (long by default) and every timestamp, the centroid of its cells (km, longitude times the cosine of the latitude,
as check_era5_concordant_lead.py) and its local bearing, from the centroid one timestamp before to one after
(check_era5_concordant_lead.local_unit, the motion of `--motion local`). No detector is involved. Per cluster:

- coherence R: the length of the mean of the local unit vectors (1: always the same direction; 0: no common direction);
- angle: the mean angle, in degrees, between the local bearings and the mean translation (first to last centroid);
- straightness: the distance between the first and the last centroid over the length of the path of the centroids;
- jumps: the share of the steps where the number of cells of the cluster changes by more than `--size-change` (parts
  joining or leaving: the centroid moves then without the system moving);
- step: the median distance between consecutive centroids, km.

With `--lead-mean` and `--lead-local` (the CSVs of check_era5_concordant_lead.py on the same clusters with the mean and
the local motion), the along-track error of `--first` and of every other detector by tercile of R, and cluster by cluster
in every tercile, `--first` against every other detector with a two-sided sign test.

    python scripts/check_era5_motion_consistency.py --cube outputs/concordate_5pc_regola_tesi \\
        --lead-mean outputs/concordate_5pc_regola_tesi/lead_r300_long.csv \\
        --lead-local outputs/concordate_5pc_regola_tesi/lead_local_r300_long.csv --csv motion_consistency_long.csv
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
from scipy.stats import binomtest, spearmanr

from physiq_pv.anomaly_detection.stgan.loading import release_pages
from scripts.check_era5_concordant_boxes import label_clusters
from scripts.check_era5_concordant_lead import CHUNK, EARTH_KM, local_unit

TERCILES = ["low R", "middle R", "high R"]


def consistency(own, x, y, size_change):
    """Coherence, angle to the mean translation (degrees), straightness, share of jumps and median step (km) of a cluster
    `own` [time, row, column] whose every timestamp has cells; NaN where undefined."""
    centre = lambda mask: np.array([x[mask].mean(), y[mask].mean()])
    times = len(own)
    centres = np.array([centre(own[t]) for t in range(times)])
    sizes = own.reshape(times, -1).sum(axis=1)
    steps = np.diff(centres, axis=0)
    lengths = np.hypot(steps[:, 0], steps[:, 1])
    net = centres[-1] - centres[0]
    units = [u for u in (local_unit(own, t, 0, times - 1, centre) for t in range(times)) if u is not None]
    out = {"timestamps": times, "values": int(sizes.sum()), "step_km": float(np.median(lengths)) if len(lengths) else np.nan,
           "straightness": float(np.hypot(*net) / lengths.sum()) if lengths.sum() > 0 else np.nan,
           "jumps": float((np.abs(np.diff(sizes)) / np.maximum(sizes[:-1], sizes[1:]) > size_change).mean())
           if times > 1 else np.nan}
    if units:
        units = np.array(units)
        out["coherence"] = float(np.hypot(*units.mean(axis=0)))
        if np.hypot(*net) > 0:
            mean_unit = net / np.hypot(*net)
            out["angle_deg"] = float(np.degrees(np.arccos(np.clip(units @ mean_unit, -1, 1))).mean())
    return out


def by_tercile(table, lead, first, label):
    """Median along-track error of every detector by tercile of coherence, and `first` against the others."""
    merged = lead.merge(table[["cluster", "tercile"]], on="cluster")
    print(f"\n[motion] {label}: median along-track error (km) by tercile of the coherence R")
    order = lambda frame: frame.reindex(columns=[c for c in TERCILES if c in frame.columns])
    print(order(merged.pivot_table(index="detector", columns="tercile", values="along_km",
                                   aggfunc="median")).round(1).to_string())
    wide = merged.pivot_table(index=["tercile", "cluster"], columns="detector", values="along_km")
    print(f"[motion] {label}: cluster by cluster, % of the clusters where {first} is further ahead (sign test p)")
    rows = {}
    for tercile, part in wide.groupby(level="tercile"):
        cells = {}
        for other in part.columns:
            if other == first or other.startswith(f"{first}_"):
                continue
            d = (part[first] - part[other]).dropna()
            k, n = int((d > 0).sum()), int((d != 0).sum())
            cells[other] = f"{100 * k / max(n, 1):.1f} ({binomtest(k, n).pvalue:.1g})" if n else ""
        rows[tercile] = cells
    print(order(pd.DataFrame(rows)).to_string())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cube", type=Path, required=True, help="Output directory of build_era5_concordant_cube.py")
    parser.add_argument("--kind", choices=("analysed", "long", "all"), default="long")
    parser.add_argument("--size-change", type=float, default=0.2)
    parser.add_argument("--first", default="Gn")
    parser.add_argument("--lead-mean", type=Path, help="CSV of check_era5_concordant_lead.py with the mean motion")
    parser.add_argument("--lead-local", type=Path, help="CSV of check_era5_concordant_lead.py with --motion local")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    summary = json.loads((args.cube / "summary.json").read_text(encoding="utf-8"))
    needed = summary["vote"]["votes_needed"]
    run = args.cube / "votes"
    grid = json.loads((run / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    latitudes = np.radians(np.asarray(grid["latitudes"], dtype=np.float64))
    longitudes = np.radians(np.asarray(grid["longitudes"], dtype=np.float64))
    times = len(np.load(run / "test_timestamps.npy"))
    table = pd.read_csv(args.cube / "clusters.csv")
    votes = np.load(run / "scores" / "anomaly_mean.npy", mmap_mode="r")
    concordant = np.zeros((times, len(rows)), dtype=bool)
    with np.errstate(invalid="ignore"):
        for start in range(0, times, CHUNK):
            concordant[start:start + CHUNK] = np.asarray(votes[start:start + CHUNK]) >= needed
    release_pages(votes)
    votes._mmap.close()
    labels, sizes = label_clusters(concordant, rows, cols, shape, summary["overlap"])
    del concordant
    if len(sizes) != len(table) or not np.array_equal(sizes, table["values"].to_numpy()):
        raise ValueError("The clusters labelled again differ from clusters.csv: not the same cube or rule")

    chosen = table if args.kind == "all" else table[table["kind"] == args.kind]
    out = []
    for row in chosen.itertuples(index=False):
        t0, t1, r0, r1, c0, c1 = int(row.t0), int(row.t1), int(row.row0), int(row.row1), int(row.col0), int(row.col1)
        own = labels[t0:t1 + 1, r0:r1 + 1, c0:c1 + 1] == row.cluster
        y = np.repeat(EARTH_KM * latitudes[r0:r1 + 1][:, None], c1 - c0 + 1, axis=1)
        x = EARTH_KM * longitudes[c0:c1 + 1][None, :] * np.cos(latitudes[r0:r1 + 1][:, None])
        out.append({"cluster": row.cluster, "hours": row.hours, **consistency(own, x, y, args.size_change)})
    result = pd.DataFrame(out)
    # Terciles on the rank, so that ties do not merge two of them (and any number of clusters works).
    names = np.array(TERCILES)
    rank = result["coherence"].rank(method="first")
    count = int(rank.notna().sum())
    result["tercile"] = pd.Series(pd.NA, index=result.index, dtype="object")
    if count:
        index = np.minimum((3 * (rank[rank.notna()] - 1) // count).astype(int), 2)
        result.loc[rank.notna(), "tercile"] = names[index.to_numpy()]
    if args.csv is not None:
        result.to_csv(args.csv, index=False)
        print(f"[motion] written {args.csv}")

    q = lambda s: f"{s.quantile(.25):.2f} / {s.median():.2f} / {s.quantile(.75):.2f}"
    print(f"\n[motion] {len(result)} clusters ({args.kind}); quartiles 25% / median / 75%")
    print(f"  coherence R            {q(result['coherence'].dropna())}")
    print(f"  angle to the mean (deg) {q(result['angle_deg'].dropna())}")
    print(f"  straightness           {q(result['straightness'].dropna())}")
    print(f"  share of jumps (> {args.size_change:g}) {q(result['jumps'].dropna())}")
    print(f"  step between centroids (km) {q(result['step_km'].dropna())}")
    both = result[["jumps", "coherence"]].dropna()
    rho = spearmanr(both["jumps"], both["coherence"])
    print(f"  Spearman(jumps, R) {rho.statistic:+.2f} (p {rho.pvalue:.1g}); Spearman(straightness, R) "
          f"{spearmanr(*result[['straightness', 'coherence']].dropna().T.values).statistic:+.2f}")
    edges = result.groupby("tercile", observed=True)["coherence"].agg(["min", "max", "count"]).reindex(TERCILES).dropna()
    print("  terciles of R: " + "; ".join(f"{t} {r['min']:.2f}-{r['max']:.2f} ({int(r['count'])})" for t, r in edges.iterrows()))
    for path, label in ((args.lead_mean, "mean motion"), (args.lead_local, "local motion")):
        if path is not None:
            by_tercile(result, pd.read_csv(path), args.first, label)


if __name__ == "__main__":
    main()
