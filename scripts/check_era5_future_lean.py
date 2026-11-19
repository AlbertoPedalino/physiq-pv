"""Do the flags of a detector lean towards where a concordant cluster will be later?

A check, not in the thesis. Reads the output of build_era5_concordant_cube.py (`--cube`); for every concordant cluster of
`--kind` (long by default) the zone is the one of the thesis (scripts/era5_km_zone.py, `--radius-km`) and the flags of a
detector are all its flags in the zone at every timestamp, as in check_era5_concordant_lead.py --radius-km. Positions are
in km, longitude times the cosine of the latitude, as there. With x_s(t) the centroid of the flags of the detector and
x_c(t) that of the cluster:

- toward the future: the offset x_s(t) - x_c(t) projected on the direction from x_c(t) to x_c(t + h), for h of `--hours`,
  the mean over the timestamps with t + h still in the cluster;
- local: the same offset projected on the local bearing of the cluster (check_era5_concordant_lead.local_unit);
- best lag: the lag k (in steps of the data, from -`--max-lag` to +`--max-lag` hours) minimising the mean distance between
  x_s(t) and x_c(t + k), over the same timestamps for every k (those with the whole range of lags inside the cluster),
  in hours. Positive: the flags at t sit where the cluster will be k hours later; zero: where it is; negative: where it
  was.

`--first` (G_n) is compared cluster by cluster with every other detector on the best lag, and with itself on the
projection toward the longest horizon against the local one, with two-sided sign tests (ties left out). With
`--consistency` (the CSV of check_era5_motion_consistency.py) the same by tercile of the coherence of the motion.

    python scripts/check_era5_future_lean.py --cube outputs/concordate_5pc_regola_tesi \\
        --consistency outputs/concordate_5pc_regola_tesi/motion_consistency_long.csv \\
        --csv outputs/concordate_5pc_regola_tesi/future_lean_long.csv
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
from scipy.stats import binomtest

from physiq_pv.anomaly_detection.stgan.loading import release_pages
from scripts.check_era5_concordant_boxes import label_clusters
from scripts.check_era5_concordant_lead import CHUNK, EARTH_KM, local_unit
from scripts.era5_km_zone import owner_lattice, radius_box

TERCILES = ["low R", "middle R", "high R"]


def lean(own, flags, x, y, horizons, max_lag):
    """Projections toward the future (one per horizon, in steps), on the local bearing, and the best lag (steps) of the
    flags `flags` [time, row, column] of a detector against the cluster `own`; NaN where undefined."""
    times = len(own)
    centre = lambda mask: np.array([x[mask].mean(), y[mask].mean()])
    xc = np.array([centre(own[t]) if own[t].any() else [np.nan, np.nan] for t in range(times)])
    xs = np.array([centre(flags[t]) if flags[t].any() else [np.nan, np.nan] for t in range(times)])
    has = ~np.isnan(xs[:, 0]) & ~np.isnan(xc[:, 0])
    out = {}
    for h in horizons:
        values = []
        for t in np.flatnonzero(has):
            if t + h >= times or np.isnan(xc[t + h, 0]):
                continue
            d = xc[t + h] - xc[t]
            if np.hypot(*d) > 0:
                values.append((xs[t] - xc[t]) @ (d / np.hypot(*d)))
        out[h] = float(np.mean(values)) if values else np.nan
    local = []
    for t in np.flatnonzero(has):
        u = local_unit(own, t, 0, times - 1, centre)
        if u is not None:
            local.append((xs[t] - xc[t]) @ u)
    out["local"] = float(np.mean(local)) if local else np.nan
    inner = [t for t in np.flatnonzero(has) if t - max_lag >= 0 and t + max_lag < times
             and not np.isnan(xc[t - max_lag:t + max_lag + 1, 0]).any()]
    if inner:
        inner = np.array(inner)
        distances = [np.hypot(*(xs[inner] - xc[inner + k]).T).mean() for k in range(-max_lag, max_lag + 1)]
        out["lag"] = int(np.argmin(distances)) - max_lag
    else:
        out["lag"] = np.nan
    return out


def sign(a, b):
    d = (a - b).dropna()
    k, n = int((d > 0).sum()), int((d != 0).sum())
    return (100 * k / n, n, binomtest(k, n).pvalue) if n else (np.nan, 0, np.nan)


def report(results, first, horizons_h, label):
    """Medians per detector and the paired comparisons of `first`."""
    longest = f"toward_{horizons_h[-1]:g}h_km"
    print(f"\n[future] {label}: medians over the clusters (best lag in hours; projections in km, positive = toward)")
    columns = ["best_lag_h"] + [f"toward_{h:g}h_km" for h in horizons_h] + ["local_km"]
    table = results.groupby("detector", sort=False)[columns].median().round(1)
    table["lag > 0 %"] = results.groupby("detector", sort=False)["best_lag_h"].apply(lambda s: round(100 * (s.dropna() > 0).mean(), 1))
    table["lag < 0 %"] = results.groupby("detector", sort=False)["best_lag_h"].apply(lambda s: round(100 * (s.dropna() < 0).mean(), 1))
    print(table.to_string())
    wide = results.pivot_table(index="cluster", columns="detector", values="best_lag_h")
    print(f"[future] {label}: cluster by cluster, % of the clusters where the best lag of {first} is larger (ties out)")
    for other in wide.columns:
        if other == first:
            continue
        pct, n, p = sign(wide[first], wide[other])
        print(f"  {other:10s} {pct:5.1f}% of {n}, sign test p {p:.2g}")
    mine = results[results["detector"] == first].set_index("cluster")
    pct, n, p = sign(mine[longest], mine["local_km"])
    print(f"[future] {label}: {first}, projection toward {horizons_h[-1]:g} h larger than on the local bearing in "
          f"{pct:.1f}% of {n} clusters, sign test p {p:.2g}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cube", type=Path, required=True, help="Output directory of build_era5_concordant_cube.py")
    parser.add_argument("--kind", choices=("analysed", "long", "all"), default="long")
    parser.add_argument("--radius-km", type=float, default=300.0)
    parser.add_argument("--hours", nargs="+", type=float, default=[12, 24])
    parser.add_argument("--max-lag", type=float, default=24.0, help="Largest lag searched, hours")
    parser.add_argument("--first", default="Gn")
    parser.add_argument("--consistency", type=Path, help="CSV of check_era5_motion_consistency.py (coherence R)")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    summary = json.loads((args.cube / "summary.json").read_text(encoding="utf-8"))
    names = list(summary["vote"]["detectors"])
    needed = summary["vote"]["votes_needed"]
    if args.first not in names:
        raise ValueError(f"--first {args.first} is not a detector of the cube")
    run = args.cube / "votes"
    grid = json.loads((run / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    latitudes = np.radians(np.asarray(grid["latitudes"], dtype=np.float64))
    longitudes = np.radians(np.asarray(grid["longitudes"], dtype=np.float64))
    timestamps = pd.DatetimeIndex(np.load(run / "test_timestamps.npy"))
    step_hours = float(np.median(np.diff(timestamps.asi8))) / 3.6e12
    horizons = [max(int(round(h / step_hours)), 1) for h in args.hours]
    max_lag = max(int(round(args.max_lag / step_hours)), 1)
    table = pd.read_csv(args.cube / "clusters.csv")
    votes = np.load(run / "scores" / "anomaly_mean.npy", mmap_mode="r")
    times = len(timestamps)
    flags = {name: np.load(args.cube / "flags" / f"{name}.npy", mmap_mode="r") for name in names}
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
    owner, reach_rows, reach_cols = owner_lattice(labels, grid, args.radius_km, CHUNK)

    chosen = table if args.kind == "all" else table[table["kind"] == args.kind]
    out = []
    for done, row in enumerate(chosen.itertuples(index=False), 1):
        t0, t1, r0, r1, c0, c1 = radius_box(row, reach_rows, reach_cols, shape)
        near = owner[t0:t1 + 1, r0:r1 + 1, c0:c1 + 1] == row.cluster
        own = labels[t0:t1 + 1, r0:r1 + 1, c0:c1 + 1] == row.cluster
        y = np.repeat(EARTH_KM * latitudes[r0:r1 + 1][:, None], c1 - c0 + 1, axis=1)
        x = EARTH_KM * longitudes[c0:c1 + 1][None, :] * np.cos(latitudes[r0:r1 + 1][:, None])
        for name in names:
            block = np.zeros((t1 - t0 + 1, *shape), dtype=bool)
            block[:, rows, cols] = np.unpackbits(np.asarray(flags[name][t0:t1 + 1]), axis=1, count=len(rows)).astype(bool)
            got = lean(own, block[:, r0:r1 + 1, c0:c1 + 1] & near, x, y, horizons, max_lag)
            out.append({"cluster": row.cluster, "detector": name, "hours": row.hours,
                        "best_lag_h": got["lag"] * step_hours if not np.isnan(got["lag"]) else np.nan,
                        **{f"toward_{h * step_hours:g}h_km": got[h] for h in horizons}, "local_km": got["local"]})
        if done % 500 == 0:
            release_pages(*flags.values())
            print(f"[future] {done} of {len(chosen)} clusters", flush=True)
    for array in flags.values():
        array._mmap.close()
    results = pd.DataFrame(out)
    if args.csv is not None:
        results.to_csv(args.csv, index=False)
        print(f"[future] written {args.csv}")
    horizons_h = [h * step_hours for h in horizons]
    report(results, args.first, horizons_h, f"{len(chosen)} clusters ({args.kind}), zone {args.radius_km:g} km")
    if args.consistency is not None:
        terc = pd.read_csv(args.consistency)[["cluster", "tercile"]]
        merged = results.merge(terc, on="cluster")
        for tercile in TERCILES:
            part = merged[merged["tercile"] == tercile]
            if len(part):
                report(part, args.first, horizons_h, f"tercile {tercile} ({part['cluster'].nunique()} clusters)")


if __name__ == "__main__":
    main()
