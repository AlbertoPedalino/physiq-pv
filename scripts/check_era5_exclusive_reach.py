"""Do the exclusive flags ahead of a concordant cluster mark where it goes? And are they on land or at sea?

A curiosity check, not in the thesis, after check_era5_exclusive_flags.py. Reads the output of
build_era5_concordant_cube.py (`--cube`); the zone of every concordant cluster of `--kind` (long by default) is the one of
the thesis (scripts/era5_km_zone.py, `--radius-km`). An exclusive flag of a detector is a value of the zone that it alone
flags (one vote, after the closing). A value is ahead when it lies ahead of the centroid of the cluster at its timestamp
along the motion of the cluster (first to last centroid, as in check_era5_concordant_lead.py).

Reach: for every exclusive flag ahead at timestamp t, is its cell a cell of the same cluster at some timestamp in
(t, t + h], for h of `--hours`? Only the values with h hours of the cluster still to come count for h. The reference is
the values ahead that no detector flags, a share `--reference-share` drawn at random (`--seed`), as the reference of the
thesis (one in five). Rates are given by band of distance from the nearest cell of the cluster at the same timestamp, so
that a detector whose flags lie closer to the cluster is not favoured for that alone; and cluster by cluster, the share
of the exclusive flags of `--first` reached within the longest horizon against that of every other detector (the bands
up to `--near-km` together), with a two-sided sign test.

Land: with `--inputs-run`, a cell is land when its standardised `--soil` input varies over the test years (its standard
deviation over one timestamp in `--every` above 1e-3): over the sea the soil moisture of ERA5 has no physical meaning.
The type of the exclusive flags of every detector (the variable with the largest mean |standardised value|, the rule of
the thesis) is given again over the land cells only, with the share of the exclusive flags on land.

    python scripts/check_era5_exclusive_reach.py --cube outputs/concordate_5pc_regola_tesi \\
        --inputs-run outputs/stgan_optuna27_scored --csv outputs/concordate_5pc_regola_tesi/reach_r300_long.csv
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
from scripts.check_era5_concordant_lead import CHUNK, EARTH_KM
from scripts.check_era5_exclusive_flags import BANDS_KM, abs_profile, nearest_km, unit_vectors
from scripts.era5_km_zone import owner_lattice, radius_box

REFERENCE = "reference"


def reached(own, steps):
    """For every horizon of `steps` (in timestamps): whether the cell of every value [time, row, column] is a cell of the
    cluster `own` at some later timestamp up to that many steps on."""
    times = len(own)
    out, acc = {}, np.zeros_like(own)
    for s in range(1, max(steps) + 1):
        if s < times:
            acc[:times - s] |= own[s:]
        if s in steps:
            out[s] = acc.copy()
    return out


def band_of(km):
    """Index of the band of BANDS_KM of every distance."""
    return np.searchsorted(np.asarray(BANDS_KM[1:]), km, side="right")


def land_cells(inputs, first, times, soil_index, every):
    """Cells whose standardised soil input varies over the scored timestamps (land), from one timestamp in `every`."""
    sample = np.asarray(inputs[first:first + times:every, :, soil_index], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.nanstd(sample, axis=0) > 1e-3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cube", type=Path, required=True, help="Output directory of build_era5_concordant_cube.py")
    parser.add_argument("--kind", choices=("analysed", "long", "all"), default="long")
    parser.add_argument("--radius-km", type=float, default=300.0)
    parser.add_argument("--hours", nargs="+", type=float, default=[3, 6, 12, 24])
    parser.add_argument("--first", default="Gn")
    parser.add_argument("--near-km", type=float, default=165.0, help="Bands up to this distance in the paired comparison")
    parser.add_argument("--reference-share", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inputs-run", type=Path, help="Scored run of the model: land cells and types on land")
    parser.add_argument("--soil", default="volumetric_soil_water_layer_1")
    parser.add_argument("--every", type=int, default=50)
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
    sphere = unit_vectors(latitudes[:, None], longitudes[None, :])
    timestamps = pd.DatetimeIndex(np.load(run / "test_timestamps.npy"))
    step_hours = float(np.median(np.diff(timestamps.asi8))) / 3.6e12
    steps = sorted({max(int(round(h / step_hours)), 1) for h in args.hours})
    table = pd.read_csv(args.cube / "clusters.csv")
    votes = np.load(run / "scores" / "anomaly_mean.npy", mmap_mode="r")
    times = len(timestamps)
    flags = {name: np.load(args.cube / "flags" / f"{name}.npy", mmap_mode="r") for name in names}
    if args.first not in names:
        raise ValueError(f"--first {args.first} is not a detector of the cube")

    inputs, features, land, cell_index = None, [], None, None
    if args.inputs_run is not None:
        from scripts.check_era5_event_zones import inputs_of
        inputs, first_row, features = inputs_of(args.inputs_run, timestamps)
        if args.soil not in features:
            raise ValueError(f"--soil {args.soil} is not an input of the run ({', '.join(features)})")
        land_flat = land_cells(inputs, first_row, times, features.index(args.soil), args.every)
        land = np.zeros(shape, dtype=bool)
        land[rows, cols] = land_flat
        cell_index = np.full(shape, -1, dtype=np.int64)
        cell_index[rows, cols] = np.arange(len(rows))
        print(f"[reach] land cells (the standardised {args.soil} varies): {land_flat.sum()} of {len(rows)}")

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

    rng = np.random.default_rng(args.seed)
    n_bands = len(BANDS_KM) - 1
    who = names + [REFERENCE]
    pooled = {(name, s): np.zeros((n_bands, 2), dtype=np.int64) for name in who for s in steps}  # [band, (valid, reached)]
    per_cluster, types = [], []
    chosen = table if args.kind == "all" else table[table["kind"] == args.kind]
    near_band = int(np.searchsorted(np.asarray(BANDS_KM[1:]), args.near_km, side="left")) + 1  # bands below near-km
    longest = max(steps)
    for done, row in enumerate(chosen.itertuples(index=False), 1):
        t0, t1, r0, r1, c0, c1 = radius_box(row, reach_rows, reach_cols, shape)
        near = owner[t0:t1 + 1, r0:r1 + 1, c0:c1 + 1] == row.cluster
        own = labels[t0:t1 + 1, r0:r1 + 1, c0:c1 + 1] == row.cluster
        count = np.full((t1 - t0 + 1, *shape), np.nan, dtype=np.float32)
        count[:, rows, cols] = np.asarray(votes[t0:t1 + 1])
        count = count[:, r0:r1 + 1, c0:c1 + 1]
        alone, nobody = count == 1, count == 0
        y = np.repeat(EARTH_KM * latitudes[r0:r1 + 1][:, None], c1 - c0 + 1, axis=1)
        x = EARTH_KM * longitudes[c0:c1 + 1][None, :] * np.cos(latitudes[r0:r1 + 1][:, None])
        centre = lambda mask: np.array([x[mask].mean(), y[mask].mean()])
        first, last = row.t0 - t0, row.t1 - t0
        motion = centre(own[last]) - centre(own[first])
        if np.hypot(*motion) == 0:
            continue
        motion = motion / np.hypot(*motion)
        ahead = np.zeros_like(own)
        for t in range(len(own)):
            if own[t].any():
                ahead[t] = (((x - centre(own[t])[0]) * motion[0] + (y - centre(own[t])[1]) * motion[1]) > 0) & near[t] & ~own[t]
        later = reached(own, steps)
        box_sphere = sphere[r0:r1 + 1, c0:c1 + 1]
        block_abs = boxed_cells = None
        if inputs is not None:
            block_abs = np.abs(np.asarray(inputs[first_row + t0:first_row + t1 + 1], dtype=np.float32))
            boxed_cells = cell_index[r0:r1 + 1, c0:c1 + 1]
            box_land = land[r0:r1 + 1, c0:c1 + 1]
        masks = {}
        for name in names:
            block = np.zeros((t1 - t0 + 1, *shape), dtype=bool)
            block[:, rows, cols] = np.unpackbits(np.asarray(flags[name][t0:t1 + 1]), axis=1, count=len(rows)).astype(bool)
            exclusive = block[:, r0:r1 + 1, c0:c1 + 1] & near & alone
            masks[name] = exclusive & ahead
            if block_abs is not None and exclusive.any():
                on_land = exclusive & box_land[None]
                sums, n = abs_profile(block_abs, on_land, boxed_cells)
                types.append({"cluster": row.cluster, "detector": name, "exclusive": int(exclusive.sum()),
                              "on_land": int(on_land.sum()),
                              "type_land": features[int(np.argmax(sums / max(n, 1)))] if n else "", "z_values": n,
                              **{f"z_{f}": float(v) for f, v in zip(features, sums)}})
        reference = ahead & nobody
        reference &= rng.random(reference.shape) < args.reference_share
        masks[REFERENCE] = reference
        cluster_rates = {}
        for name, mask in masks.items():
            if not mask.any():
                continue
            band = np.full(mask.shape, -1, dtype=np.int64)
            for t in np.flatnonzero(mask.any(axis=(1, 2))):
                band[t][mask[t]] = band_of(nearest_km(box_sphere[mask[t]], box_sphere[own[t]]))
            for s in steps:
                valid = mask.copy()
                valid[len(own) - s:] = False  # fewer than s timestamps of the cluster to come
                hit = valid & later[s]
                pooled[(name, s)][:, 0] += np.bincount(band[valid], minlength=n_bands)[:n_bands]
                pooled[(name, s)][:, 1] += np.bincount(band[hit], minlength=n_bands)[:n_bands]
                if s == longest:
                    close = valid & (band < near_band)
                    if close.any():
                        cluster_rates[name] = (close & later[s]).sum() / close.sum()
        per_cluster.append({"cluster": row.cluster, "hours": row.hours, **{f"rate_{k}": v for k, v in cluster_rates.items()}})
        if done % 500 == 0:
            release_pages(votes, *flags.values(), *([] if inputs is None else [inputs]))
            print(f"[reach] {done} of {len(chosen)} clusters", flush=True)
    for array in flags.values():
        array._mmap.close()
    votes._mmap.close()
    if inputs is not None:
        inputs._mmap.close()

    labels_band = [f"{int(a)}-{'inf' if np.isinf(b) else int(b)} km" for a, b in zip(BANDS_KM[:-1], BANDS_KM[1:])]
    rows_out = []
    for (name, s), arr in pooled.items():
        for b in range(n_bands):
            if arr[b, 0]:
                rows_out.append({"who": name, "hours": s * step_hours, "band": labels_band[b], "values": int(arr[b, 0]),
                                 "reached %": 100 * arr[b, 1] / arr[b, 0]})
    rates = pd.DataFrame(rows_out)
    if args.csv is not None:
        rates.to_csv(args.csv, index=False)
        pd.DataFrame(per_cluster).to_csv(args.csv.with_name(args.csv.stem + "_clusters.csv"), index=False)
        if types:
            pd.DataFrame(types).to_csv(args.csv.with_name(args.csv.stem + "_land.csv"), index=False)
        print(f"[reach] written {args.csv}")
    print(f"\n[reach] {len(chosen)} clusters ({args.kind}), zone {args.radius_km:g} km. Values ahead of the centroid of "
          "the cluster along its motion: % whose cell is in the cluster within the hours (only values with that many "
          f"hours of the cluster to come), by distance from the cluster; reference = values nobody flags "
          f"({args.reference_share:g} drawn)")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        for s in steps:
            part = rates[rates["hours"] == s * step_hours].pivot(index="band", columns="who", values="reached %")
            part = part.reindex([b for b in labels_band if b in part.index])[[w for w in who if w in part.columns]]
            print(f"\n  within {s * step_hours:g} h")
            print(part.round(1).to_string())
        counts = rates[rates["hours"] == longest * step_hours].pivot(index="band", columns="who", values="values")
        print("\n  values counted (within the longest horizon)")
        print(counts.reindex([b for b in labels_band if b in counts.index]).fillna(0).astype(int).to_string())

    paired = pd.DataFrame(per_cluster)
    key = f"rate_{args.first}"
    if key in paired:
        print(f"\n[reach] cluster by cluster, bands below {args.near_km:g} km, within {longest * step_hours:g} h: "
              f"% of the clusters where the exclusive flags of {args.first} are reached more often than those of the other")
        for name in who:
            other = f"rate_{name}"
            if name == args.first or other not in paired:
                continue
            both = paired[[key, other]].dropna()
            d = both[key] - both[other]
            k, n = int((d > 0).sum()), int((d != 0).sum())
            p = binomtest(k, n).pvalue if n else np.nan
            print(f"  {name:10s} {100 * k / max(n, 1):5.1f}% of {n} clusters, sign test p {p:.2g}")

    if types:
        t = pd.DataFrame(types)
        print("\n[reach] exclusive flags on land (the standardised soil input varies): share of them, and the three most "
              "common types over the land cells only (% of the clusters)")
        for name, g in t.groupby("detector", sort=False):
            share = 100 * g["on_land"].sum() / max(g["exclusive"].sum(), 1)
            common = g.loc[g["type_land"] != "", "type_land"].value_counts(normalize=True).head(3)
            print(f"  {name:10s} on land {share:5.1f}% | " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in common.items()))
        z = [f"z_{f}" for f in features]
        profile = t.groupby("detector", sort=False)[z + ["z_values"]].sum()
        profile = profile[z].div(profile["z_values"].clip(lower=1), axis=0)
        profile.columns = features
        print("\n[reach] mean |standardised value| of each input over the exclusive flags on land")
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(profile.round(2).T.to_string())


if __name__ == "__main__":
    main()
