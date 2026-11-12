"""What is in the values a detector flags around a concordant cluster, outside the concordant cube?

Reads the output of build_era5_concordant_cube.py (`--cube`). For every analysed concordant cluster, the excess of a
detector (`--detectors`, the model and MCD by default) is the set of values it flags (after its closing) inside the zone
of the cluster (check_era5_concordant_boxes.zone) and outside the concordant cube. The values of the excess are split
- by position: before the cluster starts, ahead of it or behind it along its motion while it lasts (the side of the
  centroid of the cluster at that timestamp, along the motion from its first to its last centroid, as the along-track
  error of check_era5_concordant_lead.py), after it ends;
- by who else flags them: the detector alone (exclusive flags) or with fewer other detectors than the vote needs;
- for `--probability NAME`, by the probability that a draw of the dropout lies above its threshold: [0.5, 0.7), [0.7,
  0.9), [0.9, 1].
The reference is the values of the same zones, in the same positions, that no detector flags.

For every group, at the timestamp of the value and `--hours-after` later, from the standardised inputs of `--inputs-run`
(the scored run of the model; the scarto standardizzato of the thesis): the mean |value| over the input variables, the
mean number of variables beyond 1 and beyond 2 standard deviations, the share with at least one beyond 2, the mean of the
largest |value|; and, at the timestamp of the value, which variable is the farthest from its normal. These variables are
those the detectors read: they describe the departure each group shows, they cannot say whether it is weather.

With `--unseen-dir` (folders of monthly NetCDF files of variables no score reads: CAPE and gust, the judges of
download_era5.py --judge-variables), as a preliminary check: the share of values where each of them is in its highest
`--unusual-percent` for the cell, the calendar month and the hour (lowest for `--low-variables`, both sides for
`--both-variables`; check_era5_end_lead.judges_of_month), at the same timestamps, against the reference, also year by year.
Only the years for which every folder holds the twelve months are read; they are printed.

    python scripts/check_era5_concordant_excess.py --cube outputs/concordate_5pc_regola_tesi \\
        --inputs-run outputs/stgan_optuna27_scored --probability Gn \\
        --unseen-dir data/era5_independent/independent data/era5_judges/judges --csv excess.csv
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path.cwd()))  # after the copy next to the script: a copy run from the checkout imports itself
import numpy as np
import pandas as pd

from physiq_pv.anomaly_detection.stgan.loading import release_pages
from scripts.check_era5_concordant_boxes import label_clusters, zone
from scripts.check_era5_concordant_lead import EARTH_KM, probability_block, probability_run

CHUNK = 256
POSITIONS = ("before", "ahead", "behind", "after")
def p_classes(levels):
    """Classes of the probability between 0.5 (every flag is above it) and 1, cut at the levels of
    check_era5_concordant_lead.py, so that the two read on the same scale."""
    edges = [0.5, *sorted(levels), 1.000001]
    return list(zip(edges[:-1], edges[1:]))


def p_label(low, high):
    return f"p {low:g}-{min(high, 1):g}"


def positions_of(cells, own, x, y, first, last):
    """Position of every value (indices `cells` = (t, row, column) in the block) against the cluster `own`: 0 before it
    starts, 1 ahead, 2 behind along its motion while it lasts, 3 after it ends. Ahead / behind are taken against the
    centroid of the cluster at the nearest of its timestamps; a cluster that does not move puts all its values ahead."""
    centre = lambda t: np.array([x[own[t]].mean(), y[own[t]].mean()])
    start, end = centre(first), centre(last)
    motion = end - start
    distance = np.hypot(*motion)
    unit = motion / distance if distance > 0 else np.array([0.0, 0.0])
    t, r, c = cells
    out = np.where(t < first, 0, np.where(t > last, 3, 1)).astype(np.int8)
    during = np.nonzero(out == 1)[0]
    if len(during):
        centres = np.array([centre(s) for s in range(first, last + 1)])
        offset = np.stack([x[r[during], c[during]], y[r[during], c[during]]], axis=1) - centres[t[during] - first]
        out[during] = np.where(offset @ unit < 0, 2, 1)
    return out


def judge_years(roots):
    """The years for which every folder holds a file for each of the twelve months."""
    found = None
    for root in roots:
        years = set()
        for folder in Path(root).iterdir():
            if folder.is_dir() and folder.name.isdigit():
                # <prefix>_<year>_<month>.nc, or <prefix>_<year>_<month>__<member>.nc when the download split it
                months = {match.group(1) for path in folder.glob("*.nc")
                          if (match := re.search(rf"_{folder.name}_(\d{{2}})(?:__|\.nc)", path.name))}
                if all(f"{m:02d}" in months for m in range(1, 13)):
                    years.add(int(folder.name))
        found = years if found is None else found & years
    return sorted(found or [])


def judge_bits(roots, timestamps, latitudes, longitudes, years, percent, low_names, both_names):
    """Where every judge is unusual and where it is defined, as bit masks [timestamps, cells] (uint16, one bit a judge),
    with -1 rows outside `years`; and the labels of the judges."""
    from scripts.check_era5_end_lead import judges_of_month

    unusual = defined = None
    labels = None
    for month in range(1, 13):
        positions = np.nonzero((timestamps.month == month) & np.isin(timestamps.year, years))[0]
        if len(positions) == 0:
            continue
        names, u, d = judges_of_month(roots, month, positions, timestamps, latitudes, longitudes, percent,
                                      low_names, both_names)
        if labels is None:
            labels = names
            if len(labels) > 16:
                raise ValueError("More than 16 judges")
            unusual = np.zeros((len(timestamps), len(latitudes)), dtype=np.uint16)
            defined = np.zeros_like(unusual)
        weights = (1 << np.arange(len(labels))).astype(np.uint16)
        unusual[positions] = (u.astype(np.uint16) * weights).sum(axis=-1).astype(np.uint16)
        defined[positions] = (d.astype(np.uint16) * weights).sum(axis=-1).astype(np.uint16)
        print(f"[excess] judges of month {month} read", flush=True)
    return labels, unusual, defined


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cube", type=Path, required=True)
    parser.add_argument("--detectors", nargs="+", default=["Gn", "mcd"])
    parser.add_argument("--inputs-run", type=Path, required=True, help="Scored run of the model (standardised inputs)")
    parser.add_argument("--probability", help="Detector whose excess is split by the probability of the draws")
    parser.add_argument("--levels", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.9, 0.95],
                        help="Cuts of the classes of the probability (the levels of check_era5_concordant_lead.py)")
    parser.add_argument("--hours-after", nargs="+", type=float, default=[0, 3, 6, 12, 24])
    parser.add_argument("--unseen-dir", type=Path, nargs="*", default=[])
    parser.add_argument("--unusual-percent", type=float, default=5)
    parser.add_argument("--low-variables", nargs="*", default=["mn2t"])
    parser.add_argument("--both-variables", nargs="*", default=["d2m"])
    parser.add_argument("--reference-share", type=float, default=0.2,
                        help="Share of the reference values read, drawn at random (they are many)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    from scripts.check_era5_event_zones import inputs_of

    summary = json.loads((args.cube / "summary.json").read_text(encoding="utf-8"))
    needed = summary["vote"]["votes_needed"]
    pad_steps, pad_cells = summary["padding"]["timestamps"], summary["padding"]["cells"]
    run = args.cube / "votes"
    grid = json.loads((run / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    cell_of = np.full(shape, -1, dtype=np.int64)
    cell_of[rows, cols] = np.arange(len(rows))
    lat_rad = np.radians(np.asarray(grid["latitudes"], dtype=np.float64))
    lon_rad = np.radians(np.asarray(grid["longitudes"], dtype=np.float64))
    timestamps = pd.DatetimeIndex(np.load(run / "test_timestamps.npy"))
    times = len(timestamps)
    step_hours = float(np.median(np.diff(timestamps.asi8))) / 3.6e12
    lags = [int(round(h / step_hours)) for h in args.hours_after]
    table = pd.read_csv(args.cube / "clusters.csv")
    votes = np.load(run / "scores" / "anomaly_mean.npy", mmap_mode="r")
    flags = {name: np.load(args.cube / "flags" / f"{name}.npy", mmap_mode="r") for name in args.detectors}
    inputs, first_row, variables = inputs_of(args.inputs_run, timestamps)
    variables = variables or [f"v{i}" for i in range(inputs.shape[2])]
    probability = None if args.probability is None else probability_run(args.probability, summary, timestamps)

    labels_j, unusual, defined, years = [], None, None, []
    if args.unseen_dir:
        years = judge_years(args.unseen_dir)
        if not years:
            raise ValueError("No year with all twelve months in every judge folder")
        cell_lat = np.asarray(grid["latitudes"], dtype=np.float64)[rows]
        cell_lon = np.asarray(grid["longitudes"], dtype=np.float64)[cols]
        labels_j, unusual, defined = judge_bits(args.unseen_dir, timestamps, cell_lat, cell_lon, years,
                                                args.unusual_percent, args.low_variables, args.both_variables)
        print(f"[excess] judges over the years {years[0]}-{years[-1]} ({len(years)} years): {', '.join(labels_j)}",
              flush=True)
    year_index = {y: i for i, y in enumerate(years)}
    year_of = np.array([year_index.get(y, -1) for y in timestamps.year])

    concordant = np.zeros((times, len(rows)), dtype=bool)
    with np.errstate(invalid="ignore"):
        for start in range(0, times, CHUNK):
            concordant[start:start + CHUNK] = np.asarray(votes[start:start + CHUNK]) >= needed
    release_pages(votes)
    labels, sizes = label_clusters(concordant, rows, cols, shape, summary["overlap"])
    del concordant
    if len(sizes) != len(table) or not np.array_equal(sizes, table["values"].to_numpy()):
        raise ValueError("The clusters labelled again differ from clusters.csv: not the same cube or rule")

    n_var, n_judge = inputs.shape[2], len(labels_j)
    stats = defaultdict(lambda: np.zeros((len(lags), 6)))  # values, mean|z|, n>1, n>2, any>2, max|z|
    farthest = defaultdict(lambda: np.zeros(n_var + 1, dtype=np.int64))  # the last: no variable beyond 1
    judged = defaultdict(lambda: np.zeros((len(lags), max(len(years), 1), max(n_judge, 1), 2), dtype=np.int64))

    def add(key, cells, t0, z_block, max_t):
        t, r, c = cells
        for li, lag in enumerate(lags):
            keep = t + lag <= max_t
            if not keep.any():
                continue
            z = np.abs(z_block[t[keep] + lag, r[keep], c[keep]])
            finite = np.isfinite(z).all(axis=1)
            z = z[finite]
            if len(z):
                s = stats[key][li]
                s += [len(z), z.mean(axis=1).sum(), (z > 1).sum(), (z > 2).sum(), (z > 2).any(axis=1).sum(),
                      z.max(axis=1).sum()]
                if lag == 0:
                    which = np.where(z.max(axis=1) > 1, z.argmax(axis=1), n_var)
                    farthest[key] += np.bincount(which, minlength=n_var + 1)
            if n_judge:
                g = t0 + t[keep] + lag
                cells_flat = cell_of[r[keep], c[keep]]
                y = year_of[g]
                inside = y >= 0
                if inside.any():
                    shifts = np.arange(n_judge, dtype=np.uint16)
                    u = (unusual[g[inside], cells_flat[inside]][:, None] >> shifts) & 1
                    d = (defined[g[inside], cells_flat[inside]][:, None] >> shifts) & 1
                    index = (y[inside][:, None] * n_judge + np.arange(n_judge)[None, :]).ravel()
                    size = len(years) * n_judge
                    judged[key][li, :, :, 0] += np.bincount(index, weights=(u & d).ravel(),
                                                            minlength=size).reshape(len(years), n_judge).astype(np.int64)
                    judged[key][li, :, :, 1] += np.bincount(index, weights=d.ravel(),
                                                            minlength=size).reshape(len(years), n_judge).astype(np.int64)

    chosen = table[table["kind"] == "analysed"]
    max_lag = max(lags)
    for done, row in enumerate(chosen.itertuples(index=False), 1):
        t0, t1, r0, r1, c0, c1 = row.pad_t0, row.pad_t1, row.pad_row0, row.pad_row1, row.pad_col0, row.pad_col1
        e = (max(t0 - pad_steps, 0), min(t1 + pad_steps, times - 1), max(r0 - pad_cells, 0),
             min(r1 + pad_cells, shape[0] - 1), max(c0 - pad_cells, 0), min(c1 + pad_cells, shape[1] - 1))
        extended = labels[e[0]:e[1] + 1, e[2]:e[3] + 1, e[4]:e[5] + 1]
        near = zone(extended, row.cluster, pad_steps, pad_cells)
        near = near[t0 - e[0]:t1 - e[0] + 1, r0 - e[2]:r1 - e[2] + 1, c0 - e[4]:c1 - e[4] + 1]
        own = extended[t0 - e[0]:t1 - e[0] + 1, r0 - e[2]:r1 - e[2] + 1, c0 - e[4]:c1 - e[4] + 1] == row.cluster
        first, last = row.t0 - t0, row.t1 - t0
        y_km = np.repeat(EARTH_KM * lat_rad[r0:r1 + 1][:, None], c1 - c0 + 1, axis=1)
        x_km = EARTH_KM * lon_rad[c0:c1 + 1][None, :] * np.cos(lat_rad[r0:r1 + 1][:, None])
        tz = min(t1 + max_lag, times - 1)  # the inputs are read up to the longest lag after the box
        block = np.full((tz - t0 + 1, *shape), np.nan, dtype=np.float32)
        block[:, rows, cols] = np.asarray(votes[t0:tz + 1])
        vote_box = block[:t1 - t0 + 1, r0:r1 + 1, c0:c1 + 1]
        z_full = np.full((tz - t0 + 1, *shape, n_var), np.nan, dtype=np.float32)
        z_full[:, rows, cols] = np.asarray(inputs[first_row + t0:first_row + tz + 1], dtype=np.float32)
        z_block = z_full[:, r0:r1 + 1, c0:c1 + 1]
        del z_full
        valid = np.isfinite(vote_box)
        concordant_box = valid & (vote_box >= needed)
        reference = near & valid & (vote_box == 0)
        cells = np.nonzero(reference)
        if args.reference_share < 1:  # a random share of the reference, the same every run
            keep = np.random.default_rng([args.seed, int(row.cluster)]).random(len(cells[0])) < args.reference_share
            cells = tuple(a[keep] for a in cells)
        where = positions_of(cells, own, x_km, y_km, first, last)
        for pi, position in enumerate(POSITIONS):
            pick = where == pi
            add(("reference", position, "-", "-"), tuple(a[pick] for a in cells), t0, z_block, tz - t0)
        p_box = None if probability is None else probability_block(*probability, t0, t1, rows, cols, shape,
                                                                   r0, r1, c0, c1)
        for name in args.detectors:
            m = np.zeros((t1 - t0 + 1, *shape), dtype=bool)
            m[:, rows, cols] = np.unpackbits(np.asarray(flags[name][t0:t1 + 1]), axis=1, count=len(rows)).astype(bool)
            m = m[:, r0:r1 + 1, c0:c1 + 1]
            excess = m & near & valid & ~concordant_box
            cells = np.nonzero(excess)
            if len(cells[0]) == 0:
                continue
            where = positions_of(cells, own, x_km, y_km, first, last)
            alone = vote_box[cells] == 1
            p = None if (p_box is None or name != args.probability) else p_box[cells]
            for pi, position in enumerate(POSITIONS):
                for who, pick_who in (("alone", alone), ("with others", ~alone)):
                    pick = (where == pi) & pick_who
                    if p is None:
                        add((name, position, who, "all"), tuple(a[pick] for a in cells), t0, z_block, tz - t0)
                    else:
                        for low, high in p_classes(args.levels):
                            sub = pick & (p >= low) & (p < high)
                            add((name, position, who, p_label(low, high)), tuple(a[sub] for a in cells), t0, z_block,
                                tz - t0)
        if done % 2000 == 0:
            release_pages(votes, inputs, *flags.values(), *([] if probability is None else probability[:2]))
            print(f"[excess] {done} of {len(chosen)} clusters", flush=True)

    # Tables. The accumulators hold sums, so the groups are pooled by adding them.
    def pooled_stats(grouping):
        out = defaultdict(lambda: np.zeros((len(lags), 6)))
        for key, s in stats.items():
            out[grouping(key)] += s
        return out

    def stats_frame(pooled, names):
        records = []
        for key, s in pooled.items():
            for li, hours in enumerate(args.hours_after):
                n = s[li, 0]
                if n:
                    records.append({**dict(zip(names, key)), "hours": hours, "values": int(n), "mean |z|": s[li, 1] / n,
                                    "vars > 1": s[li, 2] / n, "vars > 2": s[li, 3] / n, "any > 2 %": 100 * s[li, 4] / n,
                                    "max |z|": s[li, 5] / n})
        return pd.DataFrame(records)

    full = stats_frame(stats, ["detector", "position", "who", "p"])
    by_position = stats_frame(pooled_stats(lambda k: (k[0], k[1])), ["detector", "position"])
    by_who = stats_frame(pooled_stats(lambda k: (k[0], k[1], k[2])), ["detector", "position", "who"])
    now = args.hours_after[0]
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.2f}".format):
        if not by_position.empty:
            print(f"\n[excess] {len(chosen)} analysed clusters. Excess of each detector (flagged inside the zone, outside "
                  f"the concordant cube) and the reference (same zones and positions, flagged by no detector; a share "
                  f"{args.reference_share:g} of them at random): mean of the largest |standardised input| at the timestamp "
                  f"of the value and later (hours)")
            print(by_position.pivot_table(index=["detector", "position"], columns="hours", values="max |z|", sort=False)
                  .to_string())
            print("\n[excess] at the timestamp of the value: values, mean |z|, variables beyond 1 and 2, share with one "
                  "beyond 2")
            print(by_position[by_position["hours"] == now].drop(columns="hours").to_string(index=False))
            print("\n[excess] by who else flags them ('alone' = exclusive flags of the detector), at the timestamp of the "
                  "value")
            print(by_who[by_who["hours"] == now].drop(columns="hours").to_string(index=False))
            if args.probability:
                part = stats_frame(pooled_stats(lambda k: (k[0], k[1], k[3])), ["detector", "position", "p"])
                part = part[part["detector"] == args.probability]
                print(f"\n[excess] {args.probability} by the probability of the draws: mean of the largest |z| at the "
                      f"timestamp of the value and later; values at the timestamp of the value")
                table_p = part.pivot_table(index=["position", "p"], columns="hours", values="max |z|", sort=False)
                table_p.insert(0, "values", part[part["hours"] == now].set_index(["position", "p"])["values"])
                print(table_p.to_string())
            share = defaultdict(lambda: np.zeros(n_var + 1, dtype=np.int64))
            for key, counts in farthest.items():
                share[(key[0], key[1])] += counts
            share = pd.DataFrame({key: 100 * c / max(c.sum(), 1) for key, c in share.items()},
                                 index=[*variables, "none beyond 1"]).T
            print("\n[excess] the variable farthest from its normal, % of the values (the last column: no variable beyond "
                  "1 standard deviation), at the timestamp of the value")
            print(share.round(1).to_string())
        judge_frame = pd.DataFrame()
        if n_judge:
            pooled = defaultdict(lambda: np.zeros((len(lags), len(years), n_judge, 2), dtype=np.int64))
            for key, counts in judged.items():
                pooled[(key[0], key[1])] += counts
            records = []
            for (det, pos), counts in pooled.items():
                for li, hours in enumerate(args.hours_after):
                    for j, label in enumerate(labels_j):
                        hit, total = counts[li, :, j, 0], counts[li, :, j, 1]
                        records.append({"detector": det, "position": pos, "hours": hours, "judge": label,
                                        "values": int(total.sum()), "unusual %": 100 * hit.sum() / max(total.sum(), 1),
                                        **{f"y{years[i]}": 100 * hit[i] / total[i] if total[i] else np.nan
                                           for i in range(len(years))}})
            judge_frame = pd.DataFrame(records)
            print(f"\n[excess] PRELIMINARY: variables no score reads, share of values in their {args.unusual_percent:g}% "
                  f"most extreme for the cell, month and hour (chance {args.unusual_percent:g}%), years "
                  f"{years[0]}-{years[-1]}, at the timestamp of the value")
            here = judge_frame[judge_frame["hours"] == now]
            print(here.pivot_table(index=["detector", "position"], columns="judge", values="unusual %", sort=False)
                  .to_string(float_format="{:.1f}".format))
            print("\n[excess] PRELIMINARY: the same at every time after the value, for the model's excess and the "
                  "reference")
            print(judge_frame[judge_frame["detector"].isin([args.detectors[0], "reference"])]
                  .pivot_table(index=["detector", "position", "judge"], columns="hours", values="unusual %", sort=False)
                  .to_string(float_format="{:.1f}".format))
            lines = []
            for (det, pos), d in here[here["detector"] != "reference"].groupby(["detector", "position"], sort=False):
                for label in labels_j:
                    a = d[d["judge"] == label].iloc[0]
                    b = here[(here["detector"] == "reference") & (here["position"] == pos) & (here["judge"] == label)]
                    if b.empty:
                        continue
                    b = b.iloc[0]
                    above = sum(int(a[f"y{y}"] > b[f"y{y}"]) for y in years
                                if np.isfinite(a[f"y{y}"]) and np.isfinite(b[f"y{y}"]))
                    lines.append({"detector": det, "position": pos, "judge": label,
                                  "years above the reference": f"{above}/{len(years)}"})
            if lines:
                print("\n[excess] PRELIMINARY: years in which the excess is above the reference, at the timestamp of the "
                      "value")
                print(pd.DataFrame(lines).pivot_table(index=["detector", "position"], columns="judge",
                                                      values="years above the reference", aggfunc="first",
                                                      sort=False).to_string())
            if args.probability:
                # The judges of the excess of the model by class of the probability of the draws, next to the reference.
                by_p = defaultdict(lambda: np.zeros((len(lags), len(years), n_judge, 2), dtype=np.int64))
                for key, counts in judged.items():
                    if key[0] == args.probability or key[0] == "reference":
                        by_p[(key[1], key[3] if key[0] != "reference" else "reference")] += counts
                li = args.hours_after.index(now)
                rows_p = []
                for (pos, p), counts in by_p.items():
                    line = {"position": pos, "p": p, "values": int(counts[li, :, 0, 1].sum())}
                    for j, label in enumerate(labels_j):
                        line[label] = 100 * counts[li, :, j, 0].sum() / max(counts[li, :, j, 1].sum(), 1)
                    rows_p.append(line)
                table_p = pd.DataFrame(rows_p)
                order = {name: i for i, name in enumerate([*[p_label(*c) for c in p_classes(args.levels)], "reference"])}
                table_p = table_p.sort_values(["position", "p"], key=lambda s: s.map(order) if s.name == "p"
                                              else s.map({p: i for i, p in enumerate(POSITIONS)}))
                print(f"\n[excess] PRELIMINARY: the judges in the excess of {args.probability} by class of the probability "
                      f"of the draws, and in the reference, at the timestamp of the value (% in the most extreme "
                      f"{args.unusual_percent:g}%)")
                print(table_p.to_string(index=False, float_format="{:.1f}".format))
    if args.csv:
        full.to_csv(args.csv, index=False)
        if not judge_frame.empty:
            judge_frame.to_csv(args.csv.with_name(args.csv.stem + "_giudici.csv"), index=False)
        print(f"\nwritten {args.csv}")


if __name__ == "__main__":
    main()
