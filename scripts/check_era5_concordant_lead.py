"""Who flags a concordant cluster first, and who stops last?

Reads the output of build_era5_concordant_cube.py (`--cube`). For every concordant cluster of `--kind` (the analysed ones
by default) and every detector, the flags of the detector (after its closing) are taken inside the zone of the cluster
(check_era5_concordant_boxes.zone: the cluster dilated by the padding, nearest cluster) and joined with the rule of the
clusters of the thesis (touching cells at a timestamp, diagonals included; the same cell at consecutive timestamps). The
groups of flags that touch the cluster are its match, as detected and observed tracks are matched before their first and
last dates are compared (Bourdin et al. 2022, GMD, Sect. 2.4 and Fig. 8). The lead at the start is the first timestamp of
the match minus the first timestamp of the cluster, in hours (negative: the detector starts before the concordant
cluster); the lead at the end is the last timestamp of the match minus the last of the cluster (positive: it goes on
after). Both are bounded by the padding in time of the zone; the share at the bound is printed.

Where the cluster goes: the along-track and cross-track errors of tropical cyclone tracks (Leonardo and Colle 2020, MWR,
Sect. 2: positive along-track = the forecast is ahead along the observed motion). At every timestamp of the cluster, the
centroid of the matched flags of the detector minus the centroid of the cluster, projected on the motion of the cluster,
from its first to its last centroid (its mean translation, Davis et al. 2006b); the mean over the timestamps, in km.
Positive: the detector flags more where the cluster goes than where it has been. Given also by class of the mean speed
of the cluster (cells of 0.5 degrees every 3 h), every class shown, since a cluster that hardly moves has no direction.

Every detector of `--first` (the model by default) is then compared with every other detector cluster by cluster, on
the clusters both match: in how many it starts earlier, at the same timestamp or later, and stops later, together or
earlier, with a two-sided sign test on the clusters that are not ties. `--extra NAME=DIR[@component]` adds detectors that
do not vote (the forecast score P_n of Appendix B): each flags its highest share of the cube with its own threshold (that
of the validation years when its run names one, as for the others), is closed as the others were before the vote, and is
compared as they are.

`--probability NAME` (a detector of the cube scored with the Monte Carlo dropout, whose run keeps the spread of the draws
in the units of its score, anomaly_std.npy: the spread over the deviation of the climatology) gives every value the
probability that a draw lies above the threshold of NAME, p = Phi((score - threshold) / spread): the draws summarised by
their mean and spread and taken as Gaussian (Lakshminarayanan et al. 2017, Sect. 2.4; the ensemble smoothing and EMOS of
Gneiting et al. 2005). Every flag has p > 0.5 (its mean is above the threshold). For every level of `--levels` it adds a
detector NAME_pLEVEL, which does not vote: the flags of NAME (after its closing) with p at least the level. The comparison
is shown as a curve over the level, not at one cut (Leibig et al. 2017; Kendall and Gal 2017; Ovadia et al. 2019): the
share of the flags of NAME kept at each level, and how often each level starts earlier and stops later than every other
detector of the cube. A level keeps fewer flags than NAME, and fewer flags start later and stop earlier for that alone; so
every level has a control, NAME_randLEVEL: as many flags of NAME taken at random (`--seed`), and the curve also gives each
level against its control (stability tells something about the timing only if the level beats it). Last, where the match
of NAME starts before the cluster, the share of its flags before the start with p >= 0.9 and their median p, against all
the flags of its matches: are the flags that come first stable or uncertain?

    python scripts/check_era5_concordant_lead.py --cube outputs/concordate_5pc_regola_tesi --probability Gn --csv lead.csv
    python scripts/check_era5_concordant_lead.py --cube outputs/concordate_5pc_regola_tesi \\
        --extra Pn=outputs/optuna27/previsione/climatology/test/P --first Pn
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
from scipy.special import ndtr
from scipy.stats import binomtest

from physiq_pv.anomaly_detection.stgan.loading import release_pages
from scripts.build_era5_concordant_cube import close, segments
from scripts.check_era5_concordant_boxes import label_clusters, zone
from scripts.check_era5_dust_lead import score_file, score_limit
from scripts.check_era5_score_correlation import parse_score

CHUNK = 256  # timestamps read at a time
JOIN = np.zeros((3, 3, 3), dtype=bool)
JOIN[1] = True  # touching cells at the same timestamp, diagonals included
JOIN[0, 1, 1] = JOIN[2, 1, 1] = True  # the same cell at the timestamps before and after


def match_span(flags, own, with_mask=False):
    """First and last timestamp (indices in the block) of the groups of `flags` [time, row, column] that touch the cells
    `own` of the cluster, joined with the rule of the thesis; None when no group touches it. With `with_mask`, also the
    values of those groups."""
    groups, count = ndimage.label(flags, structure=JOIN)
    if count == 0:
        return None
    touching = np.unique(groups[own & flags])
    touching = touching[touching > 0]
    if len(touching) == 0:
        return None
    matched = np.isin(groups, touching)
    steps = np.nonzero(matched)[0]
    span = (int(steps.min()), int(steps.max()))
    return (*span, matched) if with_mask else span


def probability_block(scores, spread, threshold, t0, t1, rows, cols, shape, r0, r1, c0, c1):
    """Probability that a draw lies above the threshold, on a block [time, row, column] of the padded box."""
    mean = np.full((t1 - t0 + 1, *shape), np.nan)
    width = np.full((t1 - t0 + 1, *shape), np.nan)
    mean[:, rows, cols] = np.asarray(scores[t0:t1 + 1], dtype=np.float64)
    width[:, rows, cols] = np.asarray(spread[t0:t1 + 1], dtype=np.float64)
    mean, width = mean[:, r0:r1 + 1, c0:c1 + 1], width[:, r0:r1 + 1, c0:c1 + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(width > 0, ndtr((mean - threshold) / width), (mean > threshold).astype(float))
    p[~np.isfinite(mean)] = 0.0
    return p


def extra_flags(text, cube_summary, timestamps, rows, cols, shape):
    """Flags [time, cell] of a detector that does not vote, as the voting ones: its own threshold for the share of the
    cube, closed with the element of the cube when the cube closed every detector before the vote; packed bits."""
    name, directory, component = parse_score(text)
    manifest = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    own = pd.DatetimeIndex(np.load(directory / "test_timestamps.npy"))
    index = own.get_indexer(timestamps)
    scores = np.load(directory / score_file(manifest, component), mmap_mode="r")
    try:
        limit = score_limit(scores, cube_summary["vote"]["top_percent"])
        flagged = np.zeros((len(timestamps), *shape), dtype=bool)
        with np.errstate(invalid="ignore"):
            for start in range(0, len(timestamps), CHUNK):
                wanted = index[start:start + CHUNK]
                block = np.zeros((len(wanted), len(rows)), dtype=bool)
                present = wanted >= 0  # timestamps the detector has not scored are not flagged
                if present.any():
                    block[present] = np.asarray(scores[wanted[present]], dtype=np.float64) > limit
                flagged[start:start + CHUNK][:, rows, cols] = block
                release_pages(scores)
    finally:
        scores._mmap.close()
    closure = cube_summary["vote"]["closure"]
    if closure["where"] == "before":
        flagged = close(flagged, closure["steps"], closure["cells"], segments(timestamps))
    print(f"[lead] {name} (does not vote): threshold {limit:.4g}, {int((index < 0).sum())} timestamps of the cube not "
          f"scored by it", flush=True)
    return name, np.packbits(flagged[:, rows, cols], axis=1)


EARTH_KM = 6371.0
CELL_KM = 0.5 * np.pi / 180 * EARTH_KM  # 0.5 degrees of latitude
SPEED_CLASSES = ((0, 0.5), (0.5, 1), (1, 2), (2, np.inf))  # mean speed of a cluster, in cells every 3 h


def track_errors(own, matched, x, y, first, last):
    """Along-track and cross-track errors of the matched flags of a detector against the cluster (Leonardo and Colle
    2020: positive along-track = ahead along the motion; cross-track positive = to the right of it), in km, the mean over
    the timestamps of the cluster where both have values; the motion is the one of the cluster from its first to its
    last centroid, its mean translation (Davis et al. 2006b). Also the distance of that motion in km. NaN when the
    cluster does not move or no timestamp has both."""
    centre = lambda mask: np.array([x[mask].mean(), y[mask].mean()])
    start, end = centre(own[first]), centre(own[last])
    motion = end - start
    distance = float(np.hypot(*motion))
    if distance == 0:
        return np.nan, np.nan, distance
    unit = motion / distance
    along, cross = [], []
    for t in range(first, last + 1):
        if own[t].any() and matched[t].any():
            offset = centre(matched[t]) - centre(own[t])
            along.append(offset @ unit)
            cross.append(offset[0] * unit[1] - offset[1] * unit[0])
    if not along:
        return np.nan, np.nan, distance
    return float(np.mean(along)), float(np.mean(cross)), distance


def speed_class(distance_km, hours):
    """Label of the class of the mean speed of a cluster, in cells (0.5 degrees of latitude) every 3 h."""
    if not hours or not np.isfinite(distance_km):
        return "no motion"
    speed = distance_km / CELL_KM / (hours / 3)
    for low, high in SPEED_CLASSES:
        if low <= speed < high:
            return f"{low:g}-{high:g}" if np.isfinite(high) else f">={low:g}"
    return "no motion"


def probability_run(name, summary, timestamps):
    """The score and the spread of the draws of `name` (memory maps) and its threshold."""
    directory, _, component = summary["vote"]["detectors"][name].partition("@")
    directory = Path(directory)
    manifest = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if not manifest.get("anomaly_std_file"):
        raise ValueError(f"The run of {name} keeps no spread of the draws (anomaly_std_file)")
    if not pd.DatetimeIndex(np.load(directory / "test_timestamps.npy")).equals(timestamps):
        raise ValueError(f"The run of {name} is not over the timestamps of the cube")
    return (np.load(directory / score_file(manifest, component or "combined"), mmap_mode="r"),
            np.load(directory / manifest["anomaly_std_file"], mmap_mode="r"), float(summary["vote"]["thresholds"][name]))


def probable_flags(name, cube, summary, levels, timestamps, cells, seed):
    """Flags of `name` (after its closing, from the cube) whose probability that a draw lies above its threshold is at
    least each level, packed bits by level; the same flags thinned at random to as many values as each level keeps (the
    control at equal number: a subset starts later and stops earlier only for being smaller); and the share of its
    flags before the closing (score above the threshold) with p at least each level."""
    scores, spread, threshold = probability_run(name, summary, timestamps)
    closed = np.load(cube / "flags" / f"{name}.npy", mmap_mode="r")
    packed = {level: np.zeros(closed.shape, dtype=np.uint8) for level in levels}
    flagged, kept = 0, np.zeros(len(levels), dtype=np.int64)
    closed_total, closed_kept = 0, np.zeros(len(levels), dtype=np.int64)
    try:
        for start in range(0, len(timestamps), CHUNK):
            mean = np.asarray(scores[start:start + CHUNK], dtype=np.float64)
            width = np.asarray(spread[start:start + CHUNK], dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                p = np.where(width > 0, ndtr((mean - threshold) / width), (mean > threshold).astype(float))
            p[~np.isfinite(mean)] = 0.0
            own = np.unpackbits(np.asarray(closed[start:start + CHUNK]), axis=1, count=cells).astype(bool)
            with np.errstate(invalid="ignore"):
                raw = mean > threshold
            flagged += int(raw.sum())
            closed_total += int(own.sum())
            for index, level in enumerate(levels):
                kept[index] += int((raw & (p >= level)).sum())
                chosen = own & (p >= level)
                closed_kept[index] += int(chosen.sum())
                packed[level][start:start + CHUNK] = np.packbits(chosen, axis=1)
            release_pages(scores, spread, closed)
        # Random thinning to the same number of values as each level keeps, with a fixed seed per chunk.
        thinned = {level: np.zeros(closed.shape, dtype=np.uint8) for level in levels}
        for start in range(0, len(timestamps), CHUNK):
            own = np.unpackbits(np.asarray(closed[start:start + CHUNK]), axis=1, count=cells).astype(bool)
            draw = np.random.default_rng([seed, start]).random(own.shape)
            for index, level in enumerate(levels):
                share = closed_kept[index] / max(closed_total, 1)
                thinned[level][start:start + CHUNK] = np.packbits(own & (draw < share), axis=1)
            release_pages(closed)
    finally:
        for array in (scores, spread, closed):
            array._mmap.close()
    shares = {level: 100 * kept[i] / max(flagged, 1) for i, level in enumerate(levels)}
    print(f"[lead] {name}: probability that a draw lies above the threshold {threshold:.4g}; share of its flags with p at "
          f"least " + ", ".join(f"{level:g}: {share:.1f}%" for level, share in shares.items()), flush=True)
    derived = {f"{name}_p{level:g}": packed[level] for level in levels}
    controls = {f"{name}_rand{level:g}": thinned[level] for level in levels}
    return derived, controls, shares


def paired(table, first, other, column, earlier_is):
    """Clusters both match: share where `first` is earlier / tied / later on `column` (earlier = smaller value when
    earlier_is == 'smaller'), and the two-sided sign test on the clusters that are not ties."""
    a = table[table["detector"] == first].set_index("cluster")[column]
    b = table[table["detector"] == other].set_index("cluster")[column]
    both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if both.empty:
        return {"clusters": 0}
    wins = (both["a"] < both["b"]) if earlier_is == "smaller" else (both["a"] > both["b"])
    losses = (both["a"] > both["b"]) if earlier_is == "smaller" else (both["a"] < both["b"])
    ties = len(both) - int(wins.sum()) - int(losses.sum())
    decided = int(wins.sum()) + int(losses.sum())
    p = binomtest(int(wins.sum()), decided).pvalue if decided else np.nan
    return {"clusters": len(both), "first ahead %": 100 * wins.mean(), "same %": 100 * ties / len(both),
            "first behind %": 100 * losses.mean(), "sign test p": p}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cube", type=Path, required=True, help="Output directory of build_era5_concordant_cube.py")
    parser.add_argument("--kind", choices=("analysed", "small", "long", "all"), default="analysed")
    parser.add_argument("--first", nargs="+", default=["Gn"], help="The detectors compared with every other")
    parser.add_argument("--extra", nargs="*", default=[], help="NAME=DIR[@component] of detectors that do not vote")
    parser.add_argument("--probability", help="Detector of the cube whose flags are cut by the probability of the draws")
    parser.add_argument("--levels", nargs="+", type=float, default=[0.6, 0.7, 0.8, 0.9, 0.95])
    parser.add_argument("--seed", type=int, default=0, help="Seed of the random thinning, the control at equal number")
    parser.add_argument("--inputs-run", type=Path,
                        help="Scored run of the model (standardised inputs): the along-track error is also given for the "
                             "matched flags far from normal (an input beyond --far-sd) and for the moderate ones (none)")
    parser.add_argument("--far-sd", type=float, default=2.0)
    parser.add_argument("--exclusive", action="store_true",
                        help="Also the along-track error of the matched flags no other detector flags (one vote)")
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
    latitudes = np.radians(np.asarray(grid["latitudes"], dtype=np.float64))
    longitudes = np.radians(np.asarray(grid["longitudes"], dtype=np.float64))
    timestamps = pd.DatetimeIndex(np.load(run / "test_timestamps.npy"))
    step_hours = float(np.median(np.diff(timestamps.asi8))) / 3.6e12
    table = pd.read_csv(args.cube / "clusters.csv")
    votes = np.load(run / "scores" / "anomaly_mean.npy", mmap_mode="r")
    times = len(timestamps)

    flags = {name: np.load(args.cube / "flags" / f"{name}.npy", mmap_mode="r") for name in names}
    for text in args.extra:
        name, packed = extra_flags(text, summary, timestamps, rows, cols, shape)
        if name in flags:
            raise ValueError(f"{name} is already a detector of the cube")
        flags[name] = packed
    levels, controls, shares, probability = [], [], {}, None
    if args.probability is not None:
        if args.probability not in names:
            raise ValueError(f"--probability {args.probability} is not a detector of the cube")
        derived, thinned, shares = probable_flags(args.probability, args.cube, summary, args.levels, timestamps,
                                                  len(rows), args.seed)
        flags.update(derived)
        flags.update(thinned)
        levels, controls = list(derived), list(thinned)
        probability = probability_run(args.probability, summary, timestamps)  # for the flags before the cluster
    inputs = None
    if args.inputs_run is not None:
        from scripts.check_era5_event_zones import inputs_of
        inputs, first_row, _ = inputs_of(args.inputs_run, timestamps)
    detectors = list(flags)
    firsts = list(dict.fromkeys(args.first))
    missing = [name for name in firsts if name not in flags]
    if missing:
        raise ValueError(f"--first {', '.join(missing)} not among {', '.join(detectors)}")

    concordant = np.zeros((times, len(rows)), dtype=bool)
    with np.errstate(invalid="ignore"):
        for start in range(0, times, CHUNK):
            concordant[start:start + CHUNK] = np.asarray(votes[start:start + CHUNK]) >= needed
    release_pages(votes)
    labels, sizes = label_clusters(concordant, rows, cols, shape, summary["overlap"])
    del concordant
    if len(sizes) != len(table) or not np.array_equal(sizes, table["values"].to_numpy()):
        raise ValueError("The clusters labelled again differ from clusters.csv: not the same cube or rule")

    chosen = table if args.kind == "all" else table[table["kind"] == args.kind]
    out = []
    for done, row in enumerate(chosen.itertuples(index=False), 1):
        t0, t1, r0, r1, c0, c1 = row.pad_t0, row.pad_t1, row.pad_row0, row.pad_row1, row.pad_col0, row.pad_col1
        e = (max(t0 - pad_steps, 0), min(t1 + pad_steps, times - 1), max(r0 - pad_cells, 0),
             min(r1 + pad_cells, shape[0] - 1), max(c0 - pad_cells, 0), min(c1 + pad_cells, shape[1] - 1))
        extended = labels[e[0]:e[1] + 1, e[2]:e[3] + 1, e[4]:e[5] + 1]
        near = zone(extended, row.cluster, pad_steps, pad_cells)
        near = near[t0 - e[0]:t1 - e[0] + 1, r0 - e[2]:r1 - e[2] + 1, c0 - e[4]:c1 - e[4] + 1]
        own = extended[t0 - e[0]:t1 - e[0] + 1, r0 - e[2]:r1 - e[2] + 1, c0 - e[4]:c1 - e[4] + 1] == row.cluster
        first, last = row.t0 - t0, row.t1 - t0  # the cluster in the block
        # Cells of the box in km (equirectangular at the latitude of each cell), for the centroids.
        far = None
        if inputs is not None:  # values with an input beyond --far-sd standard deviations from normal, in the box
            largest = np.full((t1 - t0 + 1, *shape), np.nan, dtype=np.float32)
            largest[:, rows, cols] = np.abs(np.asarray(inputs[first_row + t0:first_row + t1 + 1],
                                                       dtype=np.float32)).max(axis=2)
            largest = largest[:, r0:r1 + 1, c0:c1 + 1]
            far, moderate = largest > args.far_sd, largest <= args.far_sd  # NaN is neither
        alone = None
        if args.exclusive:  # values flagged by one detector only (after the closing, as the votes of the cube)
            count = np.full((t1 - t0 + 1, *shape), np.nan, dtype=np.float32)
            count[:, rows, cols] = np.asarray(votes[t0:t1 + 1])
            alone = count[:, r0:r1 + 1, c0:c1 + 1] == 1
        y = np.repeat(EARTH_KM * latitudes[r0:r1 + 1][:, None], c1 - c0 + 1, axis=1)
        x = EARTH_KM * longitudes[c0:c1 + 1][None, :] * np.cos(latitudes[r0:r1 + 1][:, None])
        motion_km = None
        for name in detectors:
            block = np.zeros((t1 - t0 + 1, *shape), dtype=bool)
            block[:, rows, cols] = np.unpackbits(np.asarray(flags[name][t0:t1 + 1]), axis=1, count=len(rows)).astype(bool)
            stable_too = probability is not None and name == args.probability
            span = match_span(block[:, r0:r1 + 1, c0:c1 + 1] & near, own, with_mask=True)
            result = {"cluster": row.cluster, "detector": name, "values": row.values, "hours": row.hours,
                      "matched": span is not None}
            if span is not None:
                result["start_lead_h"] = (span[0] - first) * step_hours
                result["end_lead_h"] = (span[1] - last) * step_hours
                result["start_at_bound"] = span[0] == 0 and t0 < row.t0  # the zone stops it, not the detector
                result["end_at_bound"] = span[1] == t1 - t0 and t1 > row.t1
                along, cross, motion_km = track_errors(own, span[2], x, y, first, last)
                result["along_km"], result["cross_km"] = along, cross
                if far is not None:
                    result["along_far_km"] = track_errors(own, span[2] & far, x, y, first, last)[0]
                    result["along_moderate_km"] = track_errors(own, span[2] & moderate, x, y, first, last)[0]
                if alone is not None:
                    result["along_alone_km"] = track_errors(own, span[2] & alone, x, y, first, last)[0]
                if stable_too:
                    # How stable are the flags of the match that come before the cluster starts, against all of them.
                    p = probability_block(*probability, t0, t1, rows, cols, shape, r0, r1, c0, c1)
                    matched_values = span[2]
                    before = matched_values.copy()
                    before[first:] = False
                    result.update({"match_values": int(matched_values.sum()),
                                   "match_p90": int((matched_values & (p >= 0.9)).sum()),
                                   "match_p_median": float(np.median(p[matched_values])),
                                   "before_values": int(before.sum()),
                                   "before_p90": int((before & (p >= 0.9)).sum()),
                                   "before_p_median": float(np.median(p[before])) if before.any() else np.nan})
            out.append(result)
        if motion_km is None:  # no detector matched: the motion of the cluster from its own centroids
            motion_km = track_errors(own, own, x, y, first, last)[2]
        for result in out[-len(detectors):]:
            result["motion_km"] = motion_km
            result["speed_class"] = speed_class(motion_km, row.hours)
        if done % 2000 == 0:
            release_pages(votes, *[f for f in flags.values() if isinstance(f, np.memmap)],
                          *([] if probability is None else probability[:2]), *([] if inputs is None else [inputs]))
            print(f"[lead] {done} of {len(chosen)} clusters", flush=True)
    for array in flags.values():
        if isinstance(array, np.memmap):
            array._mmap.close()
    if probability is not None:
        for array in probability[:2]:
            array._mmap.close()
    if inputs is not None:
        inputs._mmap.close()
    votes._mmap.close()
    results = pd.DataFrame(out)
    if results.empty:
        print("[lead] no cluster")
        return
    for column in ("start_lead_h", "end_lead_h", "start_at_bound", "end_at_bound", "along_km", "cross_km",
                   "along_far_km", "along_moderate_km", "along_alone_km"):
        if column not in results:
            results[column] = np.nan

    matched = results[results["matched"]]
    per = matched.groupby("detector", sort=False).agg(
        median_start_h=("start_lead_h", "median"), before_start_pct=("start_lead_h", lambda s: 100 * (s < 0).mean()),
        start_at_bound_pct=("start_at_bound", lambda s: 100 * s.astype(float).mean()),
        median_end_h=("end_lead_h", "median"), after_end_pct=("end_lead_h", lambda s: 100 * (s > 0).mean()),
        end_at_bound_pct=("end_at_bound", lambda s: 100 * s.astype(float).mean()),
        median_along_km=("along_km", "median"), ahead_pct=("along_km", lambda s: 100 * (s.dropna() > 0).mean()),
        median_cross_km=("cross_km", "median"))
    per.insert(0, "matched_pct", 100 * results.groupby("detector", sort=False)["matched"].mean())
    print(f"\n[lead] {len(chosen)} clusters ({args.kind}); zone +-{pad_cells} cells, +-{pad_steps * step_hours:g} h; "
          f"lead at the start: first timestamp of the match minus that of the cluster (negative = before); at the end: last "
          f"minus last (positive = after); bounded by the zone. Along-track: centroid of the matched flags minus that of the "
          f"cluster along the motion of the cluster (first to last centroid), mean over its timestamps, km (positive = ahead, "
          f"Leonardo and Colle 2020); ahead_pct over the clusters that move")
    sides = (("start_lead_h", "smaller", "starts earlier than"), ("end_lead_h", "larger", "stops later than"),
             ("along_km", "larger", "is further ahead along the motion than"))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.1f}".format):
        print(per.reindex(detectors).to_string())
        classes = [f"{low:g}-{high:g}" if np.isfinite(high) else f">={low:g}" for low, high in SPEED_CLASSES]
        by_speed = matched.pivot_table(index="detector", columns="speed_class", values="along_km", aggfunc="median",
                                       sort=False).reindex(index=detectors, columns=[c for c in classes
                                                                                     if c in set(matched["speed_class"])])
        counts = results[results["detector"] == detectors[0]]["speed_class"].value_counts()
        print(f"\n[lead] median along-track error (km) by mean speed of the cluster (cells of 0.5 degrees every 3 h); "
              f"clusters per class: " + ", ".join(f"{c} {int(counts.get(c, 0))}" for c in [*classes, "no motion"]))
        print(by_speed.to_string())
        if args.exclusive:
            # The along-track error of the exclusive matched flags of every detector (no other detector flags them).
            exclusive = matched.groupby("detector", sort=False).agg(
                alone_km=("along_alone_km", "median"),
                alone_ahead_pct=("along_alone_km", lambda s: 100 * (s.dropna() > 0).mean()),
                alone_clusters=("along_alone_km", lambda s: int(s.notna().sum())))
            print("\n[lead] along-track error (km, median over the clusters) of the exclusive matched flags of each "
                  "detector (flagged by it alone); ahead % over the clusters with such flags")
            print(exclusive.reindex(detectors).to_string())
            table_speed = matched.pivot_table(index="detector", columns="speed_class", values="along_alone_km",
                                              aggfunc="median", sort=False).reindex(
                index=detectors, columns=[c for c in classes if c in set(matched["speed_class"])])
            print("\n[lead] along-track error of the exclusive matched flags (km, median) by mean speed of the cluster")
            print(table_speed.to_string())
            rows_out = {other: paired(matched, firsts[0], other, "along_alone_km", "larger")
                        for other in detectors if other != firsts[0]}
            print(f"\n[lead] cluster by cluster: the exclusive matched flags of {firsts[0]} further ahead along the motion "
                  f"than the exclusive ones of the other ('first ahead'); two-sided sign test")
            print(pd.DataFrame(rows_out).T.to_string(formatters={"sign test p": "{:.2g}".format}))
        if inputs is not None:
            # The along-track error of the matched flags far from normal and of the moderate ones, by speed of the cluster.
            print(f"\n[lead] along-track error (km, median over the clusters) of the matched flags with an input beyond "
                  f"{args.far_sd:g} standard deviations (far) and with none (moderate); ahead % over the clusters with "
                  f"such flags")
            states = matched.groupby("detector", sort=False).agg(
                far_km=("along_far_km", "median"),
                far_ahead_pct=("along_far_km", lambda s: 100 * (s.dropna() > 0).mean()),
                far_clusters=("along_far_km", lambda s: int(s.notna().sum())),
                moderate_km=("along_moderate_km", "median"),
                moderate_ahead_pct=("along_moderate_km", lambda s: 100 * (s.dropna() > 0).mean()),
                moderate_clusters=("along_moderate_km", lambda s: int(s.notna().sum())))
            print(states.reindex(detectors).to_string())
            for column, label in (("along_far_km", "far"), ("along_moderate_km", "moderate")):
                table_speed = matched.pivot_table(index="detector", columns="speed_class", values=column,
                                                  aggfunc="median", sort=False).reindex(
                    index=detectors, columns=[c for c in classes if c in set(matched["speed_class"])])
                print(f"\n[lead] along-track error of the {label} matched flags (km, median) by mean speed of the cluster")
                print(table_speed.to_string())
            for column, label in (("along_far_km", "far"), ("along_moderate_km", "moderate")):
                rows_out = {other: paired(matched, firsts[0], other, column, "larger")
                            for other in detectors if other != firsts[0]}
                print(f"\n[lead] cluster by cluster: the {label} matched flags of {firsts[0]} further ahead along the "
                      f"motion than those of the other ('first ahead'); two-sided sign test")
                print(pd.DataFrame(rows_out).T.to_string(formatters={"sign test p": "{:.2g}".format}))
        for first in firsts:
            for column, earlier_is, title in sides:
                rows_out = {other: paired(matched, first, other, column, earlier_is)
                            for other in detectors if other != first}
                frame = pd.DataFrame(rows_out).T
                print(f"\n[lead] cluster by cluster, on the clusters both match: {first} {title} the other ('first "
                      f"ahead'), at the same timestamp, or the other way round; two-sided sign test without the ties")
                print(frame.to_string(formatters={"sign test p": "{:.2g}".format}))
        if levels:
            # The curve over the probability: every level (the first row all the flags) and, below it, the same number of
            # flags taken at random; every column a detector of the cube, and the level against its random control;
            # ahead % / behind %, an asterisk when the sign test gives p < 0.01.
            others = [name for name in names if name != args.probability]
            cell = lambda r: (f"{r['first ahead %']:.0f}/{r['first behind %']:.0f}"
                              f"{'*' if r['sign test p'] < 0.01 else ''}") if r.get("clusters") else "-"
            curve_rows = [(args.probability, None, 100.0)]
            for level_name, control_name in zip(levels, controls):
                share = shares[float(level_name.rsplit("_p", 1)[1])]
                curve_rows += [(level_name, control_name, share), (control_name, None, share)]
            for column, earlier_is, title in sides:
                curve = {}
                for first, control, share in curve_rows:
                    line = {"flags kept %": share}
                    for other in others:
                        line[other] = cell(paired(matched, first, other, column, earlier_is))
                    line["vs same number at random"] = cell(paired(matched, first, control, column, earlier_is)) \
                        if control else ""
                    curve[first] = line
                print(f"\n[lead] curve over the probability of the draws: {args.probability} with p at least the level, "
                      f"and as many of its flags at random (_rand), {title} each detector, % of the clusters both match "
                      f"ahead / behind (* sign test p < 0.01); last column: the level against its random control")
                print(pd.DataFrame(curve).T.to_string())
            # The flags that come before the cluster: are they as stable as the flags of the match as a whole?
            own_rows = matched[(matched["detector"] == args.probability)]
            before = own_rows[own_rows["before_values"] > 0]
            print(f"\n[lead] {args.probability} starts before the concordant cluster in {len(before)} of {len(own_rows)} "
                  f"matched clusters. Flags with p >= 0.9: {100 * before['before_p90'].sum() / max(before['before_values'].sum(), 1):.1f}% "
                  f"of those before the start, {100 * own_rows['match_p90'].sum() / max(own_rows['match_values'].sum(), 1):.1f}% "
                  f"of all the flags of the matches; median over the clusters of the median p: "
                  f"{before['before_p_median'].median():.3f} before the start, {own_rows['match_p_median'].median():.3f} all")
    if args.csv:
        results.to_csv(args.csv, index=False)
        print(f"\nwritten {args.csv}")


if __name__ == "__main__":
    main()
