"""How do the scores behave in the core of an anomaly, at its edge and away from it, by type and size of event?

No reference outside the data is used. The anomalies are the events of one score (`--events NAME=DIR`, or
`NAME=DIR@component`): the values in its highest `--top-percent` (threshold of `score_limit`), grouped in the
hypercube of time, latitude and longitude as in `check_era5_event_agreement.py` (touching cells at a
timestamp, diagonals included, and the same cell at the timestamps before and after). At every timestamp the
cells are in three zones:
- core: the cells of an event;
- edge: the cells next to a core cell (the eight around it, as the 3 x 3 patch the discriminator reads) that
  are not in a core; an edge cell belongs to the largest event it touches;
- far: the cells more than `--far` cells away from every core cell of that timestamp.
The cells in between are left out. Every event has a size, the number of its values (cell and timestamp), in
the bands 1, 2-4, 5-19, 20 or more, and a type: the input variable of the detector farthest from its seasonal
normal, as the mean of the absolute standardised value over the core of the event (read from the shared
cache of `--inputs-run`, the scored run of the detector).

For every score of `--scores` (the score of the events can be one of them; `@real` and `@generated` read the
two outputs of the discriminator, as in `check_era5_score_correlation.py`), the check gives, by type and
size and for the core and the edge, and for the far zone as a whole: the number of values, the median of the
percentile of the score (its value among all the values of that score) and the share of the values the score
flags at its own highest `--top-percent`. The same is given by size alone.

    python scripts/check_era5_event_zones.py --events G+clim=outputs/gen_clim \
        --scores G+clim=outputs/gen_clim D+clim=outputs/disc_clim simple=outputs/simple_max \
        --inputs-run outputs/stgan_good_sweep_2_scored --csv zones.csv
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
from scripts.check_era5_dust_lead import score_limit
from scripts.check_era5_edge_event_types import edge_events
from scripts.check_era5_event_agreement import events_of
from scripts.check_era5_score_correlation import open_component, parse_score
from scripts.check_era5_unreached import CHUNK, SIZES, band_label

ZONES = ("core", "edge")
BINS = 101  # percentiles 0..100 of a score, counted in these bins
QUANTILE_STRIDE = 10  # the quantiles of a score are read on one timestamp every this many
PERCENTILES = np.linspace(0, 100, 1001)


def open_score(source):
    """The metadata of a run, its score (the component of `source`, D(real) and D(generated) included) and the
    memory maps it reads."""
    name, directory, component = source
    manifest = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    return (manifest, *open_component(directory, manifest, component))


def release(array):
    """release_pages of a score, or of the maps it is the difference of."""
    release_pages(*getattr(array, "parts", (array,)))


def flagged_values(scores, limit):
    """[timestamps, cells] above `limit` (NaN is not flagged)."""
    flagged = np.zeros(scores.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        for start in range(0, len(scores), CHUNK):
            flagged[start:start + CHUNK] = np.asarray(scores[start:start + CHUNK]) > limit
            release_pages(scores)
    return flagged


def inputs_of(run_dir, timestamps):
    """The standardised inputs of the scored period [cache rows, cells, variables], the cache row of the first scored
    timestamp and the names of the variables."""
    manifest = json.loads((Path(run_dir) / "metadata.json").read_text(encoding="utf-8"))
    seasonal = manifest.get("backend", {}).get("seasonal_normalization")
    if not seasonal:
        raise ValueError("The inputs run has no seasonal normalization: there are no standardised inputs to read")
    cache = Path(seasonal["mean_file"]).parent / f"{manifest.get('scored_period', 'test')}.npy"
    standardised = np.load(cache, mmap_mode="r")
    first = len(standardised) - len(timestamps)  # the cache holds a few timestamps of context before the period
    if first < 0:
        standardised._mmap.close()
        raise ValueError("The cache of the inputs run does not cover the timestamps of the scores")
    return standardised, first, list(manifest.get("preparation", {}).get("features") or [])


def size_bands(sizes):
    """Index in SIZES of every event size."""
    bands = np.full(len(sizes), -1)
    for index, (low, high) in enumerate(SIZES):
        bands[(sizes >= low) & ((sizes <= high) if high else True)] = index
    return bands


def event_types(steps, cells, events, count, inputs, first):
    """Index of the variable farthest from its normal for every event: the largest mean |standardised value| over
    its core values."""
    totals = np.zeros((count, inputs.shape[2]))
    values = np.zeros(count)
    order = np.argsort(steps, kind="stable")
    steps, cells, events = steps[order], cells[order], events[order]
    bounds = np.searchsorted(steps, np.arange(0, steps.max() + CHUNK + 1, CHUNK)) if len(steps) else [0]
    for chunk, (low, high) in enumerate(zip(bounds[:-1], bounds[1:])):
        if low == high:
            continue
        start = chunk * CHUNK
        block = np.abs(np.asarray(inputs[first + start:first + start + CHUNK], dtype=np.float64))
        chosen = block[steps[low:high] - start, cells[low:high]]
        finite = np.isfinite(chosen).all(axis=1)
        np.add.at(totals, events[low:high][finite], chosen[finite])
        np.add.at(values, events[low:high][finite], 1)
        release_pages(inputs)
    return (totals / np.maximum(values, 1)[:, None]).argmax(axis=1)


def zones_at(labels, rows, cols, shape, far):
    """For one timestamp, from the event label (1..) of every core cell and 0 elsewhere [cells]: the event of every
    edge cell (0 if not an edge cell) and whether every cell is far from all the cores."""
    # The events of this timestamp numbered 1..k, so that the sizes counted for the edge stay small.
    present, local = np.unique(labels, return_inverse=True)
    if present[0] != 0:
        present, local = np.concatenate([[0], present]), local + 1
    lattice = np.zeros(shape, dtype=np.int64)
    lattice[rows, cols] = local
    core = labels > 0
    touched = present[edge_events(lattice, rows, cols, shape)]
    edge = np.where(core, 0, touched)
    if core.any():
        near = ndimage.binary_dilation(lattice > 0, structure=np.ones((3, 3), dtype=bool), iterations=far)
        away = ~near[rows, cols]
    else:
        away = np.ones(len(labels), dtype=bool)
    return edge, away


def describe(flagged, layout, scores, types, far):
    """Counts [score, group, zone] and percentile histograms [score, group, zone, BINS] of every score, with the
    groups type x size band of the events; the far zone is the last group. `scores` is
    {name: (array, limit, quantiles)}, `types(steps, cells, events, count)` gives the type of every event."""
    rows, cols, shape = layout
    steps, cells, events, count = events_of(flagged, rows, cols, shape)
    sizes = np.bincount(events, minlength=count + 1)[1:]
    kinds = types(steps, cells, events - 1, count)
    bands = size_bands(sizes)
    variables = int(kinds.max()) + 1 if count else 1
    group_of = np.concatenate([[-1], kinds * len(SIZES) + bands])  # by label (0 = none)
    groups = variables * len(SIZES)
    names = list(scores)
    counts = np.zeros((len(names), groups + 1, len(ZONES)), dtype=np.int64)
    flagged_counts = np.zeros_like(counts)
    histograms = np.zeros((len(names), groups + 1, len(ZONES), BINS), dtype=np.int64)
    order = np.argsort(steps, kind="stable")
    steps, cells, events = steps[order], cells[order], events[order]
    for start in range(0, len(flagged), CHUNK):
        stop = min(start + CHUNK, len(flagged))
        blocks = {name: np.asarray(array[start:stop], dtype=np.float64) for name, (array, _, _) in scores.items()}
        low, high = np.searchsorted(steps, [start, stop])
        for step in range(start, stop):
            labels = np.zeros(len(rows), dtype=np.int64)
            at = slice(*np.searchsorted(steps[low:high], [step, step + 1]) + low)
            labels[cells[at]] = events[at]
            edge, away = zones_at(labels, rows, cols, shape, far)
            selections = [(group_of[labels[labels > 0]], np.nonzero(labels > 0)[0], 0),
                          (group_of[edge[edge > 0]], np.nonzero(edge > 0)[0], 1),
                          (np.full(int(away.sum()), groups), np.nonzero(away)[0], 0)]
            for index, name in enumerate(names):
                _, limit, quantiles = scores[name]
                values = blocks[name][step - start]
                for group, where, zone in selections:
                    chosen = values[where]
                    finite = np.isfinite(chosen)
                    if not finite.any():
                        continue
                    bins = np.clip(np.interp(chosen[finite], quantiles, PERCENTILES).round().astype(int), 0, BINS - 1)
                    np.add.at(counts[index], (group[finite], zone), 1)
                    np.add.at(flagged_counts[index], (group[finite], zone), chosen[finite] > limit)
                    np.add.at(histograms[index], (group[finite], zone, bins), 1)
        for array, _, _ in scores.values():
            release(array)
    return dict(counts=counts, flagged=flagged_counts, histograms=histograms, variables=variables,
                events=np.bincount(group_of[1:], minlength=groups), names=names)


def median_of(histogram):
    total = histogram.sum()
    return np.nan if total == 0 else float(np.searchsorted(np.cumsum(histogram), total / 2))


def table(described, variable_names):
    """Rows: by type and size, by size, and the far zone; for every score its counts, median percentile and
    share flagged in the core and at the edge."""
    counts, flagged, histograms = described["counts"], described["flagged"], described["histograms"]
    variables, groups = described["variables"], counts.shape[1] - 1
    selections = [(variable_names[v] if v < len(variable_names) else f"variable {v}", band_label(*SIZES[b]),
                   [v * len(SIZES) + b]) for v in range(variables) for b in range(len(SIZES))]
    selections += [("all", band_label(*SIZES[b]), [v * len(SIZES) + b for v in range(variables)]) for b in range(len(SIZES))]
    selections.append(("all", "all", list(range(groups))))
    rows = []
    for kind, size, chosen in selections:
        events = int(described["events"][chosen].sum())
        if not events:
            continue
        row = {"type": kind, "size": size, "events": events}
        for index, name in enumerate(described["names"]):
            for zone, label in enumerate(ZONES):
                count = counts[index, chosen, zone].sum()
                row[f"{name} {label} values"] = int(count)
                row[f"{name} {label} percentile"] = median_of(histograms[index, chosen, zone].sum(axis=0))
                row[f"{name} {label} flagged %"] = 100 * flagged[index, chosen, zone].sum() / max(count, 1)
        rows.append(row)
    far = {"type": "far from every event", "size": "", "events": 0}
    for index, name in enumerate(described["names"]):
        count = counts[index, groups, 0]
        far[f"{name} core values"] = int(count)  # the far values, in the columns of the core
        far[f"{name} core percentile"] = median_of(histograms[index, groups, 0])
        far[f"{name} core flagged %"] = 100 * flagged[index, groups, 0] / max(count, 1)
    rows.append(far)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events", required=True, help="NAME=DIR[@component]: the score whose flags make the events")
    parser.add_argument("--scores", nargs="+", required=True, help="NAME=DIR[@component]: the scores read in the zones")
    parser.add_argument("--inputs-run", type=Path, required=True,
                        help="Scored run of the detector: its shared cache gives the standardised inputs")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--far", type=int, default=3, help="Far: more than this many cells from every core cell")
    parser.add_argument("--csv", type=Path, help="Also write the table here")
    args = parser.parse_args(argv)
    if args.far < 1:
        raise ValueError("far must be at least 1 cell")
    event_source = parse_score(args.events)
    sources = [parse_score(text) for text in args.scores]
    if len({name for name, _, _ in sources}) != len(sources):
        raise ValueError("Give every score a different name")
    manifest, event_scores, maps = open_score(event_source)
    grid = manifest["grid"]
    timestamps = pd.DatetimeIndex(np.load(event_source[1] / "test_timestamps.npy"))
    opened = {}
    inputs = None
    try:
        for source in sources:
            other_manifest, array, more = open_score(source)
            opened[source[0]] = array
            maps.extend(more)
            if other_manifest["grid"] != grid or not timestamps.equals(pd.DatetimeIndex(np.load(source[1] / "test_timestamps.npy"))):
                raise ValueError(f"{source[0]} is not over the grid and timestamps of {event_source[0]}")
        rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
        layout = (rows, cols, (len(grid["latitudes"]), len(grid["longitudes"])))
        flagged = flagged_values(event_scores, score_limit(event_scores, args.top_percent))
        scores = {}
        for name, array in opened.items():
            sample = np.asarray(array[::QUANTILE_STRIDE], dtype=np.float64)
            scores[name] = (array, score_limit(array, args.top_percent), np.nanpercentile(sample, PERCENTILES))
            release(array)
        inputs, first, variable_names = inputs_of(args.inputs_run, timestamps)
        if inputs.shape[1] != len(rows):
            raise ValueError("The inputs run is not over the cells of the scores")
        described = describe(flagged, layout, scores,
                             lambda steps, cells, events, count: event_types(steps, cells, events, count, inputs, first),
                             args.far)
    finally:
        for array in maps:
            array._mmap.close()
        if inputs is not None:
            inputs._mmap.close()
    summary = table(described, variable_names)
    print(f"[zones] {timestamps[0].date()} to {timestamps[-1].date()} | events of {event_source[0]}, its highest "
          f"{args.top_percent:g}%: {int(described['events'].sum())} | type: the input farthest from its normal over "
          f"the core | edge: the 8 cells around a core | far: more than {args.far} cells from every core | "
          f"percentile: median over the values; flagged: at each score's own highest {args.top_percent:g}%")
    with pd.option_context("display.width", 300, "display.max_columns", None, "display.max_rows", None):
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.1f}"))
    if args.csv is not None:
        summary.to_csv(args.csv, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
