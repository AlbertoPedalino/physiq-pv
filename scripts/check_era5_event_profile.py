"""Does a score tell the centre, the edge and the outside of its own events? Its profile across them.

The flags of one score (`--events`) at a timestamp are the core of the events of `check_era5_event_zones.py`. Every
cell has a position, in cells of the grid (touching cells at distance 1, diagonals included) from the border of the
core: negative inside (-1 is a flagged cell next to an unflagged one, -2 one cell further in), positive outside (1 is
the edge of the zones, 4 or more the far zone). Positions beyond `--limit` count at the limit. A cell outside belongs
to the event of its nearest flagged cell, a type being as in the zones: the input variable farthest from its seasonal
normal over the core of the event.

For every type and position, and for all the types together, the check writes the number of values, for every score
of `--scores` the quartiles of its percentile among all the values of that score, and the median distance from normal
of the variable of the type (the absolute standardised value). That a flagged cell is above the threshold and an
unflagged one below it is by construction: what the profile adds is its shape, whether the score rises inside the
core and how fast it falls outside.

    python scripts/check_era5_event_profile.py --events G+clim=outputs/good2v/test_generator_clim_day \
        --scores G+clim=outputs/good2v/test_generator_clim_day normale=outputs/confronti_good2/climatology/test/normal \
        --inputs-run outputs/stgan_good_sweep_2_scored --csv outputs/casi_good2_g/profilo_5pc.csv

With `--cube` (a directory of build_era5_concordant_cube.py) the events are the concordant clusters instead: the values
with at least the votes needed of its summary, joined by the same rule. The scores read across them are then by default
the detectors of the cube, each over its own scores (not its flags), and the rows are grouped by the kind of the
concordant cluster instead of its type: "short" (at most the cube's max_hours from first to last timestamp), "long", and
"small" (fewer than the cube's min_values values, which the thesis does not study). A pure description of how every score
behaves from the core of the concordant clusters outwards; inside, at least the votes needed are above their threshold by
construction.

    python scripts/check_era5_event_profile.py --cube outputs/concordate_5pc_regola_tesi \
        --inputs-run outputs/stgan_optuna27_scored --limit 6 --csv outputs/concordate_5pc_regola_tesi/profilo_r6.csv
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
from scripts.check_era5_event_agreement import events_of
from scripts.check_era5_event_zones import (PERCENTILES, QUANTILE_STRIDE, event_types, flagged_values, inputs_of,
                                            open_score, parse_score, release)

CHUNK = 32  # timestamps read at a time
BINS = 101  # percentiles 0..100 of a score
DISTANCE_WIDTH, DISTANCE_BINS = 0.05, 201  # the distance from normal is counted in these bins; the last holds the rest


def positions_at(labels, limit):
    """From the event (1..) of every flagged cell of a timestamp on the grid and 0 elsewhere: the position of every cell
    (see the module docstring) and the event it belongs to. None if nothing is flagged."""
    core = labels > 0
    if not core.any():
        return None
    inside = ndimage.distance_transform_cdt(core, metric="chessboard")
    outside, nearest = ndimage.distance_transform_cdt(~core, metric="chessboard", return_indices=True)
    return (np.where(core, -np.minimum(inside, limit), np.minimum(outside, limit)),
            labels[nearest[0], nearest[1]])


def cluster_kinds(steps, events, count, timestamps, selection):
    """The kind of every event (0-based) as the concordant clusters of the thesis: 0 short (at most max_hours from its
    first to its last timestamp), 1 long, 2 small (fewer than min_values values, whatever its duration)."""
    sizes = np.bincount(events, minlength=count)
    first = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
    last = np.full(count, -1, dtype=np.int64)
    np.minimum.at(first, events, steps)
    np.maximum.at(last, events, steps)
    step_hours = float(np.median(np.diff(timestamps.asi8))) / 3.6e12 if len(timestamps) > 1 else 3.0
    hours = (last - first) * step_hours
    kinds = np.where(sizes < selection["min_values"], 2, np.where(hours <= selection["max_hours"], 0, 1))
    return kinds.astype(np.int64), ["short", "long", "small"]


def quantile_of(histogram, share):
    """The bin at `share` of the counts of a histogram."""
    total = histogram.sum()
    return np.nan if total == 0 else float(np.searchsorted(np.cumsum(histogram), share * total))


def read_table(percentiles, distances, names, score_names, limit):
    """The rows of the table from the histograms [score, type, position, BINS] and [type, position, DISTANCE_BINS]."""
    middle = (np.arange(DISTANCE_BINS) + 0.5) * DISTANCE_WIDTH
    rows = []
    groups = [(names[kind] if kind < len(names) else f"variable {kind}", [kind]) for kind in range(percentiles.shape[1])]
    groups.append(("all", list(range(percentiles.shape[1]))))
    for label, kinds in groups:
        for position in [*range(-limit, 0), *range(1, limit + 1)]:
            at = position + limit
            values = int(percentiles[0, kinds, at].sum())
            if not values:
                continue
            row = {"type": label, "position": position, "values": values}
            for index, name in enumerate(score_names):
                histogram = percentiles[index, kinds, at].sum(axis=0)
                row[f"{name} q25"] = quantile_of(histogram, 0.25)
                row[f"{name} percentile"] = quantile_of(histogram, 0.5)
                row[f"{name} q75"] = quantile_of(histogram, 0.75)
            row["variable distance"] = float(middle[int(quantile_of(distances[kinds, at].sum(axis=0), 0.5))])
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events", help="NAME=DIR[@component]: the score whose flags make the events")
    parser.add_argument("--cube", type=Path, help="Concordant cube: its clusters make the events (instead of --events)")
    parser.add_argument("--scores", nargs="+",
                        help="NAME=DIR[@component]: the scores read across them (with --cube: its detectors by default)")
    parser.add_argument("--inputs-run", type=Path, required=True,
                        help="Scored run of the detector: its shared cache gives the standardised inputs")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--limit", type=int, default=8, help="Positions beyond this many cells count here")
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.events is None) == (args.cube is None):
        raise ValueError("Give either --events or --cube")
    summary = None
    if args.cube is not None:
        summary = json.loads((args.cube / "summary.json").read_text(encoding="utf-8"))
        if summary["vote"].get("closure", {}).get("where") == "after":
            raise ValueError("A cube closed after the vote: its concordant values are not the votes above the needed")
        source = parse_score(f"concordant={args.cube / 'votes'}")
        scores = args.scores or [f"{name}={where}" for name, where in summary["vote"]["detectors"].items()]
    else:
        if not args.scores:
            raise ValueError("Give --scores")
        source, scores = parse_score(args.events), args.scores
    sources = [parse_score(text) for text in scores]
    if len({name for name, _, _ in sources}) != len(sources):
        raise ValueError("Give every score a different name")
    manifest, event_scores, maps = open_score(source)
    grid = manifest["grid"]
    timestamps = pd.DatetimeIndex(np.load(source[1] / "test_timestamps.npy"))
    inputs = None
    try:
        opened = {}
        for name, directory, component in sources:
            other, array, more = open_score((name, directory, component))
            maps.extend(more)
            if other["grid"] != grid or not timestamps.equals(pd.DatetimeIndex(np.load(directory / "test_timestamps.npy"))):
                raise ValueError(f"{name} is not over the grid and timestamps of {source[0]}")
            opened[name] = array
        rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
        shape = (len(grid["latitudes"]), len(grid["longitudes"]))
        if summary is None:
            limit_value = score_limit(event_scores, args.top_percent)
        else:
            limit_value = summary["vote"]["votes_needed"] - 0.5  # the vote counts are whole numbers
        flagged = flagged_values(event_scores, limit_value)
        steps, cells, events, count = events_of(flagged, rows, cols, shape)
        del flagged
        order = np.argsort(steps, kind="stable")
        steps, cells, events = steps[order], cells[order], events[order]
        inputs, first, names = inputs_of(args.inputs_run, timestamps)
        if inputs.shape[1] != len(rows):
            raise ValueError("The inputs run is not over the cells of the scores")
        kinds = event_types(steps, cells, events - 1, count, inputs, first)
        if summary is None:
            groups, group_names = kinds, names
        else:
            groups, group_names = cluster_kinds(steps, events - 1, count, timestamps, summary["selection"])
        quantiles = {}
        for name, array in opened.items():
            quantiles[name] = np.nanpercentile(np.asarray(array[::QUANTILE_STRIDE], dtype=np.float64), PERCENTILES)
            release(array)
        variables = inputs.shape[2] if summary is None else len(group_names)
        positions = 2 * args.limit + 1
        percentiles = np.zeros((len(opened), variables, positions, BINS), dtype=np.int64)
        distances = np.zeros((variables, positions, DISTANCE_BINS), dtype=np.int64)
        for start in range(0, len(timestamps), CHUNK):
            stop = min(start + CHUNK, len(timestamps))
            blocks = [np.asarray(array[start:stop], dtype=np.float64) for array in opened.values()]
            block_inputs = np.asarray(inputs[first + start:first + stop], dtype=np.float64)
            low, high = np.searchsorted(steps, [start, stop])
            for step in range(start, stop):
                at = slice(*np.searchsorted(steps[low:high], [step, step + 1]) + low)
                lattice = np.zeros(shape, dtype=np.int64)
                lattice[rows[cells[at]], cols[cells[at]]] = events[at]
                placed = positions_at(lattice, args.limit)
                if placed is None:
                    continue
                position = placed[0][rows, cols] + args.limit
                nearest = placed[1][rows, cols] - 1
                kind, group = kinds[nearest], groups[nearest]
                for index, (block, quantile) in enumerate(zip(blocks, quantiles.values())):
                    values = block[step - start]
                    finite = np.isfinite(values)
                    bins = np.clip(np.interp(values[finite], quantile, PERCENTILES).round().astype(np.int64), 0, BINS - 1)
                    place = (group[finite] * positions + position[finite]) * BINS + bins
                    percentiles[index] += np.bincount(place, minlength=variables * positions * BINS).reshape(variables, positions, BINS)
                chosen = block_inputs[step - start, np.arange(len(rows)), kind]
                finite = np.isfinite(chosen)
                bins = np.minimum((np.abs(chosen[finite]) / DISTANCE_WIDTH).astype(np.int64), DISTANCE_BINS - 1)
                place = (group[finite] * positions + position[finite]) * DISTANCE_BINS + bins
                distances += np.bincount(place, minlength=variables * positions * DISTANCE_BINS).reshape(variables, positions, DISTANCE_BINS)
            for array in opened.values():
                release(array)
            release_pages(inputs)
    finally:
        for array in maps:
            array._mmap.close()
        if inputs is not None:
            inputs._mmap.close()
    table = read_table(percentiles, distances, group_names, list(opened), args.limit)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.csv, index=False)
    chosen_by = (f"its highest {args.top_percent:g}%" if summary is None
                 else f"at least {summary['vote']['votes_needed']} votes, grouped short / long / small")
    if summary is not None:  # the same clusters as the cube: these counts must match its summary
        found = np.bincount(groups, minlength=3)
        expected = summary.get("clusters", {})
        print(f"[profile] concordant clusters short / long / small: {found[0]} / {found[1]} / {found[2]} "
              f"(cube summary: {expected.get('analysed', {}).get('clusters')} / {expected.get('long', {}).get('clusters')}"
              f" / {expected.get('small', {}).get('clusters')})")
    print(f"[profile] {timestamps[0].date()} to {timestamps[-1].date()} | events of {source[0]}, {chosen_by}: {count} "
          f"| position: cells from the border of the core, negative inside")
    shown = [column for column in table.columns if column != "type" and not column.endswith(("q25", "q75"))]
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(table[table["type"] == "all"][shown].to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
