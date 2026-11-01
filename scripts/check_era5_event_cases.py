"""The events of one score by type and size, and the specific cases to look at.

The events are those of `check_era5_event_zones.py`: the values of one score (`--events`) in its highest
`--top-percent`, grouped in the hypercube of time, latitude and longitude, with a size (their number of values) and
a type (the input variable farthest from its seasonal normal over the core). The check writes the events and their
values by type and size (`event_groups.csv`).

The cases are chosen by this rule, fixed before any of them is seen, so that no case is picked for how it looks:
the types are the `--types` with the most values in the events of at least `--big` values, and for every type the
case is its largest event, read at the timestamp with the most values of the event (the earliest when two have as
many). Touching flags chain up over weeks, so the largest event of a type is not one weather system: `event_times`
and `event_cells` give its cells at each of its timestamps, to show it.

Every case is written as `case_NN.npz` and listed in `cases.csv`:
- at that timestamp, for all the cells: every score of `--scores`, the input variable of the type, the zones of
  `check_era5_event_zones.py` (`core`, `edge` of this event; `far`, more than `--far` cells from every core);
- for one cell, the core cell with the highest score of the events: the same quantities `--around` timestamps before
  and after (`series_*`), to see the event start and end;
- for every score, its threshold and the quantiles that turn a value into its percentile over the scored period.

    python scripts/check_era5_event_cases.py --events G+clim=outputs/good2v/test_generator_clim_day \
        --scores G+clim=outputs/good2v/test_generator_clim_day normale=outputs/confronti_good2/climatology/test/normal \
        --inputs-run outputs/stgan_good_sweep_2_scored --output-dir outputs/casi_good2
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
from scripts.check_era5_dust_lead import score_limit
from scripts.check_era5_event_agreement import events_of
from scripts.check_era5_event_zones import (PERCENTILES, QUANTILE_STRIDE, SIZES, event_types, flagged_values, inputs_of,
                                            open_score, parse_score, release, size_bands, zones_at)
from scripts.check_era5_unreached import band_label


def choose_cases(sizes, kinds, types, big):
    """(type, event) of every case, the events counted from 0: see the module docstring."""
    large = np.nonzero(sizes >= big)[0]
    totals = np.bincount(kinds[large], weights=sizes[large], minlength=int(kinds.max(initial=0)) + 1)
    wanted = [int(kind) for kind in np.argsort(-totals, kind="stable")[:types] if totals[kind] > 0]
    cases = []
    for kind in wanted:
        members = large[kinds[large] == kind]
        cases.append((kind, int(members[np.argmax(sizes[members])])))
    return cases


def peak_of(steps):
    """The timestamp with the most of these values, the earliest when two have as many."""
    times, counts = np.unique(steps, return_counts=True)
    return int(times[np.argmax(counts)])


def group_table(sizes, kinds, names):
    """Events and values by type and size, then by type, by size and overall."""
    bands = size_bands(sizes)
    label = lambda kind: names[kind] if kind < len(names) else f"variable {kind}"
    selections = [(label(kind), band_label(*SIZES[band]), (kinds == kind) & (bands == band))
                  for kind in np.unique(kinds) for band in range(len(SIZES))]
    selections += [(label(kind), "all", kinds == kind) for kind in np.unique(kinds)]
    selections += [("all", band_label(*SIZES[band]), bands == band) for band in range(len(SIZES))]
    selections.append(("all", "all", np.ones(len(sizes), dtype=bool)))
    return pd.DataFrame([{"type": kind, "size": size, "events": int(chosen.sum()), "values": int(sizes[chosen].sum())}
                         for kind, size, chosen in selections if chosen.any()])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events", required=True, help="NAME=DIR[@component]: the score whose flags make the events")
    parser.add_argument("--scores", nargs="+", required=True, help="NAME=DIR[@component]: the scores written in the cases")
    parser.add_argument("--inputs-run", type=Path, required=True,
                        help="Scored run of the detector: its shared cache gives the standardised inputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--types", type=int, default=3, help="Types with the most values in the large events")
    parser.add_argument("--big", type=int, default=20, help="Values of an event to count as large")
    parser.add_argument("--far", type=int, default=3, help="Far: more than this many cells from every core cell")
    parser.add_argument("--around", type=int, default=24, help="Timestamps before and after the case in the series")
    args = parser.parse_args(argv)
    source = parse_score(args.events)
    sources = [parse_score(text) for text in args.scores]
    if len({name for name, _, _ in sources}) != len(sources):
        raise ValueError("Give every score a different name")
    manifest, event_scores, maps = open_score(source)
    grid = manifest["grid"]
    timestamps = pd.DatetimeIndex(np.load(source[1] / "test_timestamps.npy"))
    inputs = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
        latitudes, longitudes = np.asarray(grid["latitudes"])[rows], np.asarray(grid["longitudes"])[cols]
        flagged = flagged_values(event_scores, score_limit(event_scores, args.top_percent))
        steps, cells, events, count = events_of(flagged, rows, cols, shape)
        del flagged
        order = np.argsort(steps, kind="stable")
        steps, cells, events = steps[order], cells[order], events[order] - 1
        sizes = np.bincount(events, minlength=count)
        inputs, first, names = inputs_of(args.inputs_run, timestamps)
        if inputs.shape[1] != len(rows):
            raise ValueError("The inputs run is not over the cells of the scores")
        kinds = event_types(steps, cells, events, count, inputs, first)
        groups = group_table(sizes, kinds, names)
        groups.to_csv(args.output_dir / "event_groups.csv", index=False)
        print(f"[cases] {timestamps[0].date()} to {timestamps[-1].date()} | events of {source[0]}, its highest "
              f"{args.top_percent:g}%: {count}")
        with pd.option_context("display.width", 200, "display.max_rows", None):
            print(groups.to_string(index=False))

        scores = {}
        for name, array in opened.items():
            sample = np.asarray(array[::QUANTILE_STRIDE], dtype=np.float64)
            scores[name] = (score_limit(array, args.top_percent), np.nanpercentile(sample, PERCENTILES))
            del sample
            release(array)
        cases = []
        for number, (kind, event) in enumerate(choose_cases(sizes, kinds, args.types, args.big)):
            members = events == event
            peak = peak_of(steps[members])
            at = steps == peak
            labels = np.zeros(len(rows), dtype=np.int64)
            labels[cells[at]] = events[at] + 1
            edge, away = zones_at(labels, rows, cols, shape, args.far)
            core = labels == event + 1
            scores_now = np.asarray(event_scores[peak], dtype=np.float64)
            centre = int(np.nanargmax(np.where(core, scores_now, np.nan)))
            window = np.arange(max(peak - args.around, 0), min(peak + args.around + 1, len(timestamps)))
            event_steps, event_cells = np.unique(steps[members], return_counts=True)
            mine = np.nonzero(core)[0]
            record = {
                "case": number, "type": names[kind] if kind < len(names) else f"variable {kind}",
                "values": int(sizes[event]), "start": str(timestamps[event_steps[0]]), "end": str(timestamps[event_steps[-1]]),
                "timestamps": len(event_steps), "timestamp": str(timestamps[peak]), "values at the timestamp": len(mine),
                "latitude": float(latitudes[mine].mean()), "longitude": float(longitudes[mine].mean()),
                "cell latitude": float(latitudes[centre]), "cell longitude": float(longitudes[centre])}
            cases.append(record)
            arrays = {"rows": rows, "cols": cols, "latitudes": np.asarray(grid["latitudes"]),
                      "longitudes": np.asarray(grid["longitudes"]), "core": core, "edge": edge == event + 1, "far": away,
                      "input": np.asarray(inputs[first + peak, :, kind]),
                      "event_times": timestamps[event_steps].as_unit("ns").asi8, "event_cells": event_cells,
                      "series_times": timestamps[window].as_unit("ns").asi8, "series_cell": centre,
                      "series_input": np.asarray(inputs[first + window[0]:first + window[-1] + 1, centre, kind]),
                      "series_core": np.isin(window, steps[members & (cells == centre)])}
            for name, array in opened.items():
                arrays[f"score_{name}"] = np.asarray(array[peak], dtype=np.float32)
                arrays[f"series_{name}"] = np.asarray(array[window[0]:window[-1] + 1, centre], dtype=np.float32)
                arrays[f"limit_{name}"], arrays[f"quantiles_{name}"] = scores[name]
                release(array)
            np.savez_compressed(args.output_dir / f"case_{number:02d}.npz", case=json.dumps(record), **arrays)
            release_pages(inputs)
        table = pd.DataFrame(cases)
        table.to_csv(args.output_dir / "cases.csv", index=False)
        with pd.option_context("display.width", 300, "display.max_columns", None):
            print(table.to_string(index=False, float_format=lambda value: f"{value:.1f}"))
    finally:
        for array in maps:
            array._mmap.close()
        if inputs is not None:
            inputs._mmap.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
