"""Does the weather come to the cells only the model flags at the edge of an anomaly, in the next hours?

Not whether the cell becomes extreme: whether, for instance, it rains there in the next hours, even a little. At every
read timestamp (one every `--stride`) the cells are put in groups:
- first only: the first score (`--score-dir`, the model) flags the cell, the second (`--by-dir`, one or several: a
  value is flagged when any of them flags it) does not, and the second flags a cell next to it (distance 1, the
  larger of the row and column offsets); with `--ratio-run` (the scored run with the dropout draws) also split in
  thirds of the relative spread of the draws, the spread over the mean of the error (cuts from the read values);
- neither: nobody flags the cell, and the second flags a cell next to it: the reference at the same distance (a
  share `--neither-share` of them is read);
- second only: the second flags the cell and the first does not (a share `--second-share` is read).
Each score flags its highest `--top-percent`, with the threshold of the validation years when the run names it.

For every cell, now and `--leads` hours later, from the raw values of the prepared cache (`--prepared-dir`, its test
partition): the precipitation of the hour ending at that timestamp (mm), and whether it rains (at least `--rain-mm`);
with `--inputs-run` (a scored run with the seasonal normalization: its standardised inputs) also whether precipitation,
vertical velocity, cloud cover and solar radiation are away from their seasonal normal (more than 1 standard deviation),
and on which side: above (more cloud, more radiation than normal) or below. For every group: the share raining now and
later, the share where it starts raining (not now, later), the mean precipitation, and the shares away from normal,
above and below, now and later. A second table gives cloud cover and radiation above and below normal alone.

    python scripts/check_era5_alone_future.py --score-dir outputs/optuna27/test_generator_clim_day \
        --by-dir outputs/confronti_good2/climatology/test/mcd --ratio-run outputs/stgan_optuna27_scored \
        --inputs-run outputs/stgan_optuna27_scored --prepared-dir data/era5_prepared --csv outputs/alone_future.csv
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
from physiq_pv.era5.data import load_prepared
from scripts.check_era5_edge_lead import band_of
from scripts.check_era5_lead import common_timestamps, flags_at, short_names, union_flags

CHUNK = 128  # read timestamps handled at a time
VARIABLES = ("total_precipitation", "vertical_velocity", "total_cloud_cover", "surface_solar_radiation_downwards")
SHORT = {"total_precipitation": "rain", "vertical_velocity": "w", "total_cloud_cover": "cloud",
         "surface_solar_radiation_downwards": "radiation"}
AWAY = 1.0  # standard deviations from the seasonal normal


def read_rows(array, rows):
    """array[rows] for increasing rows, read as one contiguous block."""
    block = np.asarray(array[int(rows[0]):int(rows[-1]) + 1], dtype=np.float32)
    return block[rows - rows[0]]


def collect(args):
    timestamps = common_timestamps([args.score_dir, *args.by_dir])
    first = flags_at(args.score_dir, args.top_percent, timestamps)
    second = union_flags(args.by_dir, args.top_percent, timestamps)
    grid = json.loads((args.score_dir / "metadata.json").read_text(encoding="utf-8"))["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    shape = (len(grid["latitudes"]), len(grid["longitudes"]))
    steps = [hours // 3 for hours in args.leads]
    if any(hours % 3 for hours in args.leads):
        raise ValueError("The leads must be multiples of 3 hours")
    longest = max(steps)

    cubes = load_prepared(args.prepared_dir)[0]
    raw_rows = pd.DatetimeIndex(cubes.test_timestamps).get_indexer(timestamps)
    if (raw_rows < 0).any():
        raise ValueError("The test partition of the prepared cache does not cover the timestamps of the scores")
    rain_column = list(cubes.feature_names).index("total_precipitation")
    ratio_maps = standardised = None
    if args.ratio_run is not None:
        from scripts.check_era5_dust_lead import score_file
        manifest = json.loads((args.ratio_run / "metadata.json").read_text(encoding="utf-8"))
        ratio_maps = (np.load(args.ratio_run / manifest["generator_std_file"], mmap_mode="r"),
                      np.load(args.ratio_run / score_file(manifest, "generator"), mmap_mode="r"))
        ratio_rows = pd.DatetimeIndex(np.load(args.ratio_run / "test_timestamps.npy")).get_indexer(timestamps)
        if (ratio_rows < 0).any():
            raise ValueError("The dropout run does not cover the timestamps of the scores")
    if args.inputs_run is not None:
        from scripts.check_era5_event_zones import inputs_of
        run_times = pd.DatetimeIndex(np.load(args.inputs_run / "test_timestamps.npy"))
        standardised, offset, names = inputs_of(args.inputs_run, run_times)
        input_rows = run_times.get_indexer(timestamps)
        if (input_rows < 0).any():
            raise ValueError("The inputs run does not cover the timestamps of the scores")
        input_rows = input_rows + offset
        input_columns = [names.index(name) for name in VARIABLES]

    rng = np.random.default_rng(args.seed)
    positions = np.arange(0, len(timestamps) - longest, args.stride)
    parts = {"first only": [], "neither": [], "second only": []}
    try:
        for start in range(0, len(positions), CHUNK):
            chosen = positions[start:start + CHUNK]
            span = np.arange(chosen[0], chosen[-1] + longest + 1)
            rain = read_rows(cubes.test, raw_rows[span])[:, :, rain_column] * 1000.0  # mm in the hour
            release_pages(cubes.test)
            signed = None
            if standardised is not None:
                signed = read_rows(standardised, input_rows[span])[:, :, input_columns]  # standardised values
                release_pages(standardised)
            spread = None
            if ratio_maps is not None:
                with np.errstate(invalid="ignore", divide="ignore"):
                    spread = (read_rows(ratio_maps[0], ratio_rows[chosen])
                              / read_rows(ratio_maps[1], ratio_rows[chosen]))
                release_pages(*ratio_maps)
            near = band_of(second[chosen], rows, cols, shape) == 0  # distance 1 from the area of the second
            for index, position in enumerate(chosen):
                local = position - span[0]
                flagged_first, flagged_second = first[position], second[position]
                groups = {
                    "first only": flagged_first & ~flagged_second & near[index],
                    "neither": ~flagged_first & ~flagged_second & near[index]
                    & (rng.random(len(rows)) < args.neither_share),
                    "second only": flagged_second & ~flagged_first & (rng.random(len(rows)) < args.second_share)}
                for name, cells in groups.items():
                    cells = np.flatnonzero(cells)
                    if not len(cells):
                        continue
                    record = {"rain now": rain[local, cells]}
                    for hours, step in zip(args.leads, steps):
                        record[f"rain +{hours}h"] = rain[local + step, cells]
                    if signed is not None:
                        for v, variable in enumerate(VARIABLES):
                            for when, at in (("now", local), *((f"+{hours}h", local + step)
                                                               for hours, step in zip(args.leads, steps))):
                                values = signed[at, cells, v]
                                record[f"{SHORT[variable]} above {when}"] = values > AWAY
                                record[f"{SHORT[variable]} below {when}"] = values < -AWAY
                    record["spread"] = (spread[index, cells] if spread is not None and name == "first only"
                                        else np.full(len(cells), np.nan, dtype=np.float32))
                    parts[name].append(pd.DataFrame(record))
    finally:
        cubes.close()
        for array in (*(ratio_maps or ()), standardised):
            if array is not None:
                array._mmap.close()
    tables = {name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame() for name, frames in parts.items()}
    return tables, short_names(args.score_dir, args.by_dir), timestamps


def summarise(table, leads, rain_mm, label):
    """One row of shares for a group of cells."""
    row = {"group": label, "cells": len(table)}
    if not len(table):
        return row
    raining_now = table["rain now"] >= rain_mm
    row["raining now %"] = 100 * raining_now.mean()
    row["mean rain now mm"] = table["rain now"].mean()
    for hours in leads:
        later = table[f"rain +{hours}h"] >= rain_mm
        row[f"raining +{hours}h %"] = 100 * later.mean()
        row[f"starts raining +{hours}h %"] = 100 * (later & ~raining_now).sum() / max(int((~raining_now).sum()), 1)
        row[f"mean rain +{hours}h mm"] = table[f"rain +{hours}h"].mean()
    for variable in VARIABLES:
        name = SHORT[variable]
        if f"{name} above now" in table:
            for when in ("now", *(f"+{hours}h" for hours in leads)):
                above, below = table[f"{name} above {when}"], table[f"{name} below {when}"]
                row[f"{name} away {when} %"] = 100 * (above | below).mean()
                row[f"{name} above {when} %"] = 100 * above.mean()
                row[f"{name} below {when} %"] = 100 * below.mean()
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score-dir", type=Path, required=True, help="Run with the first score (the model)")
    parser.add_argument("--by-dir", type=Path, nargs="+", required=True, help="Run(s) of the second score")
    parser.add_argument("--prepared-dir", type=Path, required=True, help="Prepared ERA5 cache: raw values")
    parser.add_argument("--ratio-run", type=Path, help="Scored run with the dropout draws: thirds of the spread")
    parser.add_argument("--inputs-run", type=Path, help="Scored run with the seasonal normalization: standardised inputs")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--leads", type=int, nargs="+", default=[3, 6])
    parser.add_argument("--rain-mm", type=float, default=0.1, help="It rains: at least this precipitation in the hour")
    parser.add_argument("--stride", type=int, default=3, help="Timestamps read: one every this many")
    parser.add_argument("--neither-share", type=float, default=0.05)
    parser.add_argument("--second-share", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    tables, short, timestamps = collect(args)
    rows = []
    alone = tables["first only"]
    if len(alone) and alone["spread"].notna().any():
        cuts = np.nanquantile(alone["spread"], [1 / 3, 2 / 3])
        thirds = np.digitize(alone["spread"], cuts)
        for third, name in enumerate(("low", "middle", "high")):
            rows.append(summarise(alone[(thirds == third) & alone["spread"].notna()], args.leads, args.rain_mm,
                                  f"{short[0]} only, spread {name}"))
    rows.append(summarise(alone, args.leads, args.rain_mm, f"{short[0]} only, all"))
    rows.append(summarise(tables["neither"], args.leads, args.rain_mm, "neither (reference)"))
    rows.append(summarise(tables["second only"], args.leads, args.rain_mm, f"{short[1]} only"))
    table = pd.DataFrame(rows)
    print(f"[alone-future] {timestamps[0].date()} to {timestamps[-1].date()}, one timestamp every {args.stride} | "
          f"flagged: the highest {args.top_percent:g}% | {short[0]} only and neither: next to the area of {short[1]} | "
          f"it rains: at least {args.rain_mm:g} mm in the hour | away: more than {AWAY:g} standard deviation")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(table.round(2).to_string(index=False))
    sides = [column for column in table.columns
             if column.startswith(("cloud above", "cloud below", "radiation above", "radiation below"))]
    if sides:
        print("\ncloud cover and solar radiation above and below their seasonal normal (% of the cells of the group)")
        print(table[["group", *sides]].round(1).to_string(index=False))
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv, index=False)
        print(f"written {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
