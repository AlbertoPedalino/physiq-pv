"""The most striking days of a score: the days with the largest single patch of flagged cells, for a final check.

The rule was fixed on 5 November 2026, before the scores of the final run were read, so that no day is picked
for how it looks. A cell is flagged at a timestamp when the score (`--score`) is in its highest `--top-percent`
(threshold of `score_limit`, from the validation years when the run names a threshold reference). At every
timestamp the flagged cells are grouped in patches, cells touching on the grid, diagonals included, and only in
space: the timestamps are not linked, so the patches do not chain up over weeks as the events do. The size of a day
is the size of its largest patch over its timestamps (UTC days), read at the timestamp where it is largest (the
earliest when two are as large). The striking days are then chosen greedily, the largest first: a day is taken
when it is at least `--gap-days` days from every day taken before, so that one episode counts once, until there
are `--days` of them.

With `--rank-by intensity` (rule added on 6 November 2026, after the days by size were read, and fixed before the
days by intensity were) a day is ranked instead by its most intense patch: among the patches of at least
`--min-cells` cells, so that a single cell is not a band, the one with the highest median score, over the
timestamps of the day. The days are then chosen in the same way, the most intense first.

For every striking day the check writes the timestamp, the cells of the patch and their share of the domain, the
box and the centre of the patch, and, with `--inputs-run` (the scored run of the detector, whose shared cache holds
the standardised inputs), the three input variables farthest from their seasonal normal over the patch (mean
absolute standardised value): the first is the type of the patch, as for the events.

Outputs in `--output-dir`: `giorni.csv`, the size of every day (for the plot over time); `casi.csv`, the striking
days; `casi.npz`, for every striking day the score, the flagged cells and the patch on the lattice, with the
latitudes and longitudes, to draw a map.

    python scripts/list_era5_striking_days.py --score G+clim=outputs/optuna27/test_generator_clim_day \
        --inputs-run outputs/stgan_optuna27_scored --output-dir outputs/optuna27/giorni_eclatanti
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
from scripts.check_era5_event_zones import inputs_of, open_score
from scripts.check_era5_score_correlation import parse_score
from scripts.check_era5_unreached import CHUNK

TOUCHING = np.ones((3, 3), dtype=bool)  # cells touching on the grid, diagonals included


def largest_patch(flagged):
    """The cells (a boolean lattice) of the largest patch of a boolean lattice, and its size; the patch with the
    lowest label when two are as large."""
    labels, count = ndimage.label(flagged, structure=TOUCHING)
    if count == 0:
        return np.zeros_like(flagged), 0
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(sizes.argmax())
    return labels == largest + 1, int(sizes[largest])


def strongest_patch(flagged, values, min_cells):
    """The cells of the patch of at least `min_cells` cells with the highest median of `values` (a lattice of the
    same shape), and that median; the patch with the lowest label when two are as intense. No patch: 0."""
    labels, count = ndimage.label(flagged, structure=TOUCHING)
    if count == 0:
        return np.zeros_like(flagged), 0.0
    sizes = np.bincount(labels.ravel())[1:]
    kept = np.flatnonzero(sizes >= min_cells) + 1
    if len(kept) == 0:
        return np.zeros_like(flagged), 0.0
    medians = np.asarray(ndimage.median(values, labels, index=kept), dtype=np.float64)
    best = int(medians.argmax())
    return labels == kept[best], float(medians[best])


def choose_days(days, sizes, count, gap):
    """Indices of the striking days: the largest first, each at least `gap` days from those taken before.

    `days` are dates (datetime64[D]) and `sizes` the size of each; ties go to the earlier day."""
    days = np.asarray(days, dtype="datetime64[D]")
    order = np.lexsort((days, -np.asarray(sizes)))
    taken = []
    for index in order:
        if len(taken) == count:
            break
        if sizes[index] <= 0:
            break
        if all(abs(int((days[index] - days[other]).astype(int))) >= gap for other in taken):
            taken.append(int(index))
    return taken


def lattice_of(values, rows, cols, shape, fill):
    lattice = np.full(shape, fill, dtype=np.asarray(values).dtype)
    lattice[rows, cols] = values
    return lattice


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--score", required=True, help="NAME=DIR[@component]: the score whose flags are read")
    parser.add_argument("--inputs-run", type=Path, help="Scored run of the detector, for the variables of every patch")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--gap-days", type=int, default=7)
    parser.add_argument("--rank-by", choices=("size", "intensity"), default="size",
                        help="Rank a day by its largest patch, or by its most intense patch")
    parser.add_argument("--min-cells", type=int, default=10, help="With --rank-by intensity: the smallest patch read")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    intense = args.rank_by == "intensity"

    def patch_of(values):
        """The patch that ranks a timestamp, and its measure: the size, or the median score."""
        with np.errstate(invalid="ignore"):
            flagged = lattice_of(values > limit, rows, cols, shape, False)
        if intense:
            return (flagged, *strongest_patch(flagged, lattice_of(values, rows, cols, shape, np.nan), args.min_cells))
        return (flagged, *largest_patch(flagged))

    source = parse_score(args.score)
    manifest, scores, maps = open_score(source)
    grid = manifest["grid"]
    rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
    latitudes, longitudes = np.asarray(grid["latitudes"]), np.asarray(grid["longitudes"])
    shape = (len(latitudes), len(longitudes))
    timestamps = pd.DatetimeIndex(np.load(source[1] / "test_timestamps.npy"))
    inputs = None
    try:
        limit = score_limit(scores, args.top_percent)
        largest = np.zeros(len(timestamps), dtype=np.float64 if intense else np.int64)
        for start in range(0, len(timestamps), CHUNK):
            block = np.asarray(scores[start:start + CHUNK], dtype=np.float64)
            for offset, values in enumerate(block):
                largest[start + offset] = patch_of(values)[2]
            scores_release = getattr(scores, "parts", (scores,))
            release_pages(*scores_release)
        days = pd.DataFrame({"timestamp": timestamps, "day": timestamps.normalize(), "largest": largest})
        peak = days.loc[days.groupby("day")["largest"].idxmax()]  # idxmax keeps the earliest of equal sizes
        peak = peak.reset_index(drop=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        measure = "score mediano della macchia più intensa" if intense else "celle della macchia più grande"
        peak.rename(columns={"day": "giorno", "largest": measure,
                             "timestamp": "istante"})[["giorno", "istante", measure]].to_csv(
            args.output_dir / "giorni.csv", index=False)
        chosen = choose_days(peak["day"].values.astype("datetime64[D]"), peak["largest"].to_numpy(), args.days,
                             args.gap_days)
        if args.inputs_run is not None:
            inputs, first, names = inputs_of(args.inputs_run, timestamps)
            if inputs.shape[1] != len(rows):
                raise ValueError("The inputs run is not over the cells of the score")
        records, arrays = [], {"latitudes": latitudes, "longitudes": longitudes}
        for rank, index in enumerate(chosen, start=1):
            step = int(np.flatnonzero(timestamps == peak.loc[index, "timestamp"])[0])
            values = np.asarray(scores[step], dtype=np.float64)
            flagged, patch, _ = patch_of(values)
            size = int(patch.sum())
            inside = patch[rows, cols]
            record = {"rango": rank, "giorno": str(peak.loc[index, "day"].date()), "istante": str(timestamps[step]),
                      "celle": size, "quota del dominio %": 100 * size / len(rows),
                      "celle segnalate nell'istante": int(flagged.sum()),
                      "lat_min": float(latitudes[rows[inside]].min()), "lat_max": float(latitudes[rows[inside]].max()),
                      "lon_min": float(longitudes[cols[inside]].min()), "lon_max": float(longitudes[cols[inside]].max()),
                      "lat_centro": float(latitudes[rows[inside]].mean()),
                      "lon_centro": float(longitudes[cols[inside]].mean()),
                      "score mediano nella macchia": float(np.nanmedian(values[inside]))}
            if inputs is not None:
                distance = np.nanmean(np.abs(np.asarray(inputs[first + step], dtype=np.float64)[inside]), axis=0)
                for place, variable in enumerate(np.argsort(-distance)[:3], start=1):
                    record[f"variabile {place}"] = names[variable] if names else str(variable)
                    record[f"scarto {place}"] = float(distance[variable])
            records.append(record)
            arrays[f"score_{rank:02d}"] = lattice_of(values.astype(np.float32), rows, cols, shape, np.float32(np.nan))
            arrays[f"segnalate_{rank:02d}"] = flagged
            arrays[f"macchia_{rank:02d}"] = patch
        arrays["soglia"] = np.float64(limit)
        table = pd.DataFrame(records)
        table.to_csv(args.output_dir / "casi.csv", index=False, float_format="%.4g")
        np.savez_compressed(args.output_dir / "casi.npz", **arrays)
        (args.output_dir / "regola.json").write_text(json.dumps(
            {"score": args.score, "top_percent": args.top_percent, "days": args.days, "gap_days": args.gap_days,
             "rank_by": args.rank_by, "min_cells": args.min_cells if intense else None,
             "threshold": float(limit), "period": [str(timestamps[0]), str(timestamps[-1])]}, indent=2),
            encoding="utf-8")
    finally:
        for array in maps:
            array._mmap.close()
        if inputs is not None:
            inputs._mmap.close()
    print(f"[giorni] {timestamps[0].date()} to {timestamps[-1].date()} | {source[0]}, its highest {args.top_percent:g}% "
          f"| the {len(table)} days with the {'most intense patch of at least ' + str(args.min_cells) + ' cells' if intense else 'largest single patch'}, "
          f"at least {args.gap_days} days apart")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
