"""What does the spread of the Monte Carlo dropout draws say about the flags of a score, with no reference outside the data?

The flags are those of `--events` (a score with its spread over the draws, as a run written by the `climatology`
step keeps it). No label is used, and the spread is not read as the probability that a flag is right. The check gives:
- how solid the flags are: how many stay above the threshold one and two spreads lower, how many would pass it one
  spread higher, that is how much a flag depends on the masks of the dropout;
- classes of the flags at equal score: the flags are cut in `--bands` bands of the score, and in each band in
  `--classes` classes by spread (1: the draws agree most), so that the classes do not differ by how strong the flags
  are. This is done for the spread of the score itself and for every map of `--spreads`, for instance the spread of the
  error before the climatology, since the two may not say the same. Every class has the median of the score and of
  the spread, the median percentile of the distance from normal (`--normal`, a score over all the variables), the share
  of its values above the threshold of that distance, the median of the largest absolute standardised value over the
  variables of the cell (read from the inputs), and the share with no variable beyond 2 of it: flags where nothing
  observed in the cell is far from normal, so that the unusual part is in the answer of the model or in the cells around;
- the spread across the events: its percentile by position, in cells from the border of the core, as in
  `check_era5_event_profile.py`, inside and outside.

A spread grows with the error it comes from, so its size alone says little. With `--raw-error` (the error before the
climatology) and `--ratio-of` (the name in `--spreads` of the spread of that error), the ratio of the two is read like
the other spreads, as `ratio`: how large the disagreement of the draws is for the size of the error. The classes of the
raw spread and of the ratio are also given at equal raw error, the bands being cut on it instead of the score.

With `--course NAME=DIR`, a score of how fast the weather changes (the variation in three hours), every class also has
the share of its values that score flags, over all of them and over those under the threshold of the distance from
normal: flags with no extreme value but an unusual course, if the draws agree on these.

    python scripts/check_era5_dropout_flags.py --events G+clim=outputs/good2v/test_generator_clim_day \
        --spreads grezza=outputs/stgan_good_sweep_2_scored/scores/generator_std.npy \
        --normal normale=outputs/confronti_good2/climatology/test/normal \
        --raw-error G=outputs/stgan_good_sweep_2_scored@generator --ratio-of grezza \
        --inputs-run outputs/stgan_good_sweep_2_scored --csv outputs/casi_good2_g/dropout_5pc.csv
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
from scripts.check_era5_discriminator_confidence import split
from scripts.check_era5_dust_lead import score_limit
from scripts.check_era5_event_profile import positions_at, quantile_of
from scripts.check_era5_event_zones import PERCENTILES, QUANTILE_STRIDE, inputs_of, open_score, parse_score, release

CHUNK = 32  # timestamps read at a time
BINS = 101  # percentiles 0..100


def classes_table(name, bands_on, by, score, spread, normal, largest, normal_limit, normal_quantiles, bands, classes,
                  course=None, course_limit=None):
    """Rows of the classes of one spread over the flagged values, within bands of `by`: the arrays are [flagged values]."""
    finite = np.isfinite(spread) & np.isfinite(by)
    band = split(by[finite], bands)
    group = np.empty(int(finite.sum()), dtype=np.int64)
    for index in range(bands):
        inside = band == index
        group[inside] = split(spread[finite][inside], classes)
    rows = []
    for index in range(classes):
        chosen = np.flatnonzero(finite)[group == index]
        distance = normal[chosen]
        known = np.isfinite(distance)
        extra = {}
        if course is not None:
            changing = course[chosen] > course_limit
            below = known & (distance <= normal_limit)
            extra = {"course flagged %": 100 * float(changing.mean()),
                     "course flagged, below normal %": 100 * float(changing[below].mean()) if below.any() else np.nan,
                     "below normal %": 100 * float(below.mean())}
        rows.append({"spread": name, "bands": bands_on, "class": index + 1, "values": len(chosen), **extra,
                     "median score": float(np.median(score[chosen])), "median by": float(np.median(by[chosen])),
                     "median spread": float(np.median(spread[chosen])),
                     "normal percentile": float(np.median(np.interp(distance[known], normal_quantiles, PERCENTILES))),
                     "normal flagged %": 100 * float((distance[known] > normal_limit).mean()),
                     "largest value": float(np.nanmedian(largest[chosen])),
                     "none beyond 2 %": 100 * float((largest[chosen] <= 2).mean())})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events", required=True, help="NAME=DIR: the score whose flags are read, with its spread")
    parser.add_argument("--spreads", nargs="*", default=[], help="NAME=FILE.npy: other spreads over the same values")
    parser.add_argument("--normal", required=True, help="NAME=DIR: the distance from normal")
    parser.add_argument("--raw-error", help="NAME=DIR@component: the error before the climatology")
    parser.add_argument("--ratio-of", help="Name in --spreads of the spread of that error: the ratio is read too")
    parser.add_argument("--course", help="NAME=DIR: a score of how fast the weather changes")
    parser.add_argument("--inputs-run", type=Path, required=True,
                        help="Scored run of the detector: its shared cache gives the standardised inputs")
    parser.add_argument("--top-percent", type=float, default=5)
    parser.add_argument("--bands", type=int, default=20, help="Bands of the score, of the same size")
    parser.add_argument("--classes", type=int, default=5, help="Classes of the spread within each band")
    parser.add_argument("--limit", type=int, default=8, help="Positions beyond this many cells count here")
    parser.add_argument("--csv", type=Path, required=True, help="Classes; the positions go to <name>_positions.csv")
    args = parser.parse_args(argv)
    if bool(args.raw_error) != bool(args.ratio_of):
        raise ValueError("--raw-error and --ratio-of go together")
    source = parse_score(args.events)
    manifest, scores, maps = open_score(source)
    spread_file = manifest.get("anomaly_std_file")
    if not spread_file:
        raise ValueError(f"{source[1]} kept no spread of the draws")
    grid = manifest["grid"]
    timestamps = pd.DatetimeIndex(np.load(source[1] / "test_timestamps.npy"))
    spreads = {"score": np.load(source[1] / spread_file, mmap_mode="r")}
    for text in args.spreads:
        name, _, location = text.partition("=")
        spreads[name] = np.load(location, mmap_mode="r")
    maps.extend(spreads.values())
    normal_source = parse_score(args.normal)
    _, normal, more = open_score(normal_source)
    maps.extend(more)
    raw_error = None
    if args.raw_error:
        _, raw_error, more = open_score(parse_score(args.raw_error))
        maps.extend(more)
    course, course_limit = None, None
    if args.course:
        _, course, more = open_score(parse_score(args.course))
        maps.extend(more)
    inputs = None
    try:
        for name, array in spreads.items():
            if array.shape != scores.shape:
                raise ValueError(f"The spread {name} is not over the values of the score")
        rows, cols = np.asarray(grid["rows"], dtype=np.int64), np.asarray(grid["cols"], dtype=np.int64)
        shape = (len(grid["latitudes"]), len(grid["longitudes"]))
        limit = score_limit(scores, args.top_percent)
        normal_limit = score_limit(normal, args.top_percent)
        if course is not None:
            course_limit = score_limit(course, args.top_percent)
        normal_quantiles = np.nanpercentile(np.asarray(normal[::QUANTILE_STRIDE], dtype=np.float64), PERCENTILES)
        spread_quantiles = {name: np.nanpercentile(np.asarray(array[::QUANTILE_STRIDE], dtype=np.float64), PERCENTILES)
                            for name, array in spreads.items()}
        names = list(spreads)
        if raw_error is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                sample = np.asarray(spreads[args.ratio_of][::QUANTILE_STRIDE], dtype=np.float64) / np.asarray(raw_error[::QUANTILE_STRIDE], dtype=np.float64)
            spread_quantiles["ratio"] = np.nanpercentile(np.where(np.isfinite(sample), sample, np.nan), PERCENTILES)
            names.append("ratio")
            del sample
        release(normal)
        inputs, first, _ = inputs_of(args.inputs_run, timestamps)
        positions = 2 * args.limit + 1
        histograms = {name: np.zeros((positions, BINS), dtype=np.int64) for name in names}
        solid = np.zeros(3, dtype=np.int64)  # still flagged one, two spreads lower; not flagged that pass one spread higher
        unflagged = 0
        kept = {key: [] for key in ("score", "normal", "largest", "raw", "course", *(f"spread {name}" for name in names))}
        for start in range(0, len(timestamps), CHUNK):
            stop = min(start + CHUNK, len(timestamps))
            block = np.asarray(scores[start:stop], dtype=np.float64)
            spread_blocks = {name: np.asarray(array[start:stop], dtype=np.float64) for name, array in spreads.items()}
            normal_block = np.asarray(normal[start:stop], dtype=np.float32)
            if raw_error is not None:
                raw_block = np.asarray(raw_error[start:stop], dtype=np.float64)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = spread_blocks[args.ratio_of] / raw_block
                spread_blocks["ratio"] = np.where(np.isfinite(ratio), ratio, np.nan)
            with np.errstate(invalid="ignore"):
                flagged = block > limit
                own = spread_blocks["score"]
                solid[0] += np.count_nonzero(flagged & (block - own > limit))
                solid[1] += np.count_nonzero(flagged & (block - 2 * own > limit))
                other = np.isfinite(block) & ~flagged
                solid[2] += np.count_nonzero(other & (block + own > limit))
                unflagged += np.count_nonzero(other)
            # The largest absolute standardised value of the cell, only where it is needed.
            at_step, at_cell = np.nonzero(flagged)
            standardised = np.asarray(inputs[first + start:first + stop], dtype=np.float32)
            chosen = np.abs(standardised[at_step, at_cell])
            finite_cell = np.isfinite(chosen).any(axis=1)
            largest = np.full(len(chosen), np.nan, dtype=np.float32)
            largest[finite_cell] = np.nanmax(chosen[finite_cell], axis=1)
            kept["score"].append(block[flagged].astype(np.float32))
            kept["normal"].append(normal_block[flagged])
            kept["largest"].append(largest)
            if course is not None:
                kept["course"].append(np.asarray(course[start:stop], dtype=np.float32)[flagged])
            if raw_error is not None:
                kept["raw"].append(raw_block[flagged].astype(np.float32))
            for name, values in spread_blocks.items():
                kept[f"spread {name}"].append(values[flagged].astype(np.float32))
            for step in range(stop - start):
                lattice = np.zeros(shape, dtype=np.int64)
                lattice[rows, cols] = flagged[step]
                placed = positions_at(lattice, args.limit)
                if placed is None:
                    continue
                position = placed[0][rows, cols] + args.limit
                for name, values in spread_blocks.items():
                    finite = np.isfinite(values[step])
                    bins = np.clip(np.interp(values[step][finite], spread_quantiles[name], PERCENTILES).round().astype(np.int64), 0, BINS - 1)
                    histograms[name] += np.bincount(position[finite] * BINS + bins, minlength=positions * BINS).reshape(positions, BINS)
            release(scores)
            release(normal)
            if raw_error is not None:
                release(raw_error)
            if course is not None:
                release(course)
            release_pages(inputs, *spreads.values())
    finally:
        for array in maps:
            array._mmap.close()
        if inputs is not None:
            inputs._mmap.close()
    kept = {key: np.concatenate(values) for key, values in kept.items() if values}
    table = []
    groupings = [(name, "score", kept["score"]) for name in names]
    if raw_error is not None:
        groupings += [(name, "raw error", kept["raw"]) for name in (args.ratio_of, "ratio")]
    for name, bands_on, by in groupings:
        table += classes_table(name, bands_on, by, kept["score"], kept[f"spread {name}"], kept["normal"], kept["largest"],
                               normal_limit, normal_quantiles, args.bands, args.classes, kept.get("course"), course_limit)
    table = pd.DataFrame(table)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.csv, index=False)
    position_rows = []
    for at in [*range(-args.limit, 0), *range(1, args.limit + 1)]:
        row = {"position": at, "values": int(histograms["score"][at + args.limit].sum())}
        for name, histogram in histograms.items():
            row[f"{name} q25"] = quantile_of(histogram[at + args.limit], 0.25)
            row[f"{name} percentile"] = quantile_of(histogram[at + args.limit], 0.5)
            row[f"{name} q75"] = quantile_of(histogram[at + args.limit], 0.75)
            level = row[f"{name} percentile"]
            row[f"{name} value"] = np.nan if np.isnan(level) else float(spread_quantiles[name][int(10 * level)])
        position_rows.append(row)
    by_position = pd.DataFrame(position_rows)
    by_position.to_csv(args.csv.with_name(f"{args.csv.stem}_positions.csv"), index=False)

    count = len(kept["score"])
    print(f"[dropout] {timestamps[0].date()} to {timestamps[-1].date()} | {count} values flagged by {source[0]} (highest "
          f"{args.top_percent:g}%) | spread: standard deviation of the score over the draws")
    print(f"still flagged one spread lower {100 * solid[0] / count:.1f}%, two spreads lower {100 * solid[1] / count:.1f}%; "
          f"not flagged, flagged one spread higher {100 * solid[2] / max(unflagged, 1):.1f}%; median spread over the "
          f"threshold {np.nanmedian(kept['spread score']) / abs(limit):.3f}")
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.max_rows", None):
        print("\nclasses at equal score (1: the draws agree most)")
        print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        print("\nthe spread across the events: median percentile, by cells from the border of the core (negative inside)")
        print(by_position[["position", *(f"{name} {part}" for name in names for part in ("percentile", "value"))]]
              .to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
