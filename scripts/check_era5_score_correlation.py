"""How do the score of the generator, the score of the discriminator and the combined score go together?

The three mean scores of one scored run (generator G, discriminator D, combined) are compared two by
two, with the Pearson correlation and with the Spearman (rank) correlation, which a few very high
values do not decide:
- in space: at every timestamp, the correlation of two scores over the cells. If two scores are high
  in the same places at the same time, it is near 1. The series over the period is drawn as a line;
- in time: at every cell, the correlation of two scores over the timestamps. If two scores rise and
  fall together at a place, it is near 1. It is drawn as a map, one per pair.
At the timestamps of `--map-times` (by default the one with the highest mean G over the cells) the three
scores are also drawn as maps, each with its own colour scale.

The scores are read into memory, about 4 bytes per value for each of the three: with `--time-stride`
only one timestamp every that many is read (an odd number, so that every time of day is read).

    python scripts/check_era5_score_correlation.py --run-dir outputs/stgan_good_sweep_2_scored \
        --output-dir outputs/analisi_good2/correlazioni
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts.check_era5_dust_lead import COMPONENTS, score_file

PAIRS = (("generator", "discriminator"), ("generator", "combined"), ("discriminator", "combined"))
METHODS = ("pearson", "spearman")
SHORT = {"generator": "G", "discriminator": "D", "combined": "combinato"}
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")  # one per pair, in the order of PAIRS
SEASONS = {"Dec-Feb": (12, 1, 2), "Mar-May": (3, 4, 5), "Jun-Aug": (6, 7, 8), "Sep-Nov": (9, 10, 11)}
CHUNK = 256  # timestamps correlated at a time in space
CELL_BLOCK = 512  # cells correlated at a time in time
ROLLING_DAYS = 30  # the bold line of the spatial correlation: its mean over this many days
NAME = 28


def column(pair, method):
    return f"{SHORT[pair[0]]}-{SHORT[pair[1]]} {method}"


def correlation(x, y, axis):
    """Pearson correlation of x and y along `axis`, over the values finite in both (NaN with fewer than 3 of them
    or where one of the two does not vary)."""
    keep = np.isfinite(x) & np.isfinite(y)
    count = keep.sum(axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(keep, x - np.where(keep, x, 0).sum(axis=axis, keepdims=True) / count, 0)
        y = np.where(keep, y - np.where(keep, y, 0).sum(axis=axis, keepdims=True) / count, 0)
        value = (x * y).sum(axis=axis) / np.sqrt((x * x).sum(axis=axis) * (y * y).sum(axis=axis))
    return np.where(count.squeeze(axis) >= 3, value, np.nan)


def ranks(values, axis):
    """Ranks along `axis` (ties get their mean rank), NaN where the value is not finite."""
    values = np.where(np.isfinite(values), values, np.nan)
    return rankdata(values, axis=axis, nan_policy="omit")


def pair_correlations(blocks, axis):
    """Both correlations of every pair of scores along `axis`: {column: values}."""
    result = {}
    ranked = {name: ranks(block, axis) for name, block in blocks.items()}
    for pair in PAIRS:
        result[column(pair, "pearson")] = correlation(blocks[pair[0]], blocks[pair[1]], axis)
        result[column(pair, "spearman")] = correlation(ranked[pair[0]], ranked[pair[1]], axis)
    return result


def spatial_correlations(scores):
    """At every timestamp, the correlations of the pairs of scores over the cells: {column: [timestamps]}."""
    parts = []
    length = len(next(iter(scores.values())))
    for start in range(0, length, CHUNK):
        blocks = {name: np.asarray(values[start:start + CHUNK], dtype=np.float64) for name, values in scores.items()}
        parts.append(pair_correlations(blocks, axis=1))
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def temporal_correlations(scores):
    """At every cell, the correlations of the pairs of scores over the timestamps: {column: [cells]}."""
    parts = []
    cells = next(iter(scores.values())).shape[1]
    for start in range(0, cells, CELL_BLOCK):
        blocks = {name: np.asarray(values[:, start:start + CELL_BLOCK], dtype=np.float64)
                  for name, values in scores.items()}
        parts.append(pair_correlations(blocks, axis=0))
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def quartiles(values):
    values = values[np.isfinite(values)]
    return np.percentile(values, (25, 50, 75)) if len(values) else np.full(3, np.nan)


def grid_image(grid, values):
    """The values of the cells on the latitude x longitude grid, NaN where there is no cell."""
    image = np.full((len(grid["latitudes"]), len(grid["longitudes"])), np.nan)
    image[np.asarray(grid["rows"]), np.asarray(grid["cols"])] = values
    return image


def draw_map(axis, grid, values, **style):
    latitudes, longitudes = np.asarray(grid["latitudes"], dtype=float), np.asarray(grid["longitudes"], dtype=float)
    mesh = axis.pcolormesh(longitudes, latitudes, grid_image(grid, values), shading="nearest", **style)
    axis.set_aspect(1 / np.cos(np.deg2rad(latitudes.mean())))
    axis.set_xlabel("longitudine (°)")
    axis.set_ylabel("latitudine (°)")
    return mesh


def plot_spatial(path, timestamps, spatial):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(METHODS), 1, figsize=(11, 6.5), sharex=True, sharey=True)
    for axis, method in zip(axes, METHODS):
        for pair, colour in zip(PAIRS, COLOURS):
            series = pd.Series(spatial[column(pair, method)], index=timestamps)
            daily = series.resample("1D").mean()
            axis.plot(daily.index, daily.to_numpy(), color=colour, linewidth=0.5, alpha=0.3)
            axis.plot(daily.index, daily.rolling(ROLLING_DAYS, min_periods=ROLLING_DAYS // 2).mean().to_numpy(),
                      color=colour, linewidth=2, label=f"{SHORT[pair[0]]} – {SHORT[pair[1]]}")
        axis.axhline(0, color="#52514e", linewidth=0.8)
        axis.set_ylim(-1, 1)
        axis.set_ylabel(f"correlazione di {method.capitalize()}")
        axis.grid(axis="y", color="#e4e3df", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"Correlazione spaziale fra gli score a ogni istante (linea sottile: media del giorno; "
                    f"spessa: media su {ROLLING_DAYS} giorni)", fontsize=10)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.955),
                  ncols=len(PAIRS), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_temporal(path, grid, temporal, method):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(PAIRS), figsize=(15, 4.6), constrained_layout=True)
    for axis, pair in zip(axes, PAIRS):
        mesh = draw_map(axis, grid, temporal[column(pair, method)], cmap="RdBu_r", vmin=-1, vmax=1)
        axis.set_title(f"{SHORT[pair[0]]} – {SHORT[pair[1]]}")
    figure.colorbar(mesh, ax=axes, shrink=0.8, label=f"correlazione temporale di {method.capitalize()}")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_scores(path, grid, timestamp, values):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(COMPONENTS), figsize=(15, 4.6), constrained_layout=True)
    for axis, name in zip(axes, ("generator", "discriminator", "combined")):
        finite = values[name][np.isfinite(values[name])]
        low, high = np.percentile(finite, (1, 99)) if len(finite) else (0, 1)
        mesh = draw_map(axis, grid, values[name], cmap="Blues", vmin=low, vmax=high)
        figure.colorbar(mesh, ax=axis, shrink=0.8)
        axis.set_title(f"score {SHORT[name]}")
    figure.suptitle(f"{timestamp:%Y-%m-%d %H:%M} UTC (scala di colore: 1°–99° percentile della mappa)", fontsize=10)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Scored run with the G and D mean scores")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the tables and the figures")
    parser.add_argument("--time-stride", type=int, default=1, help="Read one timestamp every this many (odd)")
    parser.add_argument("--map-times", nargs="*", default=[],
                        help="Timestamps of the score maps, e.g. 2010-06-01T12:00 (default: the highest mean G)")
    args = parser.parse_args(argv)
    if args.time_stride < 1 or args.time_stride % 2 == 0:
        raise ValueError("time-stride must be an odd positive number, so that every time of day is read")
    manifest = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    grid = manifest["grid"]
    every = pd.DatetimeIndex(np.load(args.run_dir / "test_timestamps.npy"))
    timestamps = every[::args.time_stride]
    stored = {name: np.load(args.run_dir / score_file(manifest, name), mmap_mode="r") for name in COMPONENTS}
    try:
        for name, values in stored.items():
            if values.shape != (len(every), len(grid["rows"])):
                raise ValueError(f"{name} scores have shape {values.shape}, expected ({len(every)}, {len(grid['rows'])})")
        print(f"[correlation] {args.run_dir.name} | {timestamps[0]} to {timestamps[-1]}, one timestamp every "
              f"{args.time_stride} ({len(timestamps)}) | {len(grid['rows'])} cells | reading "
              f"{3 * len(timestamps) * len(grid['rows']) * 4 / 1e9:.1f} GB", flush=True)
        # A copy: the files are closed below, and a stride of 1 would otherwise keep a view of them.
        scores = {name: np.array(values[::args.time_stride], dtype=np.float32, copy=True)
                  for name, values in stored.items()}
    finally:
        for values in stored.values():
            values._mmap.close()

    spatial = spatial_correlations(scores)
    temporal = temporal_correlations(scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(spatial, index=pd.Index(timestamps, name="timestamp")).to_csv(args.output_dir / "spatial_correlation.csv")
    np.savez(args.output_dir / "temporal_correlation.npz", rows=np.asarray(grid["rows"]), cols=np.asarray(grid["cols"]),
             latitudes=np.asarray(grid["latitudes"]), longitudes=np.asarray(grid["longitudes"]),
             **{key.replace(" ", "_"): values for key, values in temporal.items()})

    months = timestamps.month.to_numpy()
    print("spatial: correlation over the cells at every timestamp, quartiles 25/50/75 over the timestamps; "
          "then the median by season")
    print(f"{'':<{NAME}}{'q25':>8}{'median':>8}{'q75':>8}" + "".join(f"{season:>9}" for season in SEASONS))
    for key, values in spatial.items():
        by_season = [quartiles(values[np.isin(months, of_season)])[1] for of_season in SEASONS.values()]
        print(f"{key:<{NAME}}" + "".join(f"{value:>8.3f}" for value in quartiles(values))
              + "".join(f"{value:>9.3f}" for value in by_season))
    print("temporal: correlation over the timestamps at every cell, quartiles 25/50/75 over the cells; "
          "share of cells below 0")
    print(f"{'':<{NAME}}{'q25':>8}{'median':>8}{'q75':>8}{'below 0':>9}")
    for key, values in temporal.items():
        finite = values[np.isfinite(values)]
        print(f"{key:<{NAME}}" + "".join(f"{value:>8.3f}" for value in quartiles(values))
              + f"{100 * (finite < 0).mean() if len(finite) else np.nan:>8.1f}%")

    plot_spatial(args.output_dir / "spatial_correlation.png", timestamps, spatial)
    for method in METHODS:
        plot_temporal(args.output_dir / f"temporal_correlation_{method}.png", grid, temporal, method)
    chosen = [pd.Timestamp(value) for value in args.map_times]
    if not chosen:
        chosen = [timestamps[int(np.nanargmax(np.nanmean(scores["generator"], axis=1)))]]
    for timestamp in chosen:
        if timestamp not in timestamps:
            raise ValueError(f"{timestamp} is not among the timestamps read")
        row = timestamps.get_loc(timestamp)
        plot_scores(args.output_dir / f"scores_{timestamp:%Y%m%d_%H%M}.png", grid, timestamp,
                    {name: values[row] for name, values in scores.items()})
    print(f"[correlation] tables and figures in {args.output_dir} | score maps at "
          + ", ".join(f"{timestamp:%Y-%m-%d %H:%M}" for timestamp in chosen), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
