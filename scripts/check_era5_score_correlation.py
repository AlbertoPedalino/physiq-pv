"""How do the scores of a run go together, in space and in time?

Up to four scores over the same period and cells, each named on the command line (`--scores NAME=DIR`, or
`NAME=DIR@component` for the generator or discriminator mean of a scored run), are compared two by two,
with the Pearson correlation and with the Spearman (rank) correlation, which a few very high values do not
decide. For instance the error of the generator G and the discriminator component D of a run, and the
same two after the climatology of the score (`run_era5_stgan.py climatology`, each its own run):
- in space: at every timestamp, the correlation of two scores over the cells. If two scores are high
  in the same places at the same time, it is near 1. The series over the period is drawn as a line;
- in time: at every cell, the correlation of two scores over the timestamps. If two scores rise and
  fall together at a place, it is near 1. It is drawn as a map, one per pair.
The discriminator component of a scored run is D(real) - D(generated), D's outputs on the observed and on
the reconstructed patch (1 = judged generated); `@real` reads D(real) alone and `@generated` D(generated),
the first minus the component, so that the two terms can be compared with G one at a time.
At the timestamps of `--map-times` (by default the one with the highest mean of the first score over the
cells) the scores are also drawn as maps, each with its own colour scale.

The scores are read into memory, about 4 bytes per value for each: with `--time-stride` only one timestamp
every that many is read (an odd number, so that every time of day is read).

    python scripts/check_era5_score_correlation.py --output-dir outputs/analisi_good2/correlazioni \
        --scores G=outputs/stgan_good_sweep_2_scored@generator D=outputs/stgan_good_sweep_2_scored@discriminator \
        G+clim=outputs/stgan_good_sweep_2_scored_gen_clim D+clim=outputs/stgan_good_sweep_2_scored_disc_clim
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts.check_era5_dust_lead import COMPONENTS, score_file

TERMS = ("real", "generated")  # the two outputs of D whose difference is the discriminator component

METHODS = ("pearson", "spearman")
MAX_SCORES = 4  # six pairs, one categorical colour each
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")  # one per pair, in order
SEASONS = {"Dec-Feb": (12, 1, 2), "Mar-May": (3, 4, 5), "Jun-Aug": (6, 7, 8), "Sep-Nov": (9, 10, 11)}
CHUNK = 256  # timestamps correlated at a time in space
CELL_BLOCK = 512  # cells correlated at a time in time
ROLLING_DAYS = 30  # the bold line of the spatial correlation: its mean over this many days
NAME = 32


def parse_score(text):
    """NAME=DIR or NAME=DIR@component -> (name, directory, component)."""
    name, separator, location = text.partition("=")
    if not separator or not name or not location:
        raise ValueError(f"Expected NAME=DIR or NAME=DIR@component, got {text!r}")
    directory, _, component = location.partition("@")
    component = component or "combined"
    if component not in (*COMPONENTS, *TERMS):
        raise ValueError(f"Unknown component {component!r} in {text!r}: one of {', '.join((*COMPONENTS, *TERMS))}")
    return name, Path(directory), component


class Difference:
    """first - second of two opened maps, read a slice at a time."""

    def __init__(self, first, second):
        self.first, self.second, self.shape = first, second, first.shape
        self.parts = (first, second)  # the maps read, for release_pages

    def __getitem__(self, key):
        return np.asarray(self.first[key], dtype=np.float32) - np.asarray(self.second[key], dtype=np.float32)


def open_component(directory, manifest, component):
    """The score of a run (a memory map, or the difference of two) and the maps to close afterwards."""
    if component not in TERMS:
        values = np.load(directory / score_file(manifest, component), mmap_mode="r")
        return values, [values]
    real = np.load(directory / "scores" / "discriminator_real.npy", mmap_mode="r")
    if component == "real":
        return real, [real]
    discriminator = np.load(directory / score_file(manifest, "discriminator"), mmap_mode="r")
    return Difference(real, discriminator), [real, discriminator]


def column(pair, method):
    return f"{pair[0]} - {pair[1]} {method}"


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
    for pair in combinations(blocks, 2):
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


def panels(count, columns, size):
    import matplotlib.pyplot as plt

    columns = min(columns, count)
    rows = -(-count // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(size[0] * columns, size[1] * rows), constrained_layout=True,
                                squeeze=False)
    for axis in axes.ravel()[count:]:
        axis.set_visible(False)
    return figure, axes.ravel()[:count]


def plot_spatial(path, timestamps, spatial, pairs):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(METHODS), 1, figsize=(11, 6.5), sharex=True, sharey=True)
    for axis, method in zip(axes, METHODS):
        for pair, colour in zip(pairs, COLOURS):
            series = pd.Series(spatial[column(pair, method)], index=timestamps)
            daily = series.resample("1D").mean()
            axis.plot(daily.index, daily.to_numpy(), color=colour, linewidth=0.5, alpha=0.3)
            axis.plot(daily.index, daily.rolling(ROLLING_DAYS, min_periods=ROLLING_DAYS // 2).mean().to_numpy(),
                      color=colour, linewidth=2, label=f"{pair[0]} – {pair[1]}")
        axis.axhline(0, color="#52514e", linewidth=0.8)
        axis.set_ylim(-1, 1)
        axis.set_ylabel(f"correlazione di {method.capitalize()}")
        axis.grid(axis="y", color="#e4e3df", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"Correlazione spaziale fra gli score a ogni istante (linea sottile: media del giorno; "
                    f"spessa: media su {ROLLING_DAYS} giorni)", fontsize=10)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.955),
                  ncols=min(len(pairs), 3), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.88 if len(pairs) > 3 else 0.92))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_temporal(path, grid, temporal, method, pairs):
    import matplotlib.pyplot as plt

    figure, axes = panels(len(pairs), 3, (5, 4.4))
    for axis, pair in zip(axes, pairs):
        mesh = draw_map(axis, grid, temporal[column(pair, method)], cmap="RdBu_r", vmin=-1, vmax=1)
        axis.set_title(f"{pair[0]} – {pair[1]}")
    figure.colorbar(mesh, ax=list(axes), shrink=0.8, label=f"correlazione temporale di {method.capitalize()}")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_scores(path, grid, timestamp, values):
    import matplotlib.pyplot as plt

    figure, axes = panels(len(values), 2 if len(values) == 4 else 3, (5.6, 4.4))
    for axis, (name, field) in zip(axes, values.items()):
        finite = field[np.isfinite(field)]
        low, high = np.percentile(finite, (1, 99)) if len(finite) else (0, 1)
        mesh = draw_map(axis, grid, field, cmap="Blues", vmin=low, vmax=high)
        figure.colorbar(mesh, ax=axis, shrink=0.8)
        axis.set_title(f"score {name}")
    figure.suptitle(f"{timestamp:%Y-%m-%d %H:%M} UTC (scala di colore: 1°–99° percentile della mappa)", fontsize=10)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scores", nargs="+", required=True,
                        help="NAME=DIR (the score of a run) or NAME=DIR@generator|discriminator|combined|real|generated")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the tables and the figures")
    parser.add_argument("--time-stride", type=int, default=1, help="Read one timestamp every this many (odd)")
    parser.add_argument("--map-times", nargs="*", default=[],
                        help="Timestamps of the score maps, e.g. 2010-06-01T12:00 (default: the highest mean of "
                             "the first score)")
    args = parser.parse_args(argv)
    if args.time_stride < 1 or args.time_stride % 2 == 0:
        raise ValueError("time-stride must be an odd positive number, so that every time of day is read")
    sources = [parse_score(text) for text in args.scores]
    names = [name for name, _, _ in sources]
    if not 2 <= len(names) <= MAX_SCORES or len(set(names)) != len(names):
        raise ValueError(f"Give from 2 to {MAX_SCORES} scores with different names")
    manifests = {name: json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
                 for name, directory, _ in sources}
    grid = manifests[names[0]]["grid"]
    every = pd.DatetimeIndex(np.load(sources[0][1] / "test_timestamps.npy"))
    for name, directory, _ in sources[1:]:
        if manifests[name]["grid"] != grid or not every.equals(pd.DatetimeIndex(np.load(directory / "test_timestamps.npy"))):
            raise ValueError(f"{name} is not over the grid and timestamps of {names[0]}")
    timestamps = every[::args.time_stride]
    stored, opened = {}, []
    try:
        for name, directory, component in sources:
            stored[name], maps = open_component(directory, manifests[name], component)
            opened.extend(maps)
            if stored[name].shape != (len(every), len(grid["rows"])):
                raise ValueError(f"{name} has shape {stored[name].shape}, expected ({len(every)}, {len(grid['rows'])})")
        print(f"[correlation] {', '.join(f'{name}={directory}@{component}' for name, directory, component in sources)}"
              f" | {timestamps[0]} to {timestamps[-1]}, one timestamp every {args.time_stride} ({len(timestamps)}) | "
              f"{len(grid['rows'])} cells | reading {len(names) * len(timestamps) * len(grid['rows']) * 4 / 1e9:.1f} GB",
              flush=True)
        # A copy: the files are closed below, and a stride of 1 would otherwise keep a view of them.
        scores = {name: np.array(values[::args.time_stride], dtype=np.float32, copy=True)
                  for name, values in stored.items()}
    finally:
        for values in opened:
            values._mmap.close()

    pairs = list(combinations(names, 2))
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

    plot_spatial(args.output_dir / "spatial_correlation.png", timestamps, spatial, pairs)
    for method in METHODS:
        plot_temporal(args.output_dir / f"temporal_correlation_{method}.png", grid, temporal, method, pairs)
    chosen = [pd.Timestamp(value) for value in args.map_times]
    if not chosen:
        chosen = [timestamps[int(np.nanargmax(np.nanmean(scores[names[0]], axis=1)))]]
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
