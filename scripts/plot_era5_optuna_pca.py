"""Where is the Optuna search going? Its trials on the plane of the first two principal components.

The trials of `--studies` are read in order, the later study keeping only the trials the earlier ones do not have.
Every trial has four coordinates: the logarithm of the learning rates of G and of D and of the weight of the
reconstruction, and log2 of D over G for the updates, all standardised. A finished trial is coloured by the objective,
a trial named in `--non-finite` is a cross, a trial still running or left running by an interrupted search is a ring;
any other failed trial is left out, since it says nothing about its parameters (the server stopped it). The trials
proposed by Optuna are joined in order. The arrows show how each parameter enters the two components. Two components
of four keep only part of the distances, and the plane moves a little with every new trial: the script prints what
they keep, beside what the first one keeps by chance, and the distance of every trial from the best one in all four.

    python scripts/plot_era5_optuna_pca.py --storage sqlite:///outputs/optuna/era5-gan.db \
        --studies era5-gan era5-gan-extended-2 --output outputs/optuna/ricerca_pca.png \
        --non-finite apricot-sweep-7 stoic-sweep-12 easy-sweep-15 legendary-sweep-20 optuna-28
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from scripts.plot_era5_sweep_pca import chance_first_component, principal_plane

LABELS = ("lr G", "lr D", "$\\lambda$", "agg. D:G")
RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")  # one hue, light to dark
INK, MUTED, SURFACE = "#1f1f1d", "#6b6a65", "#ffffff"
NAMED = {"serene-sweep-14": "serene", "good-sweep-2": "good"}  # the runs named beside those of Optuna
# Where the names of the trials that end up close to each other go: offset in points, alignment.
OFFSETS = {"27": ((6, 12), "left"), "ext-21": ((-9, 8), "right"), "29": ((9, 6), "left"), "28": ((9, -13), "left"),
           "26": ((9, -13), "left"), "serene": ((-8, 8), "right"), "good": ((-8, -14), "right")}


def coordinates(params):
    """The four coordinates of a trial, with the rate of D worked out from the ratio if the trial has that."""
    rate_d = params.get("discriminator_learning_rate", None)
    if rate_d is None:
        rate_d = params["generator_learning_rate"] * params["discriminator_lr_ratio"]
    d, g = (int(part) for part in params["discriminator_generator_update_ratio"].split(":"))
    return [np.log10(params["generator_learning_rate"]), np.log10(rate_d),
            np.log10(params["generator_reconstruction_weight"]), np.log2(d / g)]


def read_trials(studies, non_finite):
    """Per trial, in order: name, whether Optuna proposed it, coordinates, objective, and its kind: finished,
    non-finite or running. The trial of a later study is named after that study, unless it is a copy."""
    import optuna

    rows, seen = [], set()
    for position, study in enumerate(studies):
        for trial in study.trials:
            copied = trial.user_attrs.get("wandb_run")
            name = copied or (f"optuna-{trial.number}" if position == 0 else f"ext-{trial.number}")
            if name in seen or not trial.params:
                continue
            seen.add(name)
            violated = any(value > 0 for value in trial.constraints.values())
            if trial.state == optuna.trial.TrialState.COMPLETE and not violated:
                kind = "finished"
            elif name in non_finite or violated:
                kind = "non-finite"
            elif trial.state == optuna.trial.TrialState.RUNNING:
                kind = "running"
            else:
                continue
            proposed = name.startswith(("optuna-", "ext-"))
            rows.append((name, proposed, coordinates(trial.params), trial.value if kind == "finished" else np.nan, kind))
    return rows


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patheffects as effects
    import matplotlib.pyplot as plt

    points = np.array([row[2] for row in rows], dtype=float)
    values = np.array([row[3] for row in rows], dtype=float)
    kinds = np.array([row[4] for row in rows])
    _, plane, variance, loadings = principal_plane(points)
    colors = LinearSegmentedColormap.from_list("objective", RAMP[::-1])
    fig = plt.figure(figsize=(11, 6.4), facecolor=SURFACE, layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=[3.3, 1], height_ratios=[1.25, 0.8, 0.09])
    ax = fig.add_subplot(grid[:, 0])
    ax.grid(color="#eceae4", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c8c6bf")

    path = plane[[index for index, row in enumerate(rows) if row[1]]]
    for start, end in zip(path[:-1], path[1:]):
        ax.annotate("", xy=end, xytext=start, zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.1, shrinkA=6, shrinkB=6))
    done = kinds == "finished"
    shown = ax.scatter(*plane[done].T, c=values[done], cmap=colors, s=90, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.scatter(*plane[kinds == "running"].T, facecolor=SURFACE, edgecolor=INK, s=90, linewidth=1.6, zorder=3,
               label="in corso o interrotta")
    ax.scatter(*plane[kinds == "non-finite"].T, marker="x", color=INK, s=60, linewidth=1.8, zorder=4,
               label="loss non finita")
    best = int(np.nanargmin(values))
    ax.scatter(*plane[best], marker="o", facecolor="none", edgecolor=INK, s=230, linewidth=1.3, zorder=4,
               label="migliore")
    ax.plot([], [], color=INK, linewidth=1.1, label="prove di Optuna, in ordine")

    for index, row in enumerate(rows):
        if row[1] or row[0] in NAMED:
            name = NAMED.get(row[0], row[0].removeprefix("optuna-"))
            xytext, align = OFFSETS.get(name, ((8, 7), "left"))
            text = ax.annotate(name, plane[index], xytext=xytext, ha=align, textcoords="offset points", fontsize=9.5,
                               color=INK, zorder=5)
            text.set_path_effects([effects.withStroke(linewidth=2.5, foreground=SURFACE)])
    ax.margins(0.09)
    ax.set_xlabel(f"PC1 ({100 * variance[0]:.0f}% della varianza)", color=INK)
    ax.set_ylabel(f"PC2 ({100 * variance[1]:.0f}% della varianza)", color=INK)
    ax.tick_params(colors=MUTED)

    arrows = fig.add_subplot(grid[0, 1])
    arrows.set_aspect("equal")
    arrows.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), xticks=[], yticks=[])
    for side in arrows.spines.values():
        side.set_visible(False)
    arrows.axhline(0, color="#eceae4", linewidth=0.8)
    arrows.axvline(0, color="#eceae4", linewidth=0.8)
    for loading, label, shift in zip(loadings, LABELS, ((0.02, -0.14), (0.02, 0.14), (-0.06, -0.12), (0.02, 0.12))):
        arrows.annotate("", xy=loading, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.3))
        arrows.text(loading[0] + shift[0], loading[1] + shift[1], label, color=INK, fontsize=9.5, ha="center", va="center")
    arrows.set_title("Direzione dei parametri", color=INK, fontsize=10)

    handles, names = ax.get_legend_handles_labels()
    fig.add_subplot(grid[1, 1]).set_axis_off()
    fig.axes[-1].legend(handles, names, frameon=False, loc="center", fontsize=9.5, labelcolor=INK)
    bar = fig.colorbar(shown, cax=fig.add_subplot(grid[2, 1]), orientation="horizontal")
    bar.set_label("MMD (più scuro: meglio)", color=INK, fontsize=9.5)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, labelsize=8.5)
    bar.ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    return variance, loadings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--storage", required=True, help="Optuna storage, e.g. sqlite:///outputs/optuna/era5-gan.db")
    parser.add_argument("--studies", nargs="+", required=True, help="Names of the studies, in the order they ran")
    parser.add_argument("--non-finite", nargs="*", default=(), help="Failed trials stopped by a non-finite loss")
    parser.add_argument("--output", type=Path, required=True, help="PNG of the figure")
    args = parser.parse_args(argv)
    import optuna

    studies = [optuna.load_study(study_name=name, storage=args.storage) for name in args.studies]
    rows = read_trials(studies, set(args.non_finite))
    variance, loadings = plot(rows, args.output)
    points = np.array([row[2] for row in rows], dtype=float)
    standard = principal_plane(points)[0]
    values = np.array([row[3] for row in rows], dtype=float)
    distance = np.linalg.norm(standard - standard[int(np.nanargmin(values))], axis=1)
    median, high = chance_first_component(len(rows))
    print(f"[optuna] {len(rows)} trials | variance: " + ", ".join(f"PC{i + 1} {100 * v:.0f}%" for i, v in enumerate(variance))
          + f" | PC1 by chance over {len(rows)} trials: median {100 * median:.0f}%, 95th percentile {100 * high:.0f}%")
    print(f"{'':<10}{'PC1':>7}{'PC2':>7}")
    for label, loading in zip(("lr G", "lr D", "weight", "updates"), loadings):
        print(f"{label:<10}{loading[0]:>7.2f}{loading[1]:>7.2f}")
    print(f"{'trial':<24}{'objective':>10}{'distance':>10}")
    for row, far in zip(rows, distance):
        print(f"{row[0]:<24}{(f'{row[3]:.5f}' if row[4] == 'finished' else row[4]):>10}{far:>10.2f}")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
