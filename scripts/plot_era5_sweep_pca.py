"""Is the Bayesian sweep converging? Its four swept parameters in two dimensions, with the objective.

The runs of the W&B sweep are read in the order they started. Every run is a point of four coordinates: the
learning rate of G, the ratio of the learning rates of D and G and the weight of the reconstruction, on a
log10 scale as the sweep draws them, and the D:G updates as log2 of D over G (1:2 is -1, 1:1 is 0, 2:1 is 1).
Standardised, they go through a principal component analysis; the first two components are the plane of the
figure, with the share of the variance they keep. Three panels:
- the plane, every run coloured by its objective (validation PCA-MMD rolling mean, lower is better) and
  numbered in its order, the best marked; runs with no objective (failed or running) are hollow;
- the objective of every run and the best so far, in the order of the runs: a sweep that converges stops
  improving and its runs stay near the best value;
- the distance of every run from the best one in the standardised four-dimensional space (not the plane):
  a sweep that converges draws its new runs closer to the best region.
The sweep is Bayesian on few runs: the panels show a trend, not a proof of convergence.

    python scripts/plot_era5_sweep_pca.py --output outputs/analisi_sweep/sweep_pca.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SWEEP = "albertopedalino-politecnico-di-torino/physiq_pv/913ez3rt"
METRIC = "validation/pca_mmd_rolling_mean"
PARAMETERS = (("generator_learning_rate", "log10 lr G"), ("discriminator_lr_ratio", "log10 lr D/G"),
              ("generator_reconstruction_weight", "log10 rec. weight"),
              ("discriminator_generator_update_ratio", "log2 D:G updates"))


def coordinates(config):
    """The four coordinates of a run from its config, or None when one is missing."""
    try:
        d, g = (int(part) for part in str(config["discriminator_generator_update_ratio"]).split(":"))
        return [np.log10(float(config["generator_learning_rate"])), np.log10(float(config["discriminator_lr_ratio"])),
                np.log10(float(config["generator_reconstruction_weight"])), np.log2(d / g)]
    except (KeyError, ValueError, ZeroDivisionError):
        return None


def principal_plane(points):
    """Standardised points [runs, 4], their first two principal components [runs, 2], the share of the variance
    of each component and the loadings [4, 2]. A coordinate with no spread is left at zero."""
    spread = points.std(axis=0)
    standard = (points - points.mean(axis=0)) / np.where(spread > 0, spread, 1)
    _, singular, axes = np.linalg.svd(standard, full_matrices=False)
    variance = singular ** 2 / max((singular ** 2).sum(), 1e-12)
    return standard, standard @ axes[:2].T, variance, axes[:2].T


def objective(summary):
    """The objective of a run from its summary, or NaN."""
    value = summary.get(METRIC)
    if value is None:  # an older key, if the run logged it under another prefix
        value = next((v for k, v in summary.items() if str(k).endswith("pca_mmd_rolling_mean")), None)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def read_sweep(path):
    """Name, state, coordinates and objective of every run of the sweep with its parameters, in starting order."""
    import wandb

    runs = sorted(wandb.Api().sweep(path).runs, key=lambda run: run.created_at)
    rows = []
    for run in runs:
        point = coordinates(run.config)
        if point is not None:
            rows.append((run.name, run.state, point, objective(run.summary)))
    return rows


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    names = [row[0] for row in rows]
    points = np.array([row[2] for row in rows], dtype=float)
    values = np.array([row[3] for row in rows], dtype=float)
    standard, plane, variance, loadings = principal_plane(points)
    done = np.isfinite(values)
    best = int(np.nanargmin(values))
    order = np.arange(1, len(rows) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    shown = ax.scatter(plane[done, 0], plane[done, 1], c=values[done], cmap="viridis_r", s=80, edgecolor="black",
                       linewidth=0.5, zorder=3)
    ax.scatter(plane[~done, 0], plane[~done, 1], facecolor="none", edgecolor="grey", s=80, zorder=3)
    ax.scatter(*plane[best], marker="*", s=420, facecolor="none", edgecolor="crimson", linewidth=1.5, zorder=4)
    for index, (x, y) in enumerate(plane):
        ax.annotate(str(order[index]), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    scale = np.abs(plane).max() * 0.8
    for (dx, dy), (_, label) in zip(loadings, PARAMETERS):  # where each parameter grows in the plane
        ax.annotate("", (dx * scale, dy * scale), (0, 0), arrowprops={"arrowstyle": "->", "color": "grey"})
        ax.text(dx * scale * 1.08, dy * scale * 1.08, label, color="grey", fontsize=8, ha="center")
    ax.set_xlabel(f"PC1 ({100 * variance[0]:.0f}% of the variance)")
    ax.set_ylabel(f"PC2 ({100 * variance[1]:.0f}%)")
    ax.set_title("swept parameters, PCA plane (number = run order)")
    fig.colorbar(shown, ax=ax, label="PCA-MMD rolling mean (lower is better)")
    ax = axes[1]
    ax.plot(order[done], values[done], "o", color="black", label="run")
    best_so_far = np.fmin.accumulate(np.where(done, values, np.inf))
    ax.step(order, np.where(np.isfinite(best_so_far), best_so_far, np.nan), where="post", color="crimson",
            label="best so far")
    ax.set_xlabel("run order")
    ax.set_ylabel("PCA-MMD rolling mean")
    ax.set_title("objective")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax = axes[2]
    distance = np.linalg.norm(standard - standard[best], axis=1)
    ax.plot(order, distance, "o-", color="black")
    ax.plot(order[best], 0, "*", color="crimson", markersize=14)
    ax.set_xlabel("run order")
    ax.set_ylabel("distance from the best run (standardised, 4 parameters)")
    ax.set_title(f"distance from the best run ({names[best]})")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    return standard, plane, variance, loadings, distance, best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep", default=SWEEP, help="entity/project/sweep id")
    parser.add_argument("--output", type=Path, required=True, help="PNG of the three panels")
    args = parser.parse_args(argv)
    rows = read_sweep(args.sweep)
    if len(rows) < 3 or not any(np.isfinite(row[3]) for row in rows):
        raise ValueError("The sweep needs at least three runs with parameters and one with the objective")
    _, _, variance, loadings, distance, best = plot(rows, args.output)
    print(f"[sweep-pca] {args.sweep} | runs with parameters: {len(rows)}, with the objective: "
          f"{sum(np.isfinite(row[3]) for row in rows)} | PC1 {100 * variance[0]:.0f}%, PC2 {100 * variance[1]:.0f}% "
          f"of the variance")
    print(f"{'loadings':<22}{'PC1':>8}{'PC2':>8}")
    for (_, label), (one, two) in zip(PARAMETERS, loadings):
        print(f"{label:<22}{one:>8.2f}{two:>8.2f}")
    print(f"\n{'order':>5}  {'run':<28}{'state':<10}{'objective':>11}{'distance':>10}")
    for index, (name, state, _, value) in enumerate(rows):
        mark = "  best" if index == best else ""
        print(f"{index + 1:>5}  {name[:27]:<28}{state:<10}{value:>11.5f}{distance[index]:>10.2f}{mark}")
    half = len(rows) // 2
    print(f"\nmean distance from the best run: first half {distance[:half].mean():.2f}, "
          f"second half {distance[half:].mean():.2f} (smaller later: the sweep draws closer to the best region)")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
