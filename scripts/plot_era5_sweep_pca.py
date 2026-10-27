"""Is the Bayesian sweep converging? Its four swept parameters run after run, with the objective.

The runs of the W&B sweep are read in the order they started. Every run has four parameters: the learning
rate of G, the ratio of the learning rates of D and G and the weight of the reconstruction, drawn log-uniform
by the sweep, and the D:G updates (1:2, 1:1 or 2:1). The figure has three rows:
- every parameter against the order of the runs, inside the range of the sweep, coloured by the objective
  (validation PCA-MMD rolling mean, lower is better), the best run marked, runs with no objective (failed or
  running) hollow: a sweep that converges draws its later runs in a narrower band;
- every parameter against the objective: where the good runs are;
- the objective of every run with the best so far, and the distance of every run from the best one over the
  four parameters (log10 of the three drawn log-uniform, log2 of D over G for the updates, standardised): a
  sweep that converges stops improving and draws its new runs closer to the best.
The check also prints how much each parameter spreads in the first and in the second half of the runs, as a
share of its spread if drawn uniformly over the range of the sweep (1: no narrowing), and the variance kept by
the principal components of the four standardised parameters (25% each if they were independent).
The sweep is Bayesian on few runs: the figure shows a trend, not a proof of convergence.

    python scripts/plot_era5_sweep_pca.py --output outputs/analisi_sweep/sweep_pca.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SWEEP = "albertopedalino-politecnico-di-torino/physiq_pv/913ez3rt"
METRIC = "validation/pca_mmd_rolling_mean"
# Config key, label, range of the sweep (sweeps/stgan_era5_normal_validation.yaml) as coordinates.
PARAMETERS = (("generator_learning_rate", "learning rate G", (np.log10(1e-4), np.log10(3e-3))),
              ("discriminator_lr_ratio", "learning rate D / G", (np.log10(0.25), np.log10(4.0))),
              ("generator_reconstruction_weight", "reconstruction weight", (np.log10(50.0), np.log10(2000.0))),
              ("discriminator_generator_update_ratio", "updates D:G", (-1.0, 1.0)))
UPDATES = {-1.0: "1:2", 0.0: "1:1", 1.0: "2:1"}
UNIFORM_UPDATES_STD = np.sqrt(2 / 3)  # three values drawn alike


def coordinates(config):
    """The four coordinates of a run from its config (log10 of the first three, log2 of D over G), or None when
    one is missing."""
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


def narrowing(points):
    """For every coordinate, its standard deviation in the first and in the second half of the runs over that of
    a uniform draw over the range of the sweep: [4, 2]."""
    uniform = np.array([(high - low) / np.sqrt(12) for _, _, (low, high) in PARAMETERS[:3]] + [UNIFORM_UPDATES_STD])
    half = len(points) // 2
    return np.stack([points[:half].std(axis=0), points[half:].std(axis=0)], axis=1) / uniform[:, None]


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
    from matplotlib.ticker import MaxNLocator, NullFormatter

    names = [row[0] for row in rows]
    points = np.array([row[2] for row in rows], dtype=float)
    values = np.array([row[3] for row in rows], dtype=float)
    standard = principal_plane(points)[0]
    done = np.isfinite(values)
    best = int(np.nanargmin(values))
    order = np.arange(1, len(rows) + 1)
    low, high = np.nanmin(values), np.nanmax(values)
    fig = plt.figure(figsize=(18, 12))
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.1], hspace=0.4, wspace=0.3)

    real = np.column_stack([10 ** points[:, :3], points[:, 3]])  # the values of the sweep; updates as log2 D/G

    def parameter_axis(ax, index, axis):
        """The axis of a parameter: logarithmic for the three drawn log-uniform, 1:2, 1:1, 2:1 for the updates."""
        if index == 3:
            target = ax.yaxis if axis == "y" else ax.xaxis
            target.set_ticks(list(UPDATES))
            target.set_ticklabels(list(UPDATES.values()))
        else:
            (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
            target = ax.yaxis if axis == "y" else ax.xaxis
            target.set_minor_formatter(NullFormatter())
            if index == 1:  # a range too short for the powers of ten
                target.set_ticks([0.25, 0.5, 1, 2, 4])
                target.set_ticklabels(["0.25", "0.5", "1", "2", "4"])

    def runs(ax, x, y):
        shown = ax.scatter(x[done], y[done], c=values[done], cmap="viridis_r", vmin=low, vmax=high, s=60,
                           edgecolor="black", linewidth=0.5, zorder=3)
        ax.scatter(x[~done], y[~done], facecolor="none", edgecolor="grey", s=60, zorder=3)
        ax.scatter(x[best], y[best], marker="*", s=380, facecolor="none", edgecolor="crimson", linewidth=1.5, zorder=4)
        return shown

    for index, (_, label, (bottom, top)) in enumerate(PARAMETERS):
        ax = fig.add_subplot(grid[0, index])
        shown = runs(ax, order, real[:, index])
        for edge in (bottom, top):  # the range of the sweep
            ax.axhline(edge if index == 3 else 10 ** edge, color="grey", linestyle=":", linewidth=1)
        parameter_axis(ax, index, "y")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("run order")
        ax.set_title(label)
        ax = fig.add_subplot(grid[1, index])
        runs(ax, real[:, index], values)
        parameter_axis(ax, index, "x")
        margin = 0.1 * (top - bottom)  # the whole range of the sweep, as in the row above
        ax.set_xlim(*((bottom - margin, top + margin) if index == 3 else (10 ** (bottom - margin), 10 ** (top + margin))))
        ax.set_xlabel(label)
        ax.set_ylabel("PCA-MMD rolling mean" if index == 0 else "")
    fig.colorbar(shown, ax=fig.axes, shrink=0.45, location="right", label="PCA-MMD rolling mean (lower is better)")
    ax = fig.add_subplot(grid[2, :2])
    ax.plot(order[done], values[done], "o", color="black", label="run")
    best_so_far = np.fmin.accumulate(np.where(done, values, np.inf))
    ax.step(order, np.where(np.isfinite(best_so_far), best_so_far, np.nan), where="post", color="crimson",
            label="best so far")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("run order")
    ax.set_ylabel("PCA-MMD rolling mean")
    ax.set_title("objective")
    ax.legend()
    ax = fig.add_subplot(grid[2, 2:])
    distance = np.linalg.norm(standard - standard[best], axis=1)
    ax.plot(order, distance, "o-", color="black")
    ax.plot(order[best], 0, "*", color="crimson", markersize=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("run order")
    ax.set_ylabel("distance (standardised, 4 parameters)")
    ax.set_title(f"distance from the best run ({names[best]})")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
    return distance, best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep", default=SWEEP, help="entity/project/sweep id")
    parser.add_argument("--output", type=Path, required=True, help="PNG of the figure")
    args = parser.parse_args(argv)
    rows = read_sweep(args.sweep)
    if len(rows) < 4 or not any(np.isfinite(row[3]) for row in rows):
        raise ValueError("The sweep needs at least four runs with parameters and one with the objective")
    distance, best = plot(rows, args.output)
    points = np.array([row[2] for row in rows], dtype=float)
    variance = principal_plane(points)[2]
    print(f"[sweep] {args.sweep} | runs with parameters: {len(rows)}, with the objective: "
          f"{sum(np.isfinite(row[3]) for row in rows)}")
    half = len(rows) // 2
    print(f"\nspread of each parameter, first {half} runs / last {len(rows) - half}, over a uniform draw over the range "
          f"(1: no narrowing)")
    for (_, label, _), (first, last) in zip(PARAMETERS, narrowing(points)):
        print(f"{label:<24}{first:>8.2f}{last:>8.2f}")
    print("variance kept by the principal components of the four standardised parameters (25% each if independent): "
          + ", ".join(f"PC{index + 1} {100 * share:.0f}%" for index, share in enumerate(variance)))
    print(f"\n{'order':>5}  {'run':<24}{'state':<10}{'lr G':>10}{'lr D/G':>8}{'weight':>8}{'D:G':>5}"
          f"{'objective':>11}{'distance':>10}")
    for index, (name, state, point, value) in enumerate(rows):
        mark = "  best" if index == best else ""
        print(f"{index + 1:>5}  {name[:23]:<24}{state:<10}{10 ** point[0]:>10.2e}{10 ** point[1]:>8.2f}"
              f"{10 ** point[2]:>8.0f}{UPDATES.get(point[3], '?'):>5}{value:>11.5f}{distance[index]:>10.2f}{mark}")
    print(f"\nmean distance from the best run: first half {distance[:half].mean():.2f}, second half "
          f"{distance[half:].mean():.2f} (smaller later: the sweep draws closer to the best)")
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
