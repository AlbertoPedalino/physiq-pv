"""Why did the Bayesian sweep propose its runs? The choice of the W&B controller, replayed run after run.

The W&B Bayesian controller (wandb/sweeps, bayes_search.py) proposes a run from the runs before it: every
parameter goes to [0, 1] by its distribution (log-uniform values by the logarithm, categorical values by
their position: the i-th of k values at (i + 1) / k); a finished run counts with the minimum of the metric in
its history; a failed or crashed run counts with the worst value (the largest maximum of the metric in the
history of the terminal runs, the default "worst" imputation); the metric is standardised and a Gaussian
process with one Matern kernel (nu 1.5, one length scale for all the parameters, a fixed nugget of 1e-7, no
noise) is fitted; among 1000 random points of [0, 1]^d the one with the largest expected improvement is
proposed, or a random one when the fitted length scale is close to a bound of the kernel.

This check rebuilds, for every run from `--from-run` on, the data the controller had when it proposed it (the
runs started before it, taken as terminal: the sweep runs one agent), fits the same process and says:
- the fitted length scale and whether it is at a bound (then the controller picked a random point);
- the largest expected improvement among random points, and the largest near the best run so far (within
  `--near` of it in [0, 1] on every continuous parameter, its categorical values kept): if the first is far
  larger, the controller had no reason to look near the best;
- the prediction and its spread at the best run so far, and the expected improvement at the run actually
  proposed, as a percentile of the random points (high if the replay matches the controller).
With `--not-worst` the runs named there are left out instead of counted as the worst (e.g. a run stopped by
the machine, not by its parameters).
The replay follows the public code; the hosted controller may differ, and its random points are not known.

    python scripts/replay_era5_sweep_bayes.py --from-run 15 --not-worst neat-sweep-18
"""
from __future__ import annotations

import argparse

import numpy as np

SWEEP = "albertopedalino-politecnico-di-torino/physiq_pv/913ez3rt"
NUGGET = 1e-7  # GAUSSIAN_PROCESS_NUGGET of bayes_search.py
EPSILON = 1e-6  # STD_NUMERICAL_STABILITY_EPSILON
POINTS = 1000  # num_points_to_try
TERMINAL_BAD = ("failed", "crashed", "killed")


def normalizer(parameters):
    """Names of the searched parameters and a function from a run config to its point in [0, 1]^d."""
    names, maps = [], []
    for name, spec in sorted(parameters.items()):
        if "values" in spec and len(spec["values"]) > 1:
            values = [str(value) for value in spec["values"]]
            maps.append(lambda value, values=values: (values.index(str(value)) + 1) / len(values))
        elif spec.get("distribution") == "log_uniform_values":
            low, high = np.log(float(spec["min"])), np.log(float(spec["max"]))
            maps.append(lambda value, low=low, high=high: float(np.clip((np.log(float(value)) - low) / (high - low), 0, 1)))
        elif "value" in spec or ("values" in spec and len(spec["values"]) == 1):
            continue  # a constant is not searched
        else:
            raise ValueError(f"Parameter {name}: only log_uniform_values and categorical values are replayed")
        names.append(name)
    return names, lambda config: np.array([to_unit(config[name]) for name, to_unit in zip(names, maps)])


def gp_data(runs, upto, skip=()):
    """Points and metric values the controller had before proposing run `upto` (0-based): the runs before it,
    finished with the minimum of their history, failed or crashed with the worst value."""
    before = [run for run in runs[:upto] if run["name"] not in skip]
    maxima = [max(run["history"]) for run in before if run["history"]]
    worst = max(maxima) if maxima else 0.0
    points, values = [], []
    for run in before:
        if run["state"] == "finished":
            value = min(run["history"]) if run["history"] else worst
        elif run["state"] in TERMINAL_BAD:
            value = worst
        else:
            continue
        points.append(run["point"])
        values.append(value if np.isfinite(value) else worst)
    return np.array(points), np.array(values, dtype=float), worst


def fit(points, values):
    """The Gaussian process of the controller on standardised values, with their mean and spread."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    warnings.filterwarnings("ignore", category=ConvergenceWarning)  # as the controller: a bound is read below

    scale = max(float(np.abs(values).max()), 1.0)
    scaled = values / scale
    spread = scaled.std() + EPSILON / scale
    process = GaussianProcessRegressor(kernel=Matern(nu=1.5), n_restarts_optimizer=2, alpha=NUGGET, random_state=2)
    process.fit(points, (scaled - scaled.mean()) / spread)
    return process, scaled.mean() * scale, spread * scale


def expected_improvement(process, mean, spread, best, candidates):
    """Expected improvement (lower is better) at the candidates, and the prediction and its spread there."""
    from scipy.stats import norm

    predicted, deviation = process.predict(candidates, return_std=True)
    target = (best - mean) / spread
    z = -(predicted - target) / (deviation + EPSILON)
    return -(predicted - target) * norm.cdf(z) + deviation * norm.pdf(z), predicted * spread + mean, deviation * spread


def replay(runs, from_run, near, skip=(), seed=0, categorical=()):
    """One row per proposed run from `from_run` (1-based)."""
    rng = np.random.default_rng(seed)
    rows = []
    for upto in range(max(from_run, 3) - 1, len(runs)):
        points, values, worst = gp_data(runs, upto, skip)
        if len(values) < 2:
            continue
        process, mean, spread = fit(points, values)
        length = float(np.atleast_1d(process.kernel_.length_scale)[0])
        # As the controller: the log length scale close to a log bound of the kernel.
        at_bound = bool(np.isclose(process.kernel_.bounds, np.atleast_2d(process.kernel_.theta).T).any())
        candidates = rng.random((POINTS, points.shape[1]))
        gain, _, _ = expected_improvement(process, mean, spread, values.min(), candidates)
        best = points[np.argmin(values)]
        around = np.clip(best + rng.uniform(-near, near, (POINTS, points.shape[1])), 0, 1)
        around[:, list(categorical)] = best[list(categorical)]
        gain_near, _, _ = expected_improvement(process, mean, spread, values.min(), around)
        _, at_best, deviation_best = expected_improvement(process, mean, spread, values.min(), best[None])
        proposed = runs[upto]["point"]
        gain_proposed = expected_improvement(process, mean, spread, values.min(), proposed[None])[0][0]
        rows.append({"run": upto + 1, "name": runs[upto]["name"], "data": len(values), "worst": worst,
                     "length": length, "at_bound": at_bound, "best": values.min(),
                     "best_name": [run["name"] for run in runs[:upto] if run["name"] not in skip and
                                   run["state"] in ("finished", *TERMINAL_BAD)][int(np.argmin(values))],
                     "gain_max": gain.max(), "gain_near": gain_near.max(), "at_best": at_best[0],
                     "deviation_best": deviation_best[0], "gain_proposed": gain_proposed,
                     "percentile_proposed": 100 * np.mean(gain <= gain_proposed)})
    return rows


def read_sweep(path):
    """The parameters of the sweep, its runs in starting order with their point and metric history, and its metric."""
    import wandb

    sweep = wandb.Api().sweep(path)
    names, to_unit = normalizer(sweep.config["parameters"])
    metric = sweep.config["metric"]["name"]
    runs = []
    for run in sorted(sweep.runs, key=lambda run: run.created_at):
        history = [float(row[metric]) for row in run.scan_history(keys=[metric])
                   if row.get(metric) is not None and np.isfinite(float(row[metric]))]
        runs.append({"name": run.name, "state": run.state, "point": to_unit(run.config), "history": history})
    return names, runs, sweep.config["metric"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep", default=SWEEP, help="entity/project/sweep id")
    parser.add_argument("--from-run", type=int, default=3, help="First proposed run to replay (1-based)")
    parser.add_argument("--near", type=float, default=0.1, help="Half side of the box around the best run, in [0, 1]")
    parser.add_argument("--not-worst", nargs="*", default=[], help="Runs left out instead of counted as the worst")
    args = parser.parse_args(argv)
    names, runs, metric = read_sweep(args.sweep)
    if metric.get("goal") != "minimize" or metric.get("impute", "worst") != "worst":
        raise ValueError("The replay covers a minimized metric with the default worst imputation")
    categorical = [index for index, name in enumerate(names) if name == "discriminator_generator_update_ratio"]
    print(f"[replay] {args.sweep} | parameters in [0, 1]: {', '.join(names)} | runs: {len(runs)}"
          + (f" | left out: {', '.join(args.not_worst)}" if args.not_worst else ""))
    print(f"{'run':>4} {'proposed':<20}{'data':>5}{'length':>9}{'bound':>6}{'best so far':>22}{'worst':>9}"
          f"{'EI max':>10}{'EI near':>10}{'pred best':>11}{'sd best':>9}{'EI prop.':>10}{'pct':>6}")
    for row in replay(runs, args.from_run, args.near, set(args.not_worst), categorical=categorical):
        print(f"{row['run']:>4} {row['name'][:19]:<20}{row['data']:>5}{row['length']:>9.3g}"
              f"{'yes' if row['at_bound'] else 'no':>6}{row['best_name'][:13]:>14}{row['best']:>8.5f}{row['worst']:>9.5f}"
              f"{row['gain_max']:>10.3g}{row['gain_near']:>10.3g}{row['at_best']:>11.5f}{row['deviation_best']:>9.5f}"
              f"{row['gain_proposed']:>10.3g}{row['percentile_proposed']:>6.0f}")
    print("EI: expected improvement in standardised units (max over 1000 random points; near: around the best run "
          f"within {args.near:g}); pred/sd best: the process at the best run; pct: percentile of the proposed run's EI "
          "among the random points; bound yes: the controller picked a random point")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
