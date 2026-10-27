"""Search the GAN balance of the ERA5 STGAN with Optuna, a Gaussian process that estimates the noise.

The W&B Bayesian controller fits one length scale for all the parameters and no noise; on the noisy validation
objective its fit ends at a bound and it proposes random points. This search keeps the space, the fixed options
and the objective of the W&B sweep (`--sweep-config`) and replaces the controller with Optuna's GPSampler: one
length scale per parameter (automatic relevance determination) and a noise variance it estimates
(deterministic_objective False).

    import-sweep  starts the study from the runs of the W&B sweep with the seed of the search: a finished run with
                  the final value of the objective (its summary, the mean over the last epochs), a failed or
                  crashed one as a failed trial, which the sampler ignores.
    run           proposes a trial, trains it as a sweep member would (`scripts/run_stgan_wandb.py` in
                  `--repo-root`, the same code as the sweep, with the fixed options and the proposed parameters,
                  the seed of the search), reads its objective from the run's metadata.json and reports it; then
                  the next. It stops after `--max-new` new trials, or when `--patience` new trials in a row do not
                  lower the best objective by at least `--min-improvement` (failed trials count as not lowering
                  it). A trial left running by an interrupted search is marked failed when the search restarts.
    report        prints the trials, the best value so far and whether the search would stop.

The study lives in an SQLite file (`--storage`): the search can stop and resume. Every trial is a W&B run named
optuna-<study>-<trial number>, whose files go to `--output-root`/trial_<number>.

    python scripts/optuna_era5_stgan.py import-sweep --study era5-gan --storage sqlite:///outputs/optuna/era5-gan.db
    python scripts/optuna_era5_stgan.py run --study era5-gan --storage sqlite:///outputs/optuna/era5-gan.db
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

SWEEP = "albertopedalino-politecnico-di-torino/physiq_pv/913ez3rt"
SWEEP_CONFIG = Path("sweeps/stgan_era5_normal_validation.yaml")
SEED = 20
ENVIRONMENT = ("STGAN_PREPARED_DIR", "STGAN_SHARED_CACHE_DIR", "STGAN_NORMAL_MASKS_DIR", "STGAN_PCA_REFERENCE_DIR",
               "STGAN_MMD_REFERENCE_DIR")  # the inputs of every sweep member


def read_sweep_config(path):
    """The searched space as Optuna distributions, the fixed options (seed left out) and the objective's name."""
    import optuna
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    space, fixed = {}, {}
    for name, spec in config["parameters"].items():
        if name == "seed":
            continue
        if "value" in spec:
            fixed[name] = spec["value"]
        elif spec.get("distribution") == "log_uniform_values":
            space[name] = optuna.distributions.FloatDistribution(float(spec["min"]), float(spec["max"]), log=True)
        elif "values" in spec:
            space[name] = optuna.distributions.CategoricalDistribution(list(spec["values"]))
        else:
            raise ValueError(f"Parameter {name}: only fixed values, log_uniform_values and value lists are searched")
    metric = config["metric"]
    if metric.get("goal") != "minimize":
        raise ValueError("The search minimizes its objective")
    return space, fixed, metric["name"]


def objective_value(summary, metric):
    """The final value of the objective in a W&B summary, or None when it is missing or not finite."""
    try:
        value = float(summary[metric])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def import_sweep(study, runs, space, metric, seed):
    """Add the runs of the sweep with `seed` to the study; returns (finished, failed, skipped)."""
    import optuna

    if study.trials:
        raise ValueError("The study already has trials: import the sweep once, into a new study")
    counts = {"finished": 0, "failed": 0, "skipped": 0}
    for run in runs:
        config = run["config"]
        if config.get("seed") != seed or run["state"] not in ("finished", "failed", "crashed", "killed"):
            counts["skipped"] += 1
            continue
        params = {name: config[name] for name in space}
        value = objective_value(run["summary"], metric) if run["state"] == "finished" else None
        state = optuna.trial.TrialState.COMPLETE if value is not None else optuna.trial.TrialState.FAIL
        study.add_trial(optuna.trial.create_trial(
            params=params, distributions=space, value=value, state=state,
            user_attrs={"source": "sweep", "wandb_run": run["name"], "wandb_id": run["id"], "seed": seed}))
        counts["finished" if value is not None else "failed"] += 1
    return counts


def stop_reason(trials, max_new, patience, min_improvement):
    """Why the search stops, or None: `max_new` new trials done, or `patience` new trials in a row that did not
    lower the best value by `min_improvement` (a failed trial does not lower it)."""
    import optuna

    done = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL)
    old = [t.value for t in trials if t.user_attrs.get("source") == "sweep" and t.state == done[0]]
    new = sorted((t for t in trials if t.user_attrs.get("source") == "optuna" and t.state in done),
                 key=lambda t: t.number)
    if len(new) >= max_new:
        return f"{len(new)} new trials, the most allowed"
    best, streak = (min(old) if old else math.inf), 0
    for trial in new:
        value = trial.value if trial.state == done[0] else None
        if value is not None and value < best - min_improvement:
            streak = 0
        else:
            streak += 1
        if value is not None:
            best = min(best, value)
    if streak >= patience:
        return f"{streak} new trials in a row did not lower the best value by {min_improvement:g}"
    return None


def train_with_subprocess(repo_root, config_path, seed, trial_dir, name):
    """Train one trial as a sweep member and return the final objective from its metadata.json, or None."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(repo_root / "scripts" / "run_stgan_wandb.py"), "--backend", "era5",
               "--skip-final-scoring", "--model-config", str(config_path), "--seed", str(seed),
               "--output-root", str(trial_dir), "--wandb-run-name", name]
    with open(trial_dir / "train.log", "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
    found = sorted(trial_dir.glob("era5/*/results/metadata.json"))
    if code != 0 or len(found) != 1:
        return None
    pca_mmd = json.loads(found[0].read_text(encoding="utf-8")).get("pca_mmd") or {}
    value = pca_mmd.get("validation_pca_mmd_rolling_mean")
    return float(value) if value is not None and math.isfinite(float(value)) else None


def run_search(study, space, fixed, seed, output_root, max_new, patience, min_improvement, train):
    """Propose, train and report trials until a stop rule holds; `train(config_path, trial_dir, name)` returns the
    objective or None."""
    import optuna

    for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)  # left running by an interrupted search
        print(f"[optuna] trial {trial.number} was left running: marked failed", flush=True)
    while (reason := stop_reason(study.trials, max_new, patience, min_improvement)) is None:
        trial = study.ask(fixed_distributions=space)
        trial.set_user_attr("source", "optuna")
        trial.set_user_attr("seed", seed)
        trial_dir = Path(output_root) / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        config_path = trial_dir / "model_config.json"
        config_path.write_text(json.dumps({**fixed, **trial.params}, indent=1), encoding="utf-8")
        name = f"optuna-{study.study_name}-{trial.number}"
        print(f"[optuna] trial {trial.number}: {json.dumps(trial.params)}", flush=True)
        value = train(config_path, trial_dir, name)
        if value is None:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            print(f"[optuna] trial {trial.number} failed", flush=True)
        else:
            study.tell(trial, value)
            print(f"[optuna] trial {trial.number}: objective {value:.5f}, best {study.best_value:.5f}", flush=True)
    print(f"[optuna] stopped: {reason}", flush=True)
    return reason


def report(study, max_new, patience, min_improvement):
    import optuna

    names = list(study.trials[0].distributions) if study.trials else []
    print(f"[optuna] study {study.study_name}: {len(study.trials)} trials")
    print(f"{'trial':>5} {'source':<7}{'run':<22}" + "".join(f"{name[:12]:>13}" for name in names)
          + f"{'objective':>11}{'best':>10}")
    best = math.inf
    for trial in study.trials:
        value = trial.value if trial.state == optuna.trial.TrialState.COMPLETE else None
        best = min(best, value) if value is not None else best
        cells = "".join(f"{trial.params.get(name, ''):>13.4g}" if isinstance(trial.params.get(name), float)
                        else f"{str(trial.params.get(name, '')):>13}" for name in names)
        print(f"{trial.number:>5} {trial.user_attrs.get('source', ''):<7}"
              f"{str(trial.user_attrs.get('wandb_run', f'optuna-{trial.number}'))[:21]:<22}{cells}"
              f"{(f'{value:.5f}' if value is not None else trial.state.name.lower()):>11}"
              f"{(f'{best:.5f}' if math.isfinite(best) else ''):>10}")
    reason = stop_reason(study.trials, max_new, patience, min_improvement)
    print(f"[optuna] {'would stop: ' + reason if reason else 'would go on'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("import-sweep", "run", "report"))
    parser.add_argument("--study", required=True, help="Name of the Optuna study")
    parser.add_argument("--storage", required=True, help="Optuna storage, e.g. sqlite:///outputs/optuna/era5-gan.db")
    parser.add_argument("--sweep", default=SWEEP, help="entity/project/sweep id of the W&B sweep to import")
    parser.add_argument("--sweep-config", type=Path, default=SWEEP_CONFIG,
                        help="YAML of the W&B sweep: searched space, fixed options and objective")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed of every trial (and of the imported runs)")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Checkout whose code trains the trials")
    parser.add_argument("--output-root", type=Path, help="Files of the trials; default outputs/optuna/<study>")
    parser.add_argument("--max-new", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-improvement", type=float, default=0.0004)
    parser.add_argument("--sampler-seed", type=int, default=0)
    args = parser.parse_args(argv)
    import optuna

    space, fixed, metric = read_sweep_config(args.repo_root / args.sweep_config
                                             if not args.sweep_config.is_absolute() else args.sweep_config)
    sampler = optuna.samplers.GPSampler(seed=args.sampler_seed, deterministic_objective=False)
    study = optuna.create_study(study_name=args.study, storage=args.storage, direction="minimize",
                                sampler=sampler, load_if_exists=True)
    if args.command == "import-sweep":
        import wandb

        runs = [{"name": run.name, "id": run.id, "state": run.state, "config": dict(run.config),
                 "summary": dict(run.summary)}
                for run in sorted(wandb.Api().sweep(args.sweep).runs, key=lambda run: run.created_at)]
        counts = import_sweep(study, runs, space, metric, args.seed)
        print(f"[optuna] imported from {args.sweep}: {counts['finished']} finished, {counts['failed']} failed, "
              f"{counts['skipped']} skipped (other seed or not ended)")
        report(study, args.max_new, args.patience, args.min_improvement)
    elif args.command == "run":
        missing = [name for name in ENVIRONMENT if not os.environ.get(name)]
        if missing:
            raise ValueError(f"Export the inputs of the sweep first: {', '.join(missing)}")
        output_root = args.output_root or Path("outputs") / "optuna" / args.study
        repo_root = args.repo_root.resolve()
        run_search(study, space, fixed, args.seed, output_root.resolve(), args.max_new, args.patience,
                   args.min_improvement,
                   lambda config_path, trial_dir, name: train_with_subprocess(repo_root, config_path, args.seed,
                                                                              trial_dir, name))
    else:
        report(study, args.max_new, args.patience, args.min_improvement)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
