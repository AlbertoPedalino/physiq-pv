"""Search the GAN balance of the ERA5 STGAN with Optuna, a Gaussian process that estimates the noise.

The W&B Bayesian controller fits one length scale for all the parameters and no noise; on the noisy validation
objective its fit ends at a bound and it proposes random points. This search keeps the space, the fixed options
and the objective of the W&B sweep (`--sweep-config`) and replaces the controller with Optuna's GPSampler: one
length scale per parameter (automatic relevance determination) and a noise variance it estimates
(deterministic_objective False).

    import-sweep  starts the study from the runs of the W&B sweep with the seed of the search: a finished run with
                  the final value of the objective (its summary, the mean over the last epochs), a failed or
                  crashed one as a failed trial, which the sampler ignores.
    copy-study    starts the study from the finished trials of `--from-study`, in the space of `--sweep-config`:
                  a new space needs a new study, because the sampler keeps to the space of the finished trials.
                  The rate of D is worked out from the ratio if the new space searches it; a trial outside the new
                  space is left out. The copies are the starting point: the stop rules do not count them.
    run           proposes a trial, trains it as a sweep member would (`scripts/run_stgan_wandb.py` in
                  `--repo-root`, the same code as the sweep, with the fixed options and the proposed parameters,
                  the seed of the search), reads its objective from the run's metadata.json and reports it; then
                  the next. It stops after `--max-new` new trials, or when `--patience` new trials in a row do not
                  lower the best objective by at least `--min-improvement` (failed trials count as not lowering
                  it). A trial left running by an interrupted search is marked failed (and
                  interrupted) when the search restarts: the stop rules do not count it. A failed trial teaches
                  the sampler nothing, so it proposes the same point again: with `--penalize-non-finite`, a trial
                  stopped by a non-finite loss is recorded with the worst objective of the study instead.
    preview       on a copy of the study, without touching it: the next proposal, the length scale the Gaussian
                  process gives every parameter, the noise and signal it estimates in units of the objective, and
                  what it expects at the proposal and at the best trial (above the observed value of the best,
                  that is worse, when it takes part of that value for luck).
    report        prints the trials, the best value so far and whether the search would stop.

The study lives in an SQLite file (`--storage`): the search can stop and resume. Every trial is a W&B run named
optuna-<study>-<trial number>, whose files go to `--output-root`/trial_<number>.

    python scripts/optuna_era5_stgan.py import-sweep --study era5-gan --storage sqlite:///outputs/optuna/era5-gan.db
    python scripts/optuna_era5_stgan.py run --study era5-gan --storage sqlite:///outputs/optuna/era5-gan.db
    python scripts/optuna_era5_stgan.py copy-study --from-study era5-gan --study era5-gan-extended \
        --sweep-config sweeps/stgan_era5_optuna_extended.yaml --storage sqlite:///outputs/optuna/era5-gan.db
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
    """The searched space as Optuna distributions, the fixed options (seed left out) and the objective's name.
    A file with `extends` changes the parameters of the file it names (null removes one)."""
    import optuna
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "extends" in config:
        parent = yaml.safe_load((Path(path).parent / config["extends"]).read_text(encoding="utf-8"))
        parameters = {**parent["parameters"], **config["parameters"]}
        config = {**parent, "parameters": {name: spec for name, spec in parameters.items() if spec is not None}}
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


def copy_study(study, trials, space):
    """Add the finished `trials` of another study to `study`, in `space`; returns (copied, outside, not finished)."""
    import optuna

    if study.trials:
        raise ValueError("The study already has trials: copy into a new study")
    counts = {"copied": 0, "outside": 0, "not finished": 0}
    for trial in trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            counts["not finished"] += 1
            continue
        params = dict(trial.params)
        if "discriminator_learning_rate" in space and "discriminator_lr_ratio" in params:
            params["discriminator_learning_rate"] = (params["generator_learning_rate"]
                                                     * params.pop("discriminator_lr_ratio"))
        if not all(space[name]._contains(space[name].to_internal_repr(params[name])) for name in space):
            counts["outside"] += 1
            continue
        study.add_trial(optuna.trial.create_trial(
            params={name: params[name] for name in space}, distributions=space, value=trial.value,
            state=trial.state, user_attrs={"source": "sweep", "seed": trial.user_attrs.get("seed"),
                                           "wandb_run": trial.user_attrs.get("wandb_run", f"optuna-{trial.number}")}))
        counts["copied"] += 1
    return counts


def stop_reason(trials, max_new, patience, min_improvement):
    """Why the search stops, or None: `max_new` new trials done, or `patience` new trials in a row that did not
    lower the best value by `min_improvement` (a failed trial does not lower it)."""
    import optuna

    done = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL)
    old = [t.value for t in trials if t.user_attrs.get("source") == "sweep" and t.state == done[0]]
    new = sorted((t for t in trials if t.user_attrs.get("source") == "optuna" and t.state in done
                  and not t.user_attrs.get("interrupted")), key=lambda t: t.number)
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
    """Train one trial as a sweep member and return the final objective from its metadata.json, None if it failed,
    infinity if a non-finite loss stopped it."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    non_finite = False
    # -u: the output of the training reaches the log as it is written, not in blocks (it goes to a pipe).
    command = [sys.executable, "-u", str(repo_root / "scripts" / "run_stgan_wandb.py"), "--backend", "era5",
               "--skip-final-scoring", "--model-config", str(config_path), "--seed", str(seed),
               "--output-root", str(trial_dir), "--wandb-run-name", name]
    with open(trial_dir / "train.log", "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:  # flushed line by line: the search's own output may go to a pipe (tee)
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            non_finite = non_finite or line.startswith("FloatingPointError")
        code = process.wait()
    if code != 0:
        print(f"[optuna] the training exited with code {code}", flush=True)
        return math.inf if non_finite else None
    return read_objective(trial_dir)


def read_objective(trial_dir):
    """The final objective of the run trained in `trial_dir` (the mean of the validation PCA-MMD over the last epochs,
    the value the W&B summary of the sweep runs holds), or None."""
    found = sorted(Path(trial_dir).glob("era5/*/results/metadata.json"))
    if len(found) != 1:
        print(f"[optuna] {len(found)} metadata.json of the run in {trial_dir}: no objective", flush=True)
        return None
    metadata = json.loads(found[0].read_text(encoding="utf-8"))
    pca_mmd = (metadata.get("backend") or {}).get("pca_mmd") or {}  # run_era5_stgan.run_training nests it in backend
    value = pca_mmd.get("validation_pca_mmd_rolling_mean")
    if value is None or not math.isfinite(float(value)):
        print(f"[optuna] no finite validation_pca_mmd_rolling_mean in {found[0]}", flush=True)
        return None
    return float(value)


def run_search(study, space, fixed, seed, output_root, max_new, patience, min_improvement, train,
               penalize_non_finite=False):
    """Propose, train and report trials until a stop rule holds; `train(config_path, trial_dir, name)` returns the
    objective, None if the trial failed or infinity if a non-finite loss stopped it."""
    import optuna

    for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
        # Left running by an interrupted search: failed, and not counted by the stop rules.
        optuna.trial.Trial(study, trial._trial_id).set_user_attr("interrupted", True)
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        print(f"[optuna] trial {trial.number} was left running: marked failed (interrupted)", flush=True)
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
        if value == math.inf and penalize_non_finite:
            worst = max(t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
            trial.set_user_attr("non_finite", True)
            study.tell(trial, worst)
            print(f"[optuna] trial {trial.number}: non-finite loss, recorded with the worst objective {worst:.5f}",
                  flush=True)
        elif value is None or value == math.inf:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            print(f"[optuna] trial {trial.number} failed", flush=True)
        else:
            study.tell(trial, value)
            print(f"[optuna] trial {trial.number}: objective {value:.5f}, best {study.best_value:.5f}", flush=True)
    print(f"[optuna] stopped: {reason}", flush=True)
    return reason


def preview(trials, space, sampler_seed):
    """On a copy of the study: the next proposal and what the Gaussian process of the sampler believes (length
    scales of the parameters in [0, 1], noise and signal in units of the objective, prediction and its spread at
    the proposal and at the best trial), or None for the beliefs while the sampler still draws at random."""
    import numpy as np
    import optuna
    import torch
    from optuna._gp.search_space import SearchSpace
    from optuna.samplers._gp.sampler import _standardize_values

    sampler = optuna.samplers.GPSampler(seed=sampler_seed, deterministic_objective=False)
    copy = optuna.create_study(direction="minimize", sampler=sampler)
    copy.add_trials(trials)
    trial = copy.ask(fixed_distributions=space)
    if not sampler._gprs_cache_list:
        return dict(trial.params), None
    process = sampler._gprs_cache_list[0]
    search_space = sampler.infer_relative_search_space(copy, copy.trials[trial.number])
    complete = [t for t in copy.trials if t.state == optuna.trial.TrialState.COMPLETE]
    _, means, stds = _standardize_values(-np.array([[t.value] for t in complete]))  # as the sampler: maximized
    normalizer = SearchSpace(search_space)

    def predict(params):
        x = torch.from_numpy(normalizer.get_normalized_params(complete[:1], [params]))
        with torch.no_grad():
            mean, variance = process.posterior(x)
        return float(-(mean[0] * stds[0] + means[0])), float(torch.sqrt(variance[0]) * stds[0])

    best = min(complete, key=lambda t: t.value)
    beliefs = {"length_scales": dict(zip(search_space, (float(v) for v in process.length_scales))),
               "noise_sd": float(torch.sqrt(process.noise_var)) * float(stds[0]),
               "signal_sd": float(torch.sqrt(process.kernel_scale)) * float(stds[0]),
               "at_proposal": predict(dict(trial.params)),
               "best": (best.user_attrs.get("wandb_run", f"trial {best.number}"), best.value, predict(best.params))}
    return dict(trial.params), beliefs


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
        outcome = ("non-finite" if trial.user_attrs.get("non_finite") else f"{value:.5f}") if value is not None \
            else trial.state.name.lower()
        cells ="".join(f"{trial.params.get(name, ''):>13.4g}" if isinstance(trial.params.get(name), float)
                        else f"{str(trial.params.get(name, '')):>13}" for name in names)
        print(f"{trial.number:>5} {trial.user_attrs.get('source', ''):<7}"
              f"{str(trial.user_attrs.get('wandb_run', f'optuna-{trial.number}'))[:21]:<22}{cells}"
              f"{outcome:>11}{(f'{best:.5f}' if math.isfinite(best) else ''):>10}")
    reason = stop_reason(study.trials, max_new, patience, min_improvement)
    print(f"[optuna] {'would stop: ' + reason if reason else 'would go on'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("import-sweep", "copy-study", "preview", "run", "report"))
    parser.add_argument("--study", required=True, help="Name of the Optuna study")
    parser.add_argument("--from-study", help="copy-study: the study in the same storage to copy the trials from")
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
    parser.add_argument("--penalize-non-finite", action="store_true",
                        help="Record a trial stopped by a non-finite loss with the worst objective, not as failed")
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
    elif args.command == "copy-study":
        if not args.from_study:
            raise ValueError("copy-study needs --from-study")
        source = optuna.load_study(study_name=args.from_study, storage=args.storage)
        counts = copy_study(study, source.trials, space)
        print(f"[optuna] from {args.from_study}: {counts['copied']} copied, {counts['outside']} outside the new space, "
              f"{counts['not finished']} not finished")
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
                                                                              trial_dir, name),
                   args.penalize_non_finite)
    elif args.command == "preview":
        proposal, beliefs = preview(study.trials, space, args.sampler_seed)
        print(f"[optuna] next proposal: {json.dumps(proposal)}")
        if beliefs is None:
            print("[optuna] fewer finished trials than the sampler's start: the proposal is random")
        else:
            print("length scale of each parameter in [0, 1] (small: the objective changes fast along it; "
                  "large, beyond 1: it hardly matters)")
            for name, length in beliefs["length_scales"].items():
                print(f"  {name:<40}{length:>10.3g}")
            name, observed, (mean, spread) = beliefs["best"]
            print(f"noise of the objective (sd) {beliefs['noise_sd']:.5f}, signal (sd) {beliefs['signal_sd']:.5f}")
            print(f"best trial {name}: observed {observed:.5f}, the process expects {mean:.5f} +- {spread:.5f}")
            mean, spread = beliefs["at_proposal"]
            print(f"at the proposal the process expects {mean:.5f} +- {spread:.5f}")
    else:
        report(study, args.max_new, args.patience, args.min_improvement)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
