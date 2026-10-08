"""W&B tracking and sweep configuration for the existing STGAN runners."""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

from physiq_pv.anomaly_detection.stgan.config import STGANCNNConfig

WANDB_ENTITY = "albertopedalino-politecnico-di-torino"
WANDB_PROJECT = "physiq_pv"


def default_config(backend):
    if backend == "pvgis":
        return STGANCNNConfig(precision="bf16")
    if backend == "era5":
        if "spatial_encoder" not in STGANCNNConfig.__dataclass_fields__:
            raise ValueError("ERA5/GAT requires the experiment/stgan-era5-gat-mc-dropout branch.")
        return STGANCNNConfig(precision="bf16", spatial_encoder="gat", batch_size=1,
            score_batch_size=1, num_workers=4, trend_steps=56, grid_crs="EPSG:4326",
            score_storage="memmap", validation_holdout=True, discriminator_chunk_size=10611)
    raise ValueError("backend must be era5 or pvgis")


RATE_FORMS = (("learning_rate", "generator_learning_rate"), ("discriminator_lr_ratio", "discriminator_learning_rate"))


def reject_duplicate_rates(names, where):
    """A configuration names each network's rate in one form: the legacy field or the independent one."""
    for legacy, independent in RATE_FORMS:
        if legacy in names and independent in names:
            raise ValueError(f"{where}: give {independent} or {legacy}, not both.")


def resolve_config(base, values, seed):
    """Use canonical dataclass names; reject misspelled or ignored sweep keys."""
    values = dict(values)
    seed = values.pop("seed", seed)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    unknown = set(values) - set(asdict(base))
    if unknown:
        raise ValueError(f"Unsupported STGAN configuration keys: {sorted(unknown)}")
    config = replace(base, **values)
    return config, seed


class RunLogger:
    def __init__(self, run):
        self.run = run
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("validation/*", step_metric="epoch")
        self.last_epoch = None

    def epoch(self, record):
        """train/<name> for the training columns, validation/<name> for the monitoring ones."""
        values = {"epoch": record["epoch"], "train/epoch_seconds": record["seconds"]}
        for key, value in record.items():
            if key in ("epoch", "seconds") or value is None or (isinstance(value, float) and math.isnan(value)):
                continue  # A metric that an epoch does not have is not logged for it.
            name = "validation/" + key[len("validation_"):] if key.startswith("validation_") else "train/" + key
            values[name] = value
        self.last_epoch = record["epoch"]
        self.run.log(values)

    def result(self, metadata):
        metrics = {f"performance/{key}": value
                   for key, value in metadata.get("performance", {}).items()
                   if isinstance(value, (int, float)) and math.isfinite(value)}
        if metrics:
            self.run.log(metrics)
        # The four min-max factors of the score and the raw components they rescale.
        normalization = metadata.get("score_normalization") or {}
        scores = {f"score/{name}": normalization[key] for name, key in (
            ("reconstruction_min", "r_min"), ("reconstruction_max", "r_max"),
            ("discriminator_min", "d_min"), ("discriminator_max", "d_max")) if key in normalization}
        for component, statistics in (metadata.get("score_statistics") or {}).items():
            scores.update({f"score/{component}_{name}": statistics[name]
                           for name in ("median", "p95", "mean", "std")})
        if scores:
            self.run.log(scores)
        # Complete validation after training: reconstruction error before min-max, valid points only.
        objective = metadata.get("validation_objective")
        if objective:
            values = {f"validation/reconstruction_raw_{name}": objective[name]
                      for name in ("mean", "std", "median", "p95", "median_plus_p95")}
            values.update({f"validation/objective_{name}": objective[name]
                           for name in ("candidate_points", "valid_points", "excluded_points", "seconds")})
            if self.last_epoch is not None:
                values["epoch"] = self.last_epoch
            self.run.log(values)
        if metadata.get("pca_reference"):  # Which fixed PCA feature space the run loaded.
            self.run.summary.update({"pca_reference": metadata["pca_reference"]})
        if metadata.get("pca_mmd"):  # The fixed reference of validation/pca_mmd_*: subsets, bandwidth, PCA.
            self.run.summary.update({"mmd_reference": metadata["pca_mmd"]})
        self.run.summary.update({"precision": metadata["precision"],
            "parameter_counts": metadata.get("parameter_counts", {}),
            "environment": metadata.get("environment", {})})


def execute_training(args, config, seed, output, on_epoch):
    """Delegate training and exports; never duplicate the model or loss loop."""
    if args.backend == "era5":
        from scripts.run_era5_stgan import run_training
        return run_training(prepared_dir=args.prepared_dir, output_dir=output,
            config=config, device=args.device, seed=seed, on_epoch=on_epoch,
            pca_reference_dir=getattr(args, "pca_reference_dir", None),
            mmd_reference_dir=getattr(args, "mmd_reference_dir", None),
            skip_final_scoring=getattr(args, "skip_final_scoring", False))["backend"]
    if getattr(args, "skip_final_scoring", False):
        raise ValueError("--skip-final-scoring is available for the ERA5 backend only.")
    from scripts.run_pvgis_stgan import run_stgan
    if getattr(config, "spatial_encoder", "convgru") != "convgru":
        raise ValueError("PVGIS runner requires spatial_encoder=convgru; use ERA5 for GAT.")
    root = run_stgan(manifest_path=args.manifest, out_dir=output,
        paper_top_k_percent=args.paper_top_k_percent, config=config, device=args.device,
        seeds=(seed,), on_epoch=on_epoch)
    return json.loads((root / f"seed_{seed}" / "metadata.json").read_text(encoding="utf-8"))["backend"]


def run_tracked(args):
    source = args.prepared_dir if args.backend == "era5" else args.manifest
    if source is None:
        raise ValueError("Set --prepared-dir / STGAN_PREPARED_DIR for ERA5 or --manifest / STGAN_MANIFEST for PVGIS.")
    overrides = {} if args.model_config is None else json.loads(args.model_config.read_text(encoding="utf-8"))
    if not isinstance(overrides, dict):
        raise ValueError("--model-config must contain a JSON object.")
    reject_duplicate_rates(overrides, "--model-config")
    base, seed = resolve_config(default_config(args.backend), overrides, args.seed)
    if args.dry_run:
        print(json.dumps({"entity": args.wandb_entity, "project": args.wandb_project,
            "backend": args.backend, "source": str(source), "seed": seed,
            "config": asdict(base)}, indent=2))
        return None
    if not source.exists():
        raise FileNotFoundError(source)
    import wandb
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # An agent supplies WANDB_SWEEP_ID and sampled config. init joins that run
    # automatically; standalone invocations log a regular run in the same project.
    with wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                    name=args.wandb_run_name, group=f"stgan-{args.backend}", job_type="train",
                    tags=["stgan", args.backend], mode=args.wandb_mode, dir=str(output_root),
                    config={**asdict(base), "seed": seed}) as run:
        run_dir = output_root / args.backend / run.id
        run_dir.mkdir(parents=True, exist_ok=False)
        identity = {"id": run.id, "url": run.url, "sweep_id": run.sweep_id,
                    "entity": run.entity, "project": run.project, "backend": args.backend,
                    "source": str(source.resolve()), "device": args.device,
                    "paper_top_k_percent": args.paper_top_k_percent, "status": "running"}
        path = run_dir / "wandb_run.json"
        def save_identity():
            path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
        save_identity()
        try:
            config, seed = resolve_config(base, dict(run.config), seed)
            if (args.backend == "era5" and run.sweep_id and config.validation_holdout
                    and getattr(args, "pca_reference_dir", None) is None):
                raise ValueError("Set --pca-reference-dir / STGAN_PCA_REFERENCE_DIR: every member of an ERA5 sweep "
                                 "loads the same PCA reference, built once with `run_era5_stgan.py pca-reference`.")
            if (args.backend == "era5" and run.sweep_id and config.validation_holdout
                    and getattr(args, "mmd_reference_dir", None) is None):
                raise ValueError("Set --mmd-reference-dir / STGAN_MMD_REFERENCE_DIR: the sweep objective "
                                 "validation/pca_mmd_rolling_mean needs the MMD reference, built once with "
                                 "`run_era5_stgan.py mmd-reference`.")
            identity.update(config=asdict(config), seed=seed)
            save_identity()
            run.summary.update({"backend": args.backend, "source": identity["source"],
                                "device": args.device, "paper_top_k_percent": args.paper_top_k_percent,
                                "generator_learning_rate": config.effective_generator_learning_rate,
                                "discriminator_learning_rate": config.effective_discriminator_learning_rate,
                                "discriminator_to_generator_lr_ratio": (
                                    config.effective_discriminator_learning_rate
                                    / config.effective_generator_learning_rate),
                                "discriminator_generator_update_ratio": config.discriminator_generator_update_ratio,
                                "discriminator_updates_per_batch": config.discriminator_updates_per_batch,
                                "generator_updates_per_batch": config.generator_updates_per_batch,
                                "output_dir": str(run_dir / "results"), "status": "running"})
            logger = RunLogger(run)
            metadata = execute_training(args, config, seed, run_dir / "results", logger.epoch)
            logger.result(metadata)
            identity["status"] = "complete"
            run.summary["status"] = "complete"
        except BaseException:
            identity["status"] = "failed"
            run.summary["status"] = "failed"
            raise
        finally:
            save_identity()
    return run_dir


def validate_sweep(config):
    """A draft can be stored, but cannot be registered without user choices."""
    if config.get("method") != "bayes":
        raise ValueError("STGAN sweep method must be bayes.")
    metric = config.get("metric") or {}
    if not isinstance(metric, dict) or not isinstance(metric.get("name"), str) or not metric["name"].strip():
        raise ValueError("Sweep draft: choose metric.name before creating the Bayesian sweep.")
    if metric.get("goal") not in ("minimize", "maximize"):
        raise ValueError("Sweep draft: choose metric.goal (minimize or maximize).")
    parameters = config.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("Sweep draft: define the hyperparameters and search space first.")
    reject_duplicate_rates(parameters, "Sweep draft")
    if config.get("entity") != WANDB_ENTITY or config.get("project") != WANDB_PROJECT:
        raise ValueError(f"Expected project {WANDB_ENTITY}/{WANDB_PROJECT}.")
    return config
