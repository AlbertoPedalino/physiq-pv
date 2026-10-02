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
            score_storage="memmap")
    raise ValueError("backend must be era5 or pvgis")


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

    def epoch(self, record):
        self.run.log({"epoch": record["epoch"],
            "train/generator_loss": record["generator_loss"],
            "train/discriminator_loss": record["discriminator_loss"],
            "train/epoch_seconds": record["seconds"], "train/samples": record["samples"]})

    def result(self, metadata):
        metrics = {f"performance/{key}": value
                   for key, value in metadata.get("performance", {}).items()
                   if isinstance(value, (int, float)) and math.isfinite(value)}
        if metrics:
            self.run.log(metrics)
        self.run.summary.update({"precision": metadata["precision"],
            "parameter_counts": metadata.get("parameter_counts", {}),
            "environment": metadata.get("environment", {})})


def execute_training(args, config, seed, output, on_epoch):
    """Delegate training and exports; never duplicate the model or loss loop."""
    if args.backend == "era5":
        from scripts.run_era5_stgan import run_training
        return run_training(prepared_dir=args.prepared_dir, output_dir=output,
            config=config, device=args.device, seed=seed, on_epoch=on_epoch)["backend"]
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
            identity.update(config=asdict(config), seed=seed)
            save_identity()
            run.summary.update({"backend": args.backend, "source": identity["source"],
                                "device": args.device, "paper_top_k_percent": args.paper_top_k_percent,
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
    if config.get("entity") != WANDB_ENTITY or config.get("project") != WANDB_PROJECT:
        raise ValueError(f"Expected project {WANDB_ENTITY}/{WANDB_PROJECT}.")
    return config
