"""Score the test period with the last epoch checkpoint of a run trained without its final scoring.

A sweep run trains and validates only (`--skip-final-scoring`): it leaves its epoch checkpoints and a
metadata.json with the status `training_only`. This script continues that run from its last epoch
checkpoint with the same configuration, so that no epoch is trained, and writes the final scores in a
new directory, as a run with final scoring would have. Everything the run used is read from the run
itself: the configuration and the PCA, MMD and normal-mask directories from its metadata.json, the seed
and the shared cache from the checkpoint. Only the prepared data are given.

    python scripts/score_era5_stgan_run.py --run-dir outputs/stgan_wandb/era5/5qlj2lwg/results \
        --prepared-dir data/era5_prepared --output-dir outputs/stgan_good_sweep_2
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.cwd()))  # also runs as a copy outside the repository, started from its root
import torch

from physiq_pv.anomaly_detection.stgan.config import STGANCNNConfig
from scripts.run_era5_stgan import run_training


def scoring_inputs(run_dir):
    """The configuration, the seed, the epoch checkpoint and the directories the run used."""
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "training_only":
        raise ValueError(f"Not a run trained without its final scoring (status {metadata.get('status')!r}): {run_dir}")
    names = {field.name for field in fields(STGANCNNConfig)}
    config = STGANCNNConfig(**{name: value for name, value in metadata["config"].items() if name in names})
    checkpoint = run_dir / f"model_epoch_{config.epochs}.pt"
    if not checkpoint.is_file():
        raise ValueError(f"No checkpoint of the last epoch: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload["completed_epochs"]) != config.epochs:
        raise ValueError(f"{checkpoint} has {payload['completed_epochs']} epochs, the run {config.epochs}")
    backend = metadata["backend"]
    directory = lambda key: None if backend.get(key) is None else Path(backend[key]["directory"])
    seasonal = payload["normalization"].get("seasonal")
    built = None if seasonal is None else Path(seasonal["mean_file"]).parent
    if built is not None and not built.is_dir():
        raise ValueError(f"The shared cache of the run is not there: {built}")
    # The root of the shared cache: the run finds its own cache in it by the hash of its inputs.
    return dict(config=config, seed=int(payload["seed"]), resume_from=checkpoint,
                pca_reference_dir=directory("pca_reference"), mmd_reference_dir=directory("pca_mmd"),
                normal_masks_dir=directory("normal_masks"), shared_cache_dir=None if built is None else built.parent), built


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="The results directory of the run")
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for the scored run")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    inputs, built = scoring_inputs(args.run_dir)
    print(f"[score-run] {args.run_dir} -> {args.output_dir} | checkpoint {inputs['resume_from'].name} | seed "
          f"{inputs['seed']} | cache {built} | masks {inputs['normal_masks_dir']}", flush=True)
    metadata = run_training(prepared_dir=args.prepared_dir, output_dir=args.output_dir, device=args.device,
                            skip_final_scoring=False, **inputs)
    used = metadata["backend"].get("seasonal_normalization")
    if built is not None and (used is None or Path(used["mean_file"]).parent != built):
        raise ValueError("The scoring did not read the shared cache of the run")
    print(f"[score-run] {metadata['status']}: scores in {args.output_dir / 'scores'}", flush=True)
    return metadata


if __name__ == "__main__":
    main()
