#!/usr/bin/env python
"""W&B sweep member for the PVGIS-only SDE-proxy pipeline.

Run by a `wandb agent`. It owns the W&B run, reads the swept hyper-parameters
from `wandb.config`, then:

  1. builds the runner command (W&B OFF in the runner — this wrapper owns the
     run) and trains;
  2. runs the post-hoc analysis script;
  3. logs the post-hoc scalars (posthoc/daytime_picp, ...) to W&B.

No training / model / report logic lives here: it only orchestrates the runner
and the analysis script through physiq_pv.experiments.sde_proxy_pipeline.

Usage (via a sweep created with sde_proxy_pipeline.make_sweep_config):
    wandb sweep <sweep.yaml>
    wandb agent <entity>/<project>/<sweep_id>
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Repo root on sys.path so the script runs standalone (mirrors sibling scripts).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physiq_pv.experiments.sde_proxy_pipeline import (  # noqa: E402
    DEFAULT_CONFIG,
    PVGIS_DIR,
    TEST_ANOMALY_SCORES,
    TRAIN_ANOMALY_SCORES,
    WANDB_ENTITY,
    WANDB_PROJECT,
    build_analysis_command,
    build_train_command,
    build_posthoc_figures,
    log_posthoc_to_wandb,
    make_out_dir,
    make_run_name,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pvgis-dir", default=PVGIS_DIR)
    p.add_argument("--test-anomaly-scores", default=TEST_ANOMALY_SCORES)
    p.add_argument("--train-anomaly-scores", default=TRAIN_ANOMALY_SCORES)
    p.add_argument("--device", default="cuda")
    p.add_argument("--wandb-project", default=WANDB_PROJECT)
    p.add_argument("--wandb-entity", default=WANDB_ENTITY)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands that would run and exit (no training, no W&B).")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.dry_run:
        cfg = dict(DEFAULT_CONFIG)
        out_dir = make_out_dir(cfg)
        run_name = make_run_name(cfg)
        print("[dry-run] train:", " ".join(build_train_command(
            cfg, out_dir=out_dir, run_name=run_name, pvgis_dir=args.pvgis_dir,
            test_anomaly_scores=args.test_anomaly_scores,
            train_anomaly_scores=args.train_anomaly_scores,
            device=args.device, use_wandb=False)))
        print("[dry-run] analysis:", " ".join(build_analysis_command(out_dir, cfg)))
        return

    import wandb  # noqa: PLC0415 — only needed for a real sweep member

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    try:
        cfg = {**DEFAULT_CONFIG, **dict(wandb.config)}
        out_dir = make_out_dir(cfg)
        run_name = make_run_name(cfg)
        run.name = run_name
        wandb.config.update({"out_dir": out_dir, "run_name": run_name},
                            allow_val_change=True)

        # Runner W&B OFF: this wrapper owns the run and logs the post-hoc scalars.
        train_cmd = build_train_command(
            cfg, out_dir=out_dir, run_name=run_name,
            pvgis_dir=args.pvgis_dir,
            test_anomaly_scores=args.test_anomaly_scores,
            train_anomaly_scores=args.train_anomaly_scores,
            device=args.device, use_wandb=False)
        print("[sweep-member] training:", " ".join(train_cmd))
        subprocess.run(train_cmd, check=True)

        analysis_cmd = build_analysis_command(out_dir, cfg)
        print("[sweep-member] analysis:", " ".join(analysis_cmd))
        subprocess.run(analysis_cmd, check=True)

        figure_paths = build_posthoc_figures(out_dir)
        posthoc = log_posthoc_to_wandb(
            wandb,
            run,
            out_dir,
            figure_paths=figure_paths,
        )
        print("[sweep-member] posthoc summary:", posthoc["summary"])
        print("[sweep-member] posthoc artifact uploaded:", posthoc["artifact_uploaded"])
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
