#!/usr/bin/env python
"""W&B sweep member: one ensemble seed -> one MC-Dropout pvgis_stgnn run.

Launched by ``wandb agent``. The sweep varies ONLY the seed: ``wandb`` injects
``--seed=<value>`` (from the sweep's ``parameters``) into this wrapper, which

  1. derives a per-seed output dir and ensemble id (so members don't overwrite
     each other), and
  2. delegates to ``physiq_pv.experiments.pvgis_stgnn_runner`` with W&B logging
     ON (the runner logs metrics/summary; it does not upload any artifact).

The runner OWNS the W&B run: launched under ``wandb agent`` its ``wandb.init``
auto-joins the sweep run, so no W&B logic lives here (no double init).

All fixed hyper-parameters (``--pvgis-dir``, ``--train-years``, ``--epochs``,
``--mc-dropout``, ``--mc-samples``, ``--wandb-project`` ...) are baked into the
sweep ``command`` and pass straight through to the runner untouched.

Usage (sweep config built by the notebook)::

    wandb sweep <sweep.yaml>
    wandb agent <entity>/<project>/<sweep_id>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on sys.path so the script runs standalone (mirrors sibling scripts).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from physiq_pv.experiments.pvgis_stgnn_runner import main as runner_main  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # The only swept parameter: injected by `wandb agent` as `--seed=<value>`.
    p.add_argument("--seed", type=int, required=True)
    # Base dirs owned by this wrapper (per-seed derivation lives here, not in
    # the sweep config which cannot template paths from the seed).
    p.add_argument("--out-root", required=True,
                   help="Base dir; each member writes to <out-root>/seed<seed>.")
    p.add_argument("--ensemble-dir", required=True,
                   help="Shared dir for per-member ensemble predictions.")
    return p


def main() -> None:
    # Everything not declared here (pvgis-dir, model config, --mc-dropout,
    # --wandb-project, ...) is forwarded verbatim to the runner.
    args, passthrough = build_parser().parse_known_args()

    out_dir = str(Path(args.out_root) / f"seed{args.seed}")
    ensemble_id = f"seed{args.seed}"

    runner_argv = passthrough + [
        "--seed", str(args.seed),
        "--out-dir", out_dir,
        "--save-ensemble-predictions",
        "--ensemble-predictions-dir", args.ensemble_dir,
        "--ensemble-id", ensemble_id,
        "--wandb",
    ]

    print("[sweep-member] seed", args.seed, "-> out_dir", out_dir)
    print("[sweep-member] runner argv:", " ".join(runner_argv))
    sys.argv = ["pvgis_stgnn_runner", *runner_argv]
    runner_main()


if __name__ == "__main__":
    main()
