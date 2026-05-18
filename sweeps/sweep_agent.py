"""W&B sweep entry for bilstm-gat seed × bilstm_pooling experiment."""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import wandb


def sweep_main() -> None:
    with wandb.init() as run:
        cfg = dict(run.config)
        seed = cfg.get("seed")
        if seed is None:
            print("sweep_agent: no seed in run.config; aborting", file=sys.stderr)
            sys.exit(2)
        os.environ["SEEDS"] = str(int(seed))

        pooling = cfg.get("bilstm_pooling")
        if pooling is not None:
            os.environ["BILSTM_POOLING"] = str(pooling)

        from main import main as run_pipeline
        run_pipeline()


if __name__ == "__main__":
    sweep_main()
