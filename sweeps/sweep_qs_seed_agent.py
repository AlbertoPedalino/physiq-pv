"""W&B sweep entry for offline QS-weighted training seed variance."""
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
            print("sweep_qs_seed_agent: no seed in run.config; aborting", file=sys.stderr)
            sys.exit(2)

        os.environ["SEEDS"] = str(int(seed))
        os.environ["BILSTM_POOLING"] = "last"
        os.environ["QS_LOSS_WEIGHTING"] = "1"
        os.environ["QS_LOSS_FLOOR"] = "0.2"

        from main import main as run_pipeline

        run_pipeline()


if __name__ == "__main__":
    sweep_main()
