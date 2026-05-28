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

        # Optional weather-source selection from the sweep config so the run
        # doesn't depend on shell-exported env vars. Absent keys keep the
        # PVGIS-legacy default (and any pre-exported env still applies).
        for cfg_key, env_key in (
            ("weather_source", "WEATHER_SOURCE"),
            ("feature_set", "FEATURE_SET"),
            ("openmeteo_path", "OPENMETEO_PATH"),
        ):
            val = cfg.get(cfg_key)
            if val:
                os.environ[env_key] = str(val)

        from main import main as run_pipeline
        run_pipeline()


if __name__ == "__main__":
    sweep_main()
