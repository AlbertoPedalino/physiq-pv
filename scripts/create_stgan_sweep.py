"""Validate a completed STGAN sweep config; register only with --create."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physiq_pv.experiments.stgan_wandb import validate_sweep


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--create", action="store_true", help="Register the completed sweep in W&B")
    args = parser.parse_args(argv)
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        parser.error("Sweep config must be a YAML mapping.")
    try:
        validate_sweep(config)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.create:
        print("Sweep config ready. Use --create to register it in W&B.")
        return
    import wandb
    sweep_id = wandb.sweep(config, entity=config["entity"], project=config["project"])
    print(f"wandb agent {config['entity']}/{config['project']}/{sweep_id}")


if __name__ == "__main__":
    main()
