"""Track an STGAN run, or execute one member launched by a W&B sweep agent."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physiq_pv.experiments.stgan_wandb import WANDB_ENTITY, WANDB_PROJECT, run_tracked


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("era5", "pvgis"), required=True)
    parser.add_argument("--prepared-dir", type=Path, default=os.environ.get("STGAN_PREPARED_DIR"))
    parser.add_argument("--manifest", type=Path, default=os.environ.get("STGAN_MANIFEST"))
    parser.add_argument("--pca-reference-dir", type=Path, default=os.environ.get("STGAN_PCA_REFERENCE_DIR"),
                        help="ERA5: fixed PCA feature space (run_era5_stgan.py pca-reference); loaded, never refitted")
    parser.add_argument("--mmd-reference-dir", type=Path, default=os.environ.get("STGAN_MMD_REFERENCE_DIR"),
                        help="ERA5: reference of the validation MMD (run_era5_stgan.py mmd-reference); loaded, never rebuilt")
    parser.add_argument("--output-root", type=Path,
                        default=os.environ.get("STGAN_WANDB_OUTPUT_ROOT", "outputs/stgan_wandb"))
    parser.add_argument("--model-config", type=Path, help="JSON overrides using STGANCNNConfig field names")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--paper-top-k-percent", type=float, default=1.0)
    parser.add_argument("--wandb-entity", default=WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--dry-run", action="store_true", help="Print defaults without W&B access or training")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_tracked(parse_args())
