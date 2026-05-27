#!/usr/bin/env bash
set -euo pipefail

# CL replay with PVGIS legacy
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
WANDB_MODE=${WANDB_MODE:-online}

export CUDA_VISIBLE_DEVICES WANDB_MODE

echo "=== CL Replay (PVGIS legacy) ==="
python -m physiq_pv.continual.train_replay_continual \
    --data-mode real \
    --weather-source pvgis_legacy \
    --feature-set pvgis_legacy \
    --window-days 30 \
    --pv-norm-mode kwp \
    "$@"
