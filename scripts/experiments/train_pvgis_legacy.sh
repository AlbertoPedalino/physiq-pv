#!/usr/bin/env bash
set -euo pipefail

# Training with PVGIS legacy (default, unchanged)
DATA_ROOT=/data/SentinelPV
PVGIS_PATH=data/piedmont_pvgis_2019.nc
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
WANDB_MODE=${WANDB_MODE:-online}

export CUDA_VISIBLE_DEVICES WANDB_MODE

echo "=== PVGIS Legacy Training ==="
echo "PVGIS: ${PVGIS_PATH}"
python main.py
