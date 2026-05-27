#!/usr/bin/env bash
set -euo pipefail

# CL replay with Open-Meteo operational
OPENMETEO_PATH=${OPENMETEO_PATH:-data/openmeteo_piedmont_2019.nc}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
WANDB_MODE=${WANDB_MODE:-online}

if [ ! -f "$OPENMETEO_PATH" ]; then
    echo "ERROR: Open-Meteo file not found: $OPENMETEO_PATH"
    echo "Run: python scripts/download_openmeteo_historical_forecast.py first"
    exit 1
fi

export CUDA_VISIBLE_DEVICES WANDB_MODE

echo "=== CL Replay (Open-Meteo operational) ==="
python -m physiq_pv.continual.train_replay_continual \
    --data-mode real \
    --weather-source openmeteo_historical_forecast \
    --feature-set openmeteo_operational \
    --openmeteo-path "$OPENMETEO_PATH" \
    --window-days 30 \
    --pv-norm-mode kwp \
    "$@"
