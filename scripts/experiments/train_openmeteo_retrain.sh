#!/usr/bin/env bash
set -euo pipefail

# Full retrain with Open-Meteo Historical Forecast
DATA_ROOT=/data/SentinelPV
OPENMETEO_PATH=${OPENMETEO_PATH:-data/openmeteo_piedmont_2019.nc}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
WANDB_MODE=${WANDB_MODE:-online}

if [ ! -f "$OPENMETEO_PATH" ]; then
    echo "ERROR: Open-Meteo file not found: $OPENMETEO_PATH"
    echo "Run: python scripts/download_openmeteo_historical_forecast.py first"
    exit 1
fi

export CUDA_VISIBLE_DEVICES WANDB_MODE
export WEATHER_SOURCE=openmeteo_historical_forecast
export FEATURE_SET=openmeteo_operational
export OPENMETEO_PATH

echo "=== Open-Meteo Retrain ==="
echo "Source: ${OPENMETEO_PATH}"
python main.py
