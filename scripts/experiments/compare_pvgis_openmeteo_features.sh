#!/usr/bin/env bash
set -euo pipefail

PVGIS_PATH=${PVGIS_PATH:-data/piedmont_pvgis_2019.nc}
OPENMETEO_PATH=${OPENMETEO_PATH:-data/openmeteo_piedmont_2019.nc}

echo "=== Inspect PVGIS ==="
python scripts/inspect_weather_netcdf.py --path "$PVGIS_PATH"

echo ""
echo "=== Inspect Open-Meteo ==="
python scripts/inspect_weather_netcdf.py --path "$OPENMETEO_PATH"

echo ""
echo "=== Compare ==="
python scripts/compare_weather_features.py \
    --pvgis-path "$PVGIS_PATH" \
    --openmeteo-path "$OPENMETEO_PATH" \
    --out reports/weather_feature_comparison.md
