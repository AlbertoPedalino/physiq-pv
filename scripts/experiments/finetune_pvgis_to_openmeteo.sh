#!/usr/bin/env bash
set -euo pipefail

# PVGIS pretrain + Open-Meteo fine-tuning (domain adaptation)
#
# TODO: This requires checkpoint loading + resume with different weather source.
# The CL pipeline already supports checkpoint_initial.pt loading.
# Steps:
#   1. Train with PVGIS legacy -> checkpoint
#   2. Load checkpoint, switch to openmeteo, continue training
#
# Current approach: use CL pipeline with initial PVGIS window then switch.
# Full implementation deferred until Open-Meteo data is downloaded and validated.

OPENMETEO_PATH=${OPENMETEO_PATH:-data/openmeteo_piedmont_2019.nc}
PVGIS_CHECKPOINT=${PVGIS_CHECKPOINT:-""}

echo "=== Fine-tune PVGIS -> Open-Meteo ==="
echo "STATUS: placeholder script"
echo ""
echo "To implement:"
echo "  1. Run train_pvgis_legacy.sh -> produces checkpoint"
echo "  2. Load checkpoint into CL pipeline with --weather-source openmeteo"
echo "  3. Continue training on Open-Meteo data"
echo ""
echo "Manual equivalent:"
echo "  python -m physiq_pv.continual.train_replay_continual \\"
echo "    --data-mode real \\"
echo "    --weather-source openmeteo_historical_forecast \\"
echo "    --feature-set openmeteo_operational \\"
echo "    --openmeteo-path $OPENMETEO_PATH \\"
echo "    --initial-epochs 0 --update-epochs 3"
