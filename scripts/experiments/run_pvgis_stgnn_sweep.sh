#!/usr/bin/env bash
# Launch a W&B sweep for the PVGIS-only ST-GNN experiment.
#
# Usage:
#   scripts/experiments/run_pvgis_stgnn_sweep.sh [SWEEP_YAML] [N_RUNS]
#
# Examples:
#   scripts/experiments/run_pvgis_stgnn_sweep.sh configs/sweeps/pvgis_stgnn_debug.yaml
#   scripts/experiments/run_pvgis_stgnn_sweep.sh configs/sweeps/pvgis_stgnn_ablation.yaml 20
#   scripts/experiments/run_pvgis_stgnn_sweep.sh configs/sweeps/pvgis_stgnn_calibrated_group.yaml 10
#
# It creates the sweep, captures the sweep id, then runs the agent bounded to
# N_RUNS via `wandb agent --count N <id>` (supported by current wandb; if your
# wandb predates --count, run `wandb agent <id>` and stop it after N runs).
# PYTHONPATH=$PWD is exported so `main.py`/`physiq_pv` import.
set -euo pipefail

SWEEP_YAML="${1:-configs/sweeps/pvgis_stgnn_debug.yaml}"
N_RUNS="${2:-}"

export PYTHONPATH="${PYTHONPATH:-$PWD}"

echo "[sweep] creating from ${SWEEP_YAML} ..."
# `wandb sweep` prints the id to stderr; tee it and parse the `wandb agent <id>` line.
CREATE_LOG="$(mktemp)"
wandb sweep "${SWEEP_YAML}" 2>&1 | tee "${CREATE_LOG}"
SWEEP_ID="$(grep -oE 'wandb agent [^ ]+' "${CREATE_LOG}" | tail -1 | awk '{print $3}')"
rm -f "${CREATE_LOG}"

if [[ -z "${SWEEP_ID}" ]]; then
  echo "[sweep] could not parse sweep id — run 'wandb agent <SWEEP_ID>' manually." >&2
  exit 1
fi

echo "[sweep] id=${SWEEP_ID}"
if [[ -n "${N_RUNS}" ]]; then
  echo "[sweep] running agent for ${N_RUNS} run(s) ..."
  wandb agent --count "${N_RUNS}" "${SWEEP_ID}"
else
  echo "[sweep] running agent (all grid runs) ..."
  wandb agent "${SWEEP_ID}"
fi
