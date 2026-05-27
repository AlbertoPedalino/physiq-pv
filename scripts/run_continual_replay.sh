#!/bin/bash
# Run replay-based continual adaptation on real Piedmont 2019 data.
# Usage: bash scripts/run_continual_replay.sh
set -e

cd "$(dirname "$0")/.."

echo "============================================"
echo " Step 1: Sanity checks"
echo "============================================"
python tests/test_replay_continual.py
echo ""

echo "============================================"
echo " Step 2: Debug smoke test (real data)"
echo "============================================"
python -m physiq_pv.continual.train_replay_continual \
    --data-mode real --debug \
    --run-name smoke_test \
    --seed 42

echo ""
echo "--- Smoke test metrics ---"
cat outputs/continual_replay/smoke_test/metrics_per_window.csv
echo ""
echo "--- Smoke test summary ---"
cat outputs/continual_replay/smoke_test/final_summary.json
echo ""

echo "============================================"
echo " Step 3: Full run (Piedmont 2019)"
echo "============================================"
python -m physiq_pv.continual.train_replay_continual \
    --data-mode real \
    --initial-train-start 2019-03-01 \
    --initial-train-end 2019-05-31 \
    --window-months 1 \
    --replay-buffer-size 5000 \
    --replay-batch-size 64 \
    --replay-loss-weight 1.0 \
    --initial-epochs 5 \
    --update-epochs 1 \
    --seed 42 \
    --run-name full_2019

echo ""
echo "--- Full run metrics ---"
cat outputs/continual_replay/full_2019/metrics_per_window.csv
echo ""
echo "--- Full run summary ---"
cat outputs/continual_replay/full_2019/final_summary.json
echo ""
echo "============================================"
echo " Done. Outputs in outputs/continual_replay/"
echo "============================================"
