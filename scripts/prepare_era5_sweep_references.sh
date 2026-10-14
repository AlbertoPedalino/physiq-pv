#!/usr/bin/env bash
# What the sweep reads, built once and in order: the shared cache of the training years up to
# TRAIN_END_YEAR, the values of unusual weather flagged with no model, the PCA reference on the
# normal training fields and the MMD reference on the normal validation fields.
#
#   nohup bash scripts/prepare_era5_sweep_references.sh data/era5_prepared outputs/era5_sweep_1999 1999 \
#       > preparazione_sweep.log 2>&1 &
#
# The normalization options are those of the final run (seasonal normal, floors, fourth root of the
# precipitation, line in time for the temperatures and the soil water, daily-change weights); they
# name the cache, so the sweep has to use the same ones. Nothing existing is overwritten: every
# command refuses a directory that is not empty. The full output of each step is in its own log
# under the output root; this script prints the lines that matter.
set -euo pipefail

PREPARED=${1:?prepared dir}
OUTPUT=${2:?output root}
TRAIN_END_YEAR=${3:-1999}
CACHE_ROOT=${STGAN_SHARED_CACHE_DIR:-data/stgan_shared_cache}
FINAL_RUN=${FINAL_RUN:-outputs/stgan_convgru_fullgrid_recent16_cyclic_bf16_cleanD_wdaily_seasonal_trend_ep3}
PYTHON=${PYTHON:-.venv/bin/python}

step() { echo; echo "=== $(date '+%F %T') $*"; }

step "options of the final run that differ from the defaults (to compare with the ones below)"
"$PYTHON" -c "import json,dataclasses as d,sys; from physiq_pv.anomaly_detection.stgan.config import STGANCNNConfig as C; base=d.asdict(C()); c=json.load(open(sys.argv[1]+'/metadata.json'))['config']; [print(' ', k, '=', v) for k, v in c.items() if base.get(k) != v]" "$FINAL_RUN" || echo "  (final run not readable)"

mkdir -p "$OUTPUT"
step "shared cache, training through $TRAIN_END_YEAR"
"$PYTHON" scripts/run_era5_stgan.py train --prepared-dir "$PREPARED" --output-dir "$OUTPUT/cache_run" \
    --validation-holdout --train-end-year "$TRAIN_END_YEAR" --cnn-training-mode full_grid \
    --normalization seasonal --seasonal-std-floors total_precipitation=0.1 volumetric_soil_water_layer_1=0.1 \
    --seasonal-fourth-root-features total_precipitation --seasonal-trend linear \
    --seasonal-trend-features 2m_temperature temperature volumetric_soil_water_layer_1 \
    --reconstruction-weights daily_change --shared-cache-dir "$CACHE_ROOT" --device cpu --cache-only \
    > "$OUTPUT/cache_run.log" 2>&1
grep -E 'shared cache|line in time' "$OUTPUT/cache_run.log" || true
CACHE=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['backend']['shared_cache'])" "$OUTPUT/cache_run/metadata.json")
echo "cache: $CACHE"

step "normal masks"
"$PYTHON" scripts/run_era5_stgan.py normal-masks --prepared-dir "$PREPARED" --train-end-year "$TRAIN_END_YEAR" \
    --shared-cache-dir "$CACHE" --output-dir "$OUTPUT/normal_masks" > "$OUTPUT/normal_masks.log" 2>&1
grep 'normal-masks' "$OUTPUT/normal_masks.log" || true

step "PCA reference on the normal training fields"
"$PYTHON" scripts/run_era5_stgan.py pca-reference --prepared-dir "$PREPARED" --train-end-year "$TRAIN_END_YEAR" \
    --shared-cache-dir "$CACHE" --normal-masks-dir "$OUTPUT/normal_masks" --output-dir "$OUTPUT/pca_reference" \
    > "$OUTPUT/pca_reference.log" 2>&1
grep 'pca-reference' "$OUTPUT/pca_reference.log" || true

step "MMD reference on the normal validation fields"
"$PYTHON" scripts/run_era5_stgan.py mmd-reference --prepared-dir "$PREPARED" --train-end-year "$TRAIN_END_YEAR" \
    --shared-cache-dir "$CACHE" --normal-masks-dir "$OUTPUT/normal_masks" \
    --pca-reference-dir "$OUTPUT/pca_reference" --output-dir "$OUTPUT/mmd_reference" > "$OUTPUT/mmd_reference.log" 2>&1
grep 'mmd-reference' "$OUTPUT/mmd_reference.log" || true

step "done"
echo "export STGAN_PREPARED_DIR=$PREPARED STGAN_SHARED_CACHE_DIR=$CACHE_ROOT STGAN_NORMAL_MASKS_DIR=$OUTPUT/normal_masks STGAN_PCA_REFERENCE_DIR=$OUTPUT/pca_reference STGAN_MMD_REFERENCE_DIR=$OUTPUT/mmd_reference"
