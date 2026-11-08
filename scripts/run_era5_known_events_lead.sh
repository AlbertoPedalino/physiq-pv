#!/usr/bin/env bash
# The lead of the detectors on the known events (docs/era5_casi_noti/anticipo.md, rules of 8 November 2026): the
# onset of the 39 events of lista_esatta.csv measured on the raw ERA5 values, and the share of the box and of the
# rings around it flagged by G_n (MODEL, the climatology of G over the test years with the threshold of the
# validation) and by the 8 baselines of run_era5_comparisons.sh (COMPARISON), against random dates.
#
#   bash /tmp/lead/scripts/run_era5_known_events_lead.sh outputs/optuna27/anticipo 2>&1 \
#       | tee outputs/optuna27_anticipo.log
#
# Run from the checkout, with SCRIPTS pointing at a copy of the scripts when the checkout must not change.
set -euo pipefail

OUTPUT=${1:?output folder}
SCRIPTS=${SCRIPTS:-$(dirname "$0")}
PYTHON=${PYTHON:-.venv/bin/python}
EVENTS=${EVENTS:-docs/era5_casi_noti/lista_esatta.csv}
PREPARED=${PREPARED:-data/era5_prepared}
MODEL=${MODEL:-outputs/optuna27/test_generator_clim_day}
COMPARISON=${COMPARISON:-outputs/confronti_good2/climatology/test}
SIMPLE=${SIMPLE:-"normal change_3h"}
STATISTICAL=${STATISTICAL:-"pca cblof mcd hbos copod iforest"}

complete() { [ -f "$1/metadata.json" ] && grep -q '"status": "complete"' "$1/metadata.json"; }
complete "$MODEL" || { echo "no complete scores in $MODEL" >&2; exit 1; }
simple=() statistical=()
for name in $SIMPLE; do complete "$COMPARISON/$name" || { echo "no scores of $name" >&2; exit 1; }; simple+=("$name=$COMPARISON/$name"); done
for name in $STATISTICAL; do complete "$COMPARISON/$name" || { echo "no scores of $name" >&2; exit 1; }; statistical+=("$name=$COMPARISON/$name"); done

echo "=== $(date '+%F %T') lead on the known events -> $OUTPUT"
"$PYTHON" "$SCRIPTS/check_era5_known_events_lead.py" --events "$EVENTS" --prepared-dir "$PREPARED" \
    --train-end-year 1999 --model "G_n=$MODEL" --simple "${simple[@]}" --statistical "${statistical[@]}" \
    --output-dir "$OUTPUT"
echo "=== $(date '+%F %T') done: $OUTPUT"
