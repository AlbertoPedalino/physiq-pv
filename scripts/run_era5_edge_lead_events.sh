#!/usr/bin/env bash
# Where only the model flags, next to the area the baselines flag, do the baselines reach that cell later (the model
# saw it first) or did they leave it (the model holds on)? check_era5_edge_lead.py with G_n (MODEL) against the union
# of the 8 baselines of run_era5_comparisons.sh (COMPARISON), each at its highest 5% with the threshold of the
# validation years: over the whole test period, and on the 39 known events only (box and a ring of 250 km, from
# 72 h before the measured onset of check_era5_known_events_lead.py, ONSETS, to the end), with intervals from
# resampling the events, the excess event by event, and the relative spread of the Monte Carlo dropout of G
# (spread / mean of the error, from the scored run TEST_RUN) on fronts and trails.
#
#   EVENTS=/tmp/lead/docs/era5_casi_noti/lista_esatta.csv SCRIPTS=/tmp/lead/scripts \
#       bash /tmp/lead/scripts/run_era5_edge_lead_events.sh outputs/optuna27/bordo 2>&1 | tee outputs/optuna27_bordo.log
set -euo pipefail

OUTPUT=${1:?output folder}
SCRIPTS=${SCRIPTS:-$(dirname "$0")}
PYTHON=${PYTHON:-.venv/bin/python}
EVENTS=${EVENTS:-docs/era5_casi_noti/lista_esatta.csv}
ONSETS=${ONSETS:-outputs/optuna27/anticipo/inizi.csv}
MODEL=${MODEL:-outputs/optuna27/test_generator_clim_day}
TEST_RUN=${TEST_RUN:-outputs/stgan_optuna27_scored}
COMPARISON=${COMPARISON:-outputs/confronti_good2/climatology/test}
BASELINES=${BASELINES:-"normal change_3h pca cblof mcd hbos copod iforest"}

complete() { [ -f "$1/metadata.json" ] && grep -q '"status": "complete"' "$1/metadata.json"; }
complete "$MODEL" || { echo "no complete scores in $MODEL" >&2; exit 1; }
[ -f "$TEST_RUN/scores/generator_std.npy" ] || { echo "no dropout spread in $TEST_RUN" >&2; exit 1; }
[ -f "$ONSETS" ] || { echo "no onsets in $ONSETS: run run_era5_known_events_lead.sh first" >&2; exit 1; }
by=()
for name in $BASELINES; do complete "$COMPARISON/$name" || { echo "no scores of $name" >&2; exit 1; }; by+=("$COMPARISON/$name"); done
mkdir -p "$OUTPUT"

echo "=== $(date '+%F %T') edge lead of G_n against the 8 baselines, whole test period and known events -> $OUTPUT"
"$PYTHON" "$SCRIPTS/check_era5_edge_lead.py" --score-dir "$MODEL" --by-dir "${by[@]}" --top-percent 5 \
    --csv "$OUTPUT/bordo_periodo.csv" --events "$EVENTS" --onsets "$ONSETS" --ring-km 250 --before-hours 72 \
    --bootstrap 1000 --events-csv "$OUTPUT/bordo_eventi.csv" --ratio-run "$TEST_RUN"
echo "=== $(date '+%F %T') done: $OUTPUT"
