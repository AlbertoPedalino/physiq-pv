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
TOP=${TOP:-5}  # flagged: the highest TOP% of every score, threshold from the validation years
TEST_RUN=${TEST_RUN-outputs/stgan_optuna27_scored}  # empty (TEST_RUN=): no dropout spread, e.g. when MODEL is a baseline
COMPARISON=${COMPARISON:-outputs/confronti_good2/climatology/test}
BASELINES=${BASELINES:-"normal change_3h pca cblof mcd hbos copod iforest"}

complete() { [ -f "$1/metadata.json" ] && grep -q '"status": "complete"' "$1/metadata.json"; }
# MODEL may list several runs: the first score is then their union (e.g. the 8 baselines, with BY=the model and
# TEST_RUN= empty); BY, when set, lists the runs of the second score instead of BASELINES under COMPARISON.
BY=${BY:-}
models=()
for run in $MODEL; do complete "$run" || { echo "no complete scores in $run" >&2; exit 1; }; models+=("$run"); done
UNSEEN=${UNSEEN:-}  # e.g. data/era5_independent/independent: CAPE and gust, no detector reads them (independent check)
unseen=()
UNSEEN_HOURS=${UNSEEN_HOURS:-6}  # kinds of CAPE and gust within this many hours (24: never reached in a day)
[ -n "$UNSEEN" ] && unseen=(--unseen-dir "$UNSEEN" --unseen-hours "$UNSEEN_HOURS")
ratio=()
if [ -n "$TEST_RUN" ]; then
    [ -f "$TEST_RUN/scores/generator_std.npy" ] || { echo "no dropout spread in $TEST_RUN" >&2; exit 1; }
    ratio=(--ratio-run "$TEST_RUN" --ratio-hours ${RATIO_HOURS:-3 24})  # RATIO_HOURS="3 6 12 24": every window
fi
[ -f "$ONSETS" ] || { echo "no onsets in $ONSETS: run run_era5_known_events_lead.sh first" >&2; exit 1; }
by=()
if [ -n "$BY" ]; then
    for run in $BY; do complete "$run" || { echo "no complete scores in $run" >&2; exit 1; }; by+=("$run"); done
else
    for name in $BASELINES; do complete "$COMPARISON/$name" || { echo "no scores of $name" >&2; exit 1; }; by+=("$COMPARISON/$name"); done
fi
mkdir -p "$OUTPUT"

echo "=== $(date '+%F %T') edge lead of $MODEL against ${BY:-$BASELINES}, whole test period and known events -> $OUTPUT"
"$PYTHON" "$SCRIPTS/check_era5_edge_lead.py" --score-dir "${models[@]}" --by-dir "${by[@]}" --top-percent "$TOP" \
    --csv "$OUTPUT/bordo_periodo.csv" --events "$EVENTS" --onsets "$ONSETS" --ring-km 250 --before-hours 72 \
    --bootstrap 1000 --events-csv "$OUTPUT/bordo_eventi.csv" "${ratio[@]}" "${unseen[@]}"
echo "=== $(date '+%F %T') done: $OUTPUT"
