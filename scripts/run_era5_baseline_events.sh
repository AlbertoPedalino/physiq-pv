#!/usr/bin/env bash
# The events of the baselines, as those of G in the results: for every detector compared with the model, its own
# events over the test years (its highest 5%, threshold from the validation years), their zones, sizes and types
# (check_era5_event_zones.py) and the profile of its score across them (check_era5_event_profile.py). The baselines
# do not depend on the trained model, so this can run before the final run is scored.
#
#   SCRIPTS=/tmp/baseev/scripts bash /tmp/baseev/scripts/run_era5_baseline_events.sh outputs/eventi_baseline 2>&1 \
#       | tee outputs/eventi_baseline.log
#
# COMPARISON is the output of run_era5_comparisons.sh (its climatology/test/<name> scores, with the threshold
# reference of the validation years); INPUTS the scored run whose shared cache gives the types of the events.
# A detector whose two tables are already written is skipped; the full output of each step is in OUTPUT/logs.
set -euo pipefail

OUTPUT=${1:?output root}
SCRIPTS=${SCRIPTS:-$(dirname "$0")}
PYTHON=${PYTHON:-.venv/bin/python}
COMPARISON=${COMPARISON:-outputs/confronti_good2}
INPUTS=${INPUTS:-outputs/stgan_good_sweep_2_scored}
BASELINES=${BASELINES:-"normal change_3h pca cblof mcd hbos copod iforest"}
TOP=${TOP:-5}

step() { echo; echo "=== $(date '+%F %T') $*"; }
complete() { [ -f "$1/metadata.json" ] && grep -q '"status": "complete"' "$1/metadata.json"; }

mkdir -p "$OUTPUT/logs"
for name in $BASELINES; do
    complete "$COMPARISON/climatology/test/$name" || { echo "no test scores of $name in $COMPARISON" >&2; exit 1; }
done
for name in $BASELINES; do
    scores="$name=$COMPARISON/climatology/test/$name"
    if [ ! -f "$OUTPUT/zone_${name}_${TOP}pc.csv" ]; then
        step "$name: zones, sizes and types of its own events, highest $TOP%"
        "$PYTHON" "$SCRIPTS/check_era5_event_zones.py" --events "$scores" --scores "$scores" --inputs-run "$INPUTS" \
            --top-percent "$TOP" --csv "$OUTPUT/zone_${name}_${TOP}pc.csv" > "$OUTPUT/logs/zone_$name.log" 2>&1
        tail -n 25 "$OUTPUT/logs/zone_$name.log"
    fi
    if [ ! -f "$OUTPUT/profilo_${name}_${TOP}pc.csv" ]; then
        step "$name: profile of its score across its own events, highest $TOP%"
        "$PYTHON" "$SCRIPTS/check_era5_event_profile.py" --events "$scores" --scores "$scores" --inputs-run "$INPUTS" \
            --top-percent "$TOP" --csv "$OUTPUT/profilo_${name}_${TOP}pc.csv" > "$OUTPUT/logs/profilo_$name.log" 2>&1
        tail -n 12 "$OUTPUT/logs/profilo_$name.log"
    fi
done
step "done: $OUTPUT"
