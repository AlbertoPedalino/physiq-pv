#!/usr/bin/env bash
# Does the model see weather forming, and does its edge say where an anomaly moves? G_n (MODEL) against the union of
# the 8 baselines of run_era5_comparisons.sh (COMPARISON), each at its highest 5% with the threshold of the validation
# years, over the test years:
#   - formation (check_era5_formation_lead.py): the births of the baselines (a flag after 24 h of calm within 5 cells),
#     and whether the model flagged nearby before them, against chance; births at 5%, and births at 1% with the
#     baselines at 5% as a watcher too (does the model alone see them first);
#   - direction (check_era5_edge_direction.py): whether the side of an area of the baselines where only the model flags
#     points where the area goes 3 and 6 h later, against the ring, the past motion and the wind at 850 hPa (from the
#     prepared cache, PREPARED).
#
#   SCRIPTS=/tmp/lead/scripts bash /tmp/lead/scripts/run_era5_direction_formation.sh outputs/optuna27/direzione 2>&1 \
#       | tee outputs/optuna27_direzione.log
set -euo pipefail

OUTPUT=${1:?output folder}
SCRIPTS=${SCRIPTS:-$(dirname "$0")}
PYTHON=${PYTHON:-.venv/bin/python}
MODEL=${MODEL:-outputs/optuna27/test_generator_clim_day}
PREPARED=${PREPARED:-data/era5_prepared}
COMPARISON=${COMPARISON:-outputs/confronti_good2/climatology/test}
BASELINES=${BASELINES:-"normal change_3h pca cblof mcd hbos copod iforest"}

complete() { [ -f "$1/metadata.json" ] && grep -q '"status": "complete"' "$1/metadata.json"; }
complete "$MODEL" || { echo "no complete scores in $MODEL" >&2; exit 1; }
by=()
for name in $BASELINES; do complete "$COMPARISON/$name" || { echo "no scores of $name" >&2; exit 1; }; by+=("$COMPARISON/$name"); done
mkdir -p "$OUTPUT/logs"
step() { echo; echo "=== $(date '+%F %T') $*"; }

step "formation: births of the baselines at 5%"
"$PYTHON" "$SCRIPTS/check_era5_formation_lead.py" --score-dir "$MODEL" --by-dir "${by[@]}" --top-percent 5 \
    --by-top-percent 5 --csv "$OUTPUT/nascite_5.csv" > "$OUTPUT/logs/nascite_5.log" 2>&1
cat "$OUTPUT/logs/nascite_5.log"
step "formation: births of the baselines at 1%, the baselines at 5% watching too"
"$PYTHON" "$SCRIPTS/check_era5_formation_lead.py" --score-dir "$MODEL" --by-dir "${by[@]}" --top-percent 5 \
    --by-top-percent 1 --early-percent 5 --csv "$OUTPUT/nascite_1.csv" > "$OUTPUT/logs/nascite_1.log" 2>&1
cat "$OUTPUT/logs/nascite_1.log"
step "direction: the edge of the model against ring, past motion and wind at 850 hPa"
"$PYTHON" "$SCRIPTS/check_era5_edge_direction.py" --score-dir "$MODEL" --by-dir "${by[@]}" --top-percent 5 \
    --by-top-percent 5 --wind-prepared "$PREPARED" --csv "$OUTPUT/direzione_5.csv" > "$OUTPUT/logs/direzione_5.log" 2>&1
cat "$OUTPUT/logs/direzione_5.log"
step "done: $OUTPUT"
