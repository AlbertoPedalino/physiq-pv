#!/usr/bin/env bash
# What does each detector see alone? For each of the 9 detectors (the model G_n, the 6 shallow methods, the 2
# statistical ones) the values it flags and none of the other 8 flags, each at its highest TOP% with the threshold of
# the validation years, over the whole test period:
#   - check_era5_variables_away.py: how many variables are far from their seasonal normal (<name>_variabili.log);
#   - check_era5_edge_lead.py with the unseen variables: next to the area of the other 8, CAPE and gust in their
#     highest 5%, fronts and trails, year by year (<name>_bordo.csv and _unseen_anni.csv);
#   - check_era5_alone_future.py: rain now and later, cloud cover and radiation above or below normal
#     (<name>_futuro.csv).
# Only the model has the dropout draws (ratio run). A step whose output is already there is skipped.
#
#   SCRIPTS=/tmp/lead/scripts bash /tmp/lead/scripts/run_era5_who_sees_what.sh outputs/optuna27/chi_vede 2>&1 \
#       | tee outputs/optuna27_chi_vede.log
set -euo pipefail

OUTPUT=${1:?output folder}
SCRIPTS=${SCRIPTS:-$(dirname "$0")}
PYTHON=${PYTHON:-.venv/bin/python}
MODEL=${MODEL:-outputs/optuna27/test_generator_clim_day}
SCORED=${SCORED:-outputs/stgan_optuna27_scored}  # dropout draws of the model and standardised inputs
COMPARISON=${COMPARISON:-outputs/confronti_good2/climatology/test}
BASELINES=${BASELINES:-"normal change_3h pca cblof mcd hbos copod iforest"}
UNSEEN=${UNSEEN:-data/era5_independent/independent}
PREPARED=${PREPARED:-data/era5_prepared}
TOP=${TOP:-5}

complete() { [ -f "$1/metadata.json" ] && grep -q '"status": "complete"' "$1/metadata.json"; }
check() {  # run a step; on failure show its log
    local log=$1; shift
    "$@" > "$log" 2>&1 || { echo "FAILED, log $log:" >&2; tail -40 "$log" >&2; exit 1; }
}
declare -A dirs=([model]="$MODEL")
names=(model)
for name in $BASELINES; do dirs[$name]="$COMPARISON/$name"; names+=("$name"); done
for name in "${names[@]}"; do complete "${dirs[$name]}" || { echo "no complete scores in ${dirs[$name]}" >&2; exit 1; }; done
mkdir -p "$OUTPUT"

for name in "${names[@]}"; do
    others=()
    for other in "${names[@]}"; do [ "$other" = "$name" ] || others+=("${dirs[$other]}"); done
    ratio=()
    [ "$name" = model ] && ratio=(--ratio-run "$SCORED")
    echo "=== $(date '+%F %T') $name alone against the other ${#others[@]}"
    if [ -s "$OUTPUT/${name}_variabili.log" ] && grep -q "with no variable far" "$OUTPUT/${name}_variabili.log"; then
        echo "variables: already written"
    else
        check "$OUTPUT/${name}_variabili.log" "$PYTHON" "$SCRIPTS/check_era5_variables_away.py" --run-dir "$SCORED" \
            --score-dir "${dirs[$name]}" --by-dirs "${others[@]}" --top-percent "$TOP"
    fi
    if [ -f "$OUTPUT/${name}_bordo_unseen_anni.csv" ]; then
        echo "edge and CAPE: already written"
    else
        check "$OUTPUT/${name}_bordo.log" "$PYTHON" "$SCRIPTS/check_era5_edge_lead.py" --score-dir "${dirs[$name]}" \
            --by-dir "${others[@]}" --top-percent "$TOP" --unseen-dir "$UNSEEN" --csv "$OUTPUT/${name}_bordo.csv" \
            "${ratio[@]}"
    fi
    if [ -f "$OUTPUT/${name}_futuro.csv" ]; then
        echo "rain and clouds: already written"
    else
        check "$OUTPUT/${name}_futuro.log" "$PYTHON" "$SCRIPTS/check_era5_alone_future.py" --score-dir "${dirs[$name]}" \
            --by-dir "${others[@]}" --inputs-run "$SCORED" --prepared-dir "$PREPARED" --top-percent "$TOP" \
            --csv "$OUTPUT/${name}_futuro.csv" "${ratio[@]}"
    fi
done
echo "=== $(date '+%F %T') done: $OUTPUT"
