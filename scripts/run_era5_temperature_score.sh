#!/usr/bin/env bash
# The temperature score of the final run against the temperature of the seasonal normal, on the known events.
#
# The reconstruction error of G kept to the temperature variables (FEATURES), every variable on the scale of its
# rarity read on the validation years (rebalance_era5_stgan_scores.py --scale rarity), and the error of the seasonal
# normal, with no model, computed again on the same variables (score_era5_seasonal_normal.py --features) and put on
# the same kind of scale. Both then go through the same steps as the score of the results: climatology of the
# validation one year out, climatology of the test from the validation, threshold of the validation; and both are read on the known events, day by day from three days before.
#
#     bash scripts/run_era5_temperature_score.sh outputs/temperatura_optuna27
#
# A step whose run is already complete is skipped, so the script can be started again after a stop.
set -euo pipefail

OUTPUT=${1:?output root}
SCRIPTS=${SCRIPTS:-$(dirname "$0")}
PYTHON=${PYTHON:-.venv/bin/python}
TEST_RUN=${TEST_RUN:-outputs/stgan_optuna27_scored}
VALIDATION_RUN=${VALIDATION_RUN:-outputs/stgan_optuna27_validation}
MASKS=${MASKS:-outputs/era5_sweep_1999/normal_masks}
EVENTS=${EVENTS:-$SCRIPTS/../docs/era5_casi_noti/lista_ridotta.csv}
FEATURES=${FEATURES:-"2m_temperature temperature"}

step() { echo; echo "=== $(date '+%F %T') $*"; }
complete() { [ -f "$1/metadata.json" ] && grep -q '"status": "complete"' "$1/metadata.json"; }

mkdir -p "$OUTPUT/logs"
for name in model normal; do
    if [ "$name" = model ]; then
        validation=$VALIDATION_RUN; test=$TEST_RUN
    else
        # The error of the seasonal normal, kept to the same variables: the saved one has no errors by variable.
        validation=$OUTPUT/raw/validation/normal; test=$OUTPUT/raw/test/normal
        for period in validation test; do
            target=$([ "$period" = validation ] && echo "$validation" || echo "$test")
            source=$([ "$period" = validation ] && echo "$VALIDATION_RUN" || echo "$TEST_RUN")
            complete "$target" && continue
            rm -rf "$target"
            step "normal, $period: error of the seasonal normal on the temperature variables"
            # shellcheck disable=SC2086
            "$PYTHON" "$SCRIPTS/score_era5_seasonal_normal.py" --run-dir "$source" --output-dir "$target" \
                --features $FEATURES 2>&1 | tee "$OUTPUT/logs/normal_${period}.log"
        done
    fi
    for period in validation test; do
        source=$([ "$period" = validation ] && echo "$validation" || echo "$test")
        complete "$OUTPUT/$period/$name" && continue
        rm -rf "${OUTPUT:?}/$period/$name"
        step "$name, $period: temperature variables, scale of their rarity on the validation"
        # shellcheck disable=SC2086
        "$PYTHON" "$SCRIPTS/rebalance_era5_stgan_scores.py" --run-dir "$source" --output-dir "$OUTPUT/$period/$name" \
            --scale rarity --features $FEATURES --scale-run-dir "$validation" --normal-masks-dir "$MASKS" \
            2>&1 | tee "$OUTPUT/logs/rebalance_${period}_$name.log"
    done
    if ! complete "$OUTPUT/clim/validation/$name"; then
        step "$name: climatology of the validation, one year out"
        "$PYTHON" "$SCRIPTS/run_era5_stgan.py" climatology --run-dir "$OUTPUT/validation/$name" \
            --output-dir "$OUTPUT/clim/validation/$name" --score-component combined --reference loyo \
            --normal-masks-dir "$MASKS" --period day --window-days 15 2>&1 | tee "$OUTPUT/logs/clim_validation_$name.log"
    fi
    if ! complete "$OUTPUT/clim/test/$name"; then
        step "$name: climatology of the test, from the validation"
        "$PYTHON" "$SCRIPTS/run_era5_stgan.py" climatology --run-dir "$OUTPUT/test/$name" \
            --output-dir "$OUTPUT/clim/test/$name" --score-component combined --reference validation \
            --reference-run-dir "$OUTPUT/validation/$name" --normal-masks-dir "$MASKS" --period day --window-days 15 \
            2>&1 | tee "$OUTPUT/logs/clim_test_$name.log"
    fi
    "$PYTHON" "$SCRIPTS/set_era5_threshold_reference.py" --run-dir "$OUTPUT/clim/test/$name" \
        --reference-run-dir "$OUTPUT/clim/validation/$name"
    step "$name: the known events, day by day"
    "$PYTHON" "$SCRIPTS/check_era5_known_events.py" --events "$EVENTS" --days-before 3 \
        --score "temperatura_$name=$OUTPUT/clim/test/$name" --output-dir "$OUTPUT/eventi_noti/$name" \
        2>&1 | tee "$OUTPUT/logs/eventi_noti_$name.log"
done
