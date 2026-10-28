#!/usr/bin/env bash
# The detectors the model is compared with, treated as the model is: scored on the validation and on the
# test years, expressed against the climatology of their own score (day of year +-15 x hour, estimated on
# the validation years leaving out the unusual values, one year out on the validation itself), flagged
# above the 95th percentile of the same score over the validation years; then compared with the model.
#
#   bash /tmp/ana/scripts/run_era5_comparisons.sh outputs/confronti_good2 2>&1 | tee confronti_good2.log
#
# Run from the checkout, with SCRIPTS pointing at a copy of the scripts when the checkout must not change.
# Detectors: the seven statistical ones of score_era5_shallow.py, the four tests of
# score_era5_spatial_outliers.py, the error of the seasonal normal and of the value three hours before
# (score_era5_seasonal_normal.py). A step whose run is already complete is skipped, so the script can be
# started again after a stop; a directory left incomplete has to be removed by hand first. The full output
# of each step is in its own log under OUTPUT/logs.
set -euo pipefail

OUTPUT=${1:?output root}
SCRIPTS=${SCRIPTS:-$(dirname "$0")}
PYTHON=${PYTHON:-.venv/bin/python}
TEST_RUN=${TEST_RUN:-outputs/stgan_good_sweep_2_scored}
VALIDATION_RUN=${VALIDATION_RUN:-outputs/stgan_good_sweep_2_validation}
MASKS=${MASKS:-outputs/era5_sweep_1999/normal_masks}
MODEL=${MODEL:-outputs/good2v/test_generator_clim_day}
WORKERS=${WORKERS:-16}
SHALLOW="pca kmeans cblof mcd hbos copod iforest"
SPATIAL="shekhar_spatial shekhar_temporal shekhar_spatiotemporal chen_multivariate"
SIMPLE="normal change_3h"

step() { echo; echo "=== $(date '+%F %T') $*"; }
complete() { [ -f "$1/metadata.json" ] && grep -q '"status": "complete"' "$1/metadata.json"; }
run_of() {  # the raw run of a detector over a period
    case " $SHALLOW " in *" $2 "*) echo "$OUTPUT/$1/shallow/$2"; return;; esac
    case " $SPATIAL " in *" $2 "*) echo "$OUTPUT/$1/spatial/$2"; return;; esac
    echo "$OUTPUT/$1/simple/$2"
}

mkdir -p "$OUTPUT/logs"
for period in validation test; do
    source=$([ "$period" = validation ] && echo "$VALIDATION_RUN" || echo "$TEST_RUN")
    if ! complete "$OUTPUT/$period/shallow/iforest"; then
        step "$period: statistical detectors"
        "$PYTHON" "$SCRIPTS/score_era5_shallow.py" --run-dir "$source" --output-dir "$OUTPUT/$period/shallow" \
            > "$OUTPUT/logs/${period}_shallow.log" 2>&1
        tail -n 3 "$OUTPUT/logs/${period}_shallow.log"
    fi
    if ! complete "$OUTPUT/$period/spatial/chen_multivariate"; then
        step "$period: spatial outlier tests"
        "$PYTHON" "$SCRIPTS/score_era5_spatial_outliers.py" --run-dir "$source" --output-dir "$OUTPUT/$period/spatial" \
            --workers "$WORKERS" > "$OUTPUT/logs/${period}_spatial.log" 2>&1
        tail -n 3 "$OUTPUT/logs/${period}_spatial.log"
    fi
    for name in $SIMPLE; do
        complete "$OUTPUT/$period/simple/$name" && continue
        step "$period: $name"
        previous=$([ "$name" = change_3h ] && echo 1 || echo 0)
        "$PYTHON" "$SCRIPTS/score_era5_seasonal_normal.py" --run-dir "$source" \
            --output-dir "$OUTPUT/$period/simple/$name" --previous-value "$previous" \
            > "$OUTPUT/logs/${period}_${name}.log" 2>&1
    done
done

for name in $SHALLOW $SPATIAL $SIMPLE; do
    validation=$(run_of validation "$name")
    test=$(run_of test "$name")
    if ! complete "$OUTPUT/climatology/validation/$name"; then
        step "$name: climatology of the validation, one year out"
        "$PYTHON" "$SCRIPTS/run_era5_stgan.py" climatology --run-dir "$validation" \
            --output-dir "$OUTPUT/climatology/validation/$name" --score-component combined --reference loyo \
            --normal-masks-dir "$MASKS" --period day --window-days 15 \
            > "$OUTPUT/logs/climatology_validation_$name.log" 2>&1
    fi
    if ! complete "$OUTPUT/climatology/test/$name"; then
        step "$name: climatology of the test, from the validation"
        "$PYTHON" "$SCRIPTS/run_era5_stgan.py" climatology --run-dir "$test" \
            --output-dir "$OUTPUT/climatology/test/$name" --score-component combined --reference validation \
            --reference-run-dir "$validation" --normal-masks-dir "$MASKS" --period day --window-days 15 \
            > "$OUTPUT/logs/climatology_test_$name.log" 2>&1
    fi
    "$PYTHON" "$SCRIPTS/set_era5_threshold_reference.py" --run-dir "$OUTPUT/climatology/test/$name" \
        --reference-run-dir "$OUTPUT/climatology/validation/$name"
done

step "comparison with the model"
runs=("G+clim=$MODEL")
for name in $SHALLOW $SPATIAL $SIMPLE; do runs+=("$name=$OUTPUT/climatology/test/$name"); done
vote=$OUTPUT/vote
[ -e "$vote" ] && vote=$OUTPUT/vote_$(date +%Y%m%d_%H%M%S)  # a new one at every comparison, nothing removed
"$PYTHON" "$SCRIPTS/compare_era5_detectors.py" --runs "${runs[@]}" --vote-dir "$vote" \
    --figure "$OUTPUT/jaccard.png" 2>&1 | tee "$OUTPUT/logs/comparison.log"
