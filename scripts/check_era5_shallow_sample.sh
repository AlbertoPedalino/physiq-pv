#!/usr/bin/env bash
# Do the flags of the statistical detectors depend on the training sample they are fitted on?
#
# run_era5_comparisons.sh fits every statistical detector on 200000 vectors drawn from 2000 training timestamps
# (seed 0). Here the same detectors are fitted again on the validation years of the same run, once on a sample
# five times larger (1000000 vectors from 5000 timestamps) and once on a sample of the same size drawn with
# another seed; each is expressed against the climatology of its score as in the comparison (day of year +-15 x
# hour, one year out, leaving out the unusual values). For every detector the flags of the three versions, at
# the highest 5% and 1%, are compared two by two with compare_era5_detectors.py: a Jaccard index near 1 means
# that the sample does not matter.
#
#   bash /tmp/ana/scripts/check_era5_shallow_sample.sh outputs/campione_good2 2>&1 | tee campione_good2.log
#
# Run after run_era5_comparisons.sh, whose validation runs and climatologies (COMPARISON) are the first version.
# A step whose run is already complete is skipped. The output of each step is shown and kept in its own log under
# OUTPUT/logs.
set -euo pipefail

OUTPUT=${1:?output root}
SCRIPTS=${SCRIPTS:-$(dirname "$0")}
PYTHON=${PYTHON:-.venv/bin/python}
RUN=${RUN:-outputs/stgan_good_sweep_2_validation}
COMPARISON=${COMPARISON:-outputs/confronti_good2}
MASKS=${MASKS:-outputs/era5_sweep_1999/normal_masks}
METHODS=${METHODS:-"pca cblof mcd hbos copod iforest"}

step() { echo; echo "=== $(date '+%F %T') $*"; }
complete() { [ -f "$1/metadata.json" ] && grep -q '"status": "complete"' "$1/metadata.json"; }

mkdir -p "$OUTPUT/logs"
for name in $METHODS; do
    complete "$COMPARISON/climatology/validation/$name" || { echo "run_era5_comparisons.sh has not scored $name yet" >&2; exit 1; }
done
# version: options of score_era5_shallow.py
declare -A OPTIONS=([grande]="--train-sample 1000000 --train-timestamps 5000 --seed 0"
                    [seme1]="--train-sample 200000 --train-timestamps 2000 --seed 1")
for version in grande seme1; do
    last=${METHODS##* }
    if ! complete "$OUTPUT/$version/shallow/$last"; then
        step "$version: statistical detectors on the validation years (${OPTIONS[$version]})"
        # shellcheck disable=SC2086
        "$PYTHON" "$SCRIPTS/score_era5_shallow.py" --run-dir "$RUN" --output-dir "$OUTPUT/$version/shallow" \
            --methods $METHODS ${OPTIONS[$version]} 2>&1 | tee "$OUTPUT/logs/${version}_shallow.log"
    fi
    for name in $METHODS; do
        complete "$OUTPUT/$version/climatology/$name" && continue
        step "$version: $name, climatology of the validation, one year out"
        "$PYTHON" "$SCRIPTS/run_era5_stgan.py" climatology --run-dir "$OUTPUT/$version/shallow/$name" \
            --output-dir "$OUTPUT/$version/climatology/$name" --score-component combined --reference loyo \
            --normal-masks-dir "$MASKS" --period day --window-days 15 \
            2>&1 | tee "$OUTPUT/logs/climatology_${version}_$name.log"
    done
done

for name in $METHODS; do
    for top in 5 1; do
        step "$name: the three training samples, highest $top%"
        "$PYTHON" "$SCRIPTS/compare_era5_detectors.py" --top-percent "$top" \
            --runs "base=$COMPARISON/climatology/validation/$name" "grande=$OUTPUT/grande/climatology/$name" \
                   "seme1=$OUTPUT/seme1/climatology/$name" 2>&1 | tee "$OUTPUT/logs/${name}_${top}pc.log"
    done
done
