#!/usr/bin/env bash
# Does the model see the end of an anomaly first? check_era5_end_lead.py four times over the test years:
#   G_n against the 8 baselines (with the thirds of its dropout spread), P_n against the 8 baselines,
#   G_n against P_n (with the thirds), P_n against G_n; and the reverse, the 8 baselines against G_n and P_n.
# Judges: the other score stops; the input variables go back within 2 sd; CAPE and gust (read by no score) fall;
# the same at equal strength; hours until the end, up to HORIZON. Logs and CSVs under OUTPUT; a step whose CSV is already written is skipped.
#
#   SCRIPTS=/tmp/fine/scripts bash /tmp/fine/scripts/run_era5_end_lead.sh outputs/optuna27/fine 2>&1 \
#       | tee outputs/optuna27_fine.log
set -euo pipefail

OUTPUT=${1:?output folder}
SCRIPTS=${SCRIPTS:-$(dirname "$0")}
PYTHON=${PYTHON:-.venv/bin/python}
TOP=${TOP:-5}
MODEL=${MODEL:-outputs/optuna27/test_generator_clim_day}
FORECAST=${FORECAST:-outputs/optuna27/previsione/climatology/test/P}
TEST_RUN=${TEST_RUN:-outputs/stgan_optuna27_scored}
# Folders of judges no score reads: CAPE and gust, and those of download_era5.py --judge-variables when there.
UNSEEN=${UNSEEN:-"data/era5_independent/independent data/era5_judges/judges"}
PREPARED=${PREPARED:-data/era5_prepared}  # the radiation the model reads, for the clearness of the sky
COMPARISON=${COMPARISON:-outputs/confronti_good2/climatology/test}
BASELINES=${BASELINES:-"normal change_3h pca cblof mcd hbos copod iforest"}

complete() { [ -f "$1/metadata.json" ] && grep -q '"status": "complete"' "$1/metadata.json"; }
for run in "$MODEL" "$FORECAST" "$TEST_RUN"; do complete "$run" || { echo "no complete run in $run" >&2; exit 1; }; done
by=()
for name in $BASELINES; do complete "$COMPARISON/$name" || { echo "no scores of $name" >&2; exit 1; }; by+=("$COMPARISON/$name"); done
unseen=()
for folder in $UNSEEN; do [ -d "$folder" ] && unseen+=("$folder") || echo "no judges in $folder: left out" >&2; done
[ ${#unseen[@]} -gt 0 ] || { echo "no judges at all" >&2; exit 1; }
mkdir -p "$OUTPUT"
HORIZON=${HORIZON:-24}  # hours until the end, step by step
common=(--top-percent "$TOP" --inputs-run "$TEST_RUN" --unseen-dir "${unseen[@]}" --prepared-dir "$PREPARED"
        --horizon-hours "$HORIZON")

step() {  # name, the runs of the first, --by, the runs of the second, then -- and extra options
    local name=$1; shift
    local first=() second=() extra=()
    while [ $# -gt 0 ] && [ "$1" != "--by" ]; do first+=("$1"); shift; done
    shift
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do second+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    extra=("$@")
    if [ -f "$OUTPUT/fine_$name.csv" ]; then echo "=== $name: already written"; return; fi
    echo; echo "=== $(date '+%F %T') $name"
    "$PYTHON" "$SCRIPTS/check_era5_end_lead.py" --score-dir "${first[@]}" --by-dir "${second[@]}" "${common[@]}"         ${extra[@]+"${extra[@]}"} --csv "$OUTPUT/fine_$name.csv" > "$OUTPUT/fine_$name.log" 2>&1
    grep -v '\[end-lead\] month' "$OUTPUT/fine_$name.log"
}

# The model stops while the others go on
step G_vs_8 "$MODEL" --by "${by[@]}" -- --ratio-run "$TEST_RUN"
step P_vs_8 "$FORECAST" --by "${by[@]}"
step G_vs_P "$MODEL" --by "$FORECAST" -- --ratio-run "$TEST_RUN"
step P_vs_G "$FORECAST" --by "$MODEL"
# The reverse: the 8 baselines (any of them) stop while the model goes on
step 8_vs_G "${by[@]}" --by "$MODEL"
step 8_vs_P "${by[@]}" --by "$FORECAST"
echo "=== $(date '+%F %T') done: $OUTPUT"
