#!/usr/bin/env bash
# The anomaly the generator forecasts (P): how far from the seasonal normal the field it forecasts for the next
# timestamp is, known one step before (score_era5_forecast_anomaly.py, dropout off), and the same measure on the
# observed field (persistence, "the next field as this one"). Both over the validation and the test years, then
# expressed against their own climatology (day of year +-15 x hour, from the validation, as G_n and the baselines),
# with the threshold of the validation. Then the edge lead, head to head on the 39 known events and over the test
# years: P against MCD, the two statistical methods (distance from normal, change in 3 h), the persistence and G_n,
# in both directions.
#
#   SCRIPTS=/tmp/lead/scripts EVENTS=/tmp/lead/docs/era5_casi_noti/lista_esatta.csv \
#       bash /tmp/lead/scripts/run_era5_forecast_score.sh outputs/optuna27/previsione 2>&1 | tee outputs/optuna27_previsione.log
#
# Steps already complete are skipped. The forecast needs the GPU (DEVICE=cuda).
set -euo pipefail

ROOT=${1:?output root}
SCRIPTS=${SCRIPTS:-$(dirname "$0")}
PYTHON=${PYTHON:-.venv/bin/python}
RUN=${RUN:-outputs/stgan_optuna27_scored}
CHECKPOINT=${CHECKPOINT:-outputs/optuna/era5-gan/trial_27/era5/usk5o8rc/results/model_epoch_6.pt}
PREPARED=${PREPARED:-data/era5_prepared}
MASKS=${MASKS:-outputs/era5_sweep_1999/normal_masks}
MODEL=${MODEL:-outputs/optuna27/test_generator_clim_day}
BASELINE_ROOT=${BASELINE_ROOT:-outputs/confronti_good2/climatology/test}
DEVICE=${DEVICE:-cuda}
MC_SAMPLES=${MC_SAMPLES:-0}  # > 0: the forecast drawn this many times with the dropouts on (mean and spread of P)
MC_BATCH=${MC_BATCH:-10}  # draws of a timestamp in one batch (lower it if the GPU runs out of memory)
export SCRIPTS

complete() { [ -f "$1/metadata.json" ] && grep -q '"status": "complete"' "$1/metadata.json"; }
step() { echo; echo "=== $(date '+%F %T') $*"; }
[ -f "$CHECKPOINT" ] || { echo "no checkpoint $CHECKPOINT" >&2; exit 1; }
complete "$RUN" || { echo "no complete run $RUN" >&2; exit 1; }
mkdir -p "$ROOT"

for period in validation test; do
    if complete "$ROOT/$period/P" && complete "$ROOT/$period/persistenza"; then
        echo "$period: forecast already written"
    else
        rm -rf "$ROOT/$period/P" "$ROOT/$period/persistenza"
        step "$period: the forecast anomaly P and the persistence"
        "$PYTHON" "$SCRIPTS/score_era5_forecast_anomaly.py" --run-dir "$RUN" --checkpoint "$CHECKPOINT" \
            --prepared-dir "$PREPARED" --period "$period" --output-dir "$ROOT/$period/P" \
            --persistence-dir "$ROOT/$period/persistenza" --device "$DEVICE" --mc-samples "$MC_SAMPLES" \
            --mc-batch "$MC_BATCH"
    fi
done

for name in P persistenza; do
    if ! complete "$ROOT/climatology/validation/$name"; then
        step "$name: climatology of the validation, one year out"
        "$PYTHON" "$SCRIPTS/run_era5_stgan.py" climatology --run-dir "$ROOT/validation/$name" \
            --output-dir "$ROOT/climatology/validation/$name" --score-component combined --reference loyo \
            --normal-masks-dir "$MASKS" --period day --window-days 15
    fi
    if ! complete "$ROOT/climatology/test/$name"; then
        step "$name: climatology of the test, from the validation"
        "$PYTHON" "$SCRIPTS/run_era5_stgan.py" climatology --run-dir "$ROOT/test/$name" \
            --output-dir "$ROOT/climatology/test/$name" --score-component combined --reference validation \
            --reference-run-dir "$ROOT/validation/$name" --normal-masks-dir "$MASKS" --period day --window-days 15
    fi
    "$PYTHON" "$SCRIPTS/set_era5_threshold_reference.py" --run-dir "$ROOT/climatology/test/$name" \
        --reference-run-dir "$ROOT/climatology/validation/$name"
done

P=$ROOT/climatology/test/P
PERSISTENCE=$ROOT/climatology/test/persistenza
# first score | folder of the second | names of the second | tag
comparisons=(
    "$P|$BASELINE_ROOT|mcd|P_vs_mcd"
    "$BASELINE_ROOT/mcd|$ROOT/climatology/test|P|mcd_vs_P"
    "$P|$BASELINE_ROOT|normal change_3h|P_vs_statistici"
    "$P|$ROOT/climatology/test|persistenza|P_vs_persistenza"
    "$PERSISTENCE|$ROOT/climatology/test|P|persistenza_vs_P"
    "$P|$(dirname "$MODEL")|$(basename "$MODEL")|P_vs_Gn"
    "$MODEL|$ROOT/climatology/test|P|Gn_vs_P"
)
for row in "${comparisons[@]}"; do
    IFS='|' read -r first folder names tag <<< "$row"
    if [ -f "$ROOT/bordo_$tag/bordo_eventi_per_evento.csv" ]; then
        echo "bordo_$tag: already written"
        continue
    fi
    step "edge lead: $tag"
    spread_run=""  # with the draws, P first is read with the spread of its draws (dropout thirds, strength)
    [ "$MC_SAMPLES" -gt 0 ] && [ "$first" = "$P" ] && spread_run=$ROOT/test/P
    MODEL=$first COMPARISON=$folder BASELINES=$names TEST_RUN=$spread_run \
        bash "$SCRIPTS/run_era5_edge_lead_events.sh" "$ROOT/bordo_$tag" > "$ROOT/bordo_$tag.log" 2>&1 \
        || { tail -30 "$ROOT/bordo_$tag.log"; exit 1; }
done

step "summary: distance 1 on the 39 events, and the events with a positive excess"
for row in "${comparisons[@]}"; do
    IFS='|' read -r _ _ _ tag <<< "$row"
    echo "##### $tag"
    grep -A4 "^distance, window" "$ROOT/bordo_$tag.log"
    grep -A6 "^event by event" "$ROOT/bordo_$tag.log"
done
step "done: $ROOT"
