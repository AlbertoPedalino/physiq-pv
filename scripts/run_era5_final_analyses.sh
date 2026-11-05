#!/usr/bin/env bash
# The analyses of the results chapter that read the final run, after run_era5_score_behaviour.sh (the validation
# years, BEHAVIOUR) and the climatology of G over the test years (MODEL, with the threshold of the validation):
#   - the maps of G, D and D(real) at the typical timestamp and at the one with the highest mean G (validation);
#   - the thirds of the flags of G by D(real) (validation);
#   - the confirmation on the test years: correlation of G and D and of the terms of D, the scores in the zones of
#     the events of G+clim at 5% and 1%, the smoothness of the reconstructions, where G+clim flags;
#   - the events of G+clim on the test years: zones and types, profile around the events, the largest event of
#     every type (5% and 1%) and the example of every type (the largest lasting at most three days);
#   - the spread of the Monte Carlo dropout of G: by score band, on the flags, by class and by type, with the maps
#     at the timestamps of the examples.
# Every check reads the threshold of its score from the validation years. The baselines (COMPARISON) are those of
# run_era5_comparisons.sh, which do not depend on the model.
#
#   bash /tmp/o27/scripts/run_era5_final_analyses.sh outputs/finale_optuna27 2>&1 | tee outputs/finale_optuna27.log
#
# Run from the checkout, with SCRIPTS pointing at a copy of the scripts when the checkout must not change. A
# climatology already complete is skipped; the checks are run again at every start. The full output of each step is
# in its own log under OUTPUT/logs.
set -euo pipefail

OUTPUT=${1:?output root}
SCRIPTS=${SCRIPTS:-$(dirname "$0")}
PYTHON=${PYTHON:-.venv/bin/python}
TEST_RUN=${TEST_RUN:-outputs/stgan_optuna27_scored}
VALIDATION_RUN=${VALIDATION_RUN:-outputs/stgan_optuna27_validation}
CHECKPOINT=${CHECKPOINT:-outputs/optuna/era5-gan/trial_27/era5/usk5o8rc/results/model_epoch_6.pt}
BEHAVIOUR=${BEHAVIOUR:-outputs/comportamento_optuna27}
MODEL=${MODEL:-outputs/optuna27/test_generator_clim_day}
COMPARISON=${COMPARISON:-outputs/confronti_good2/climatology}
MASKS=${MASKS:-outputs/era5_sweep_1999/normal_masks}

step() { echo; echo "=== $(date '+%F %T') $*"; }
complete() { [ -f "$1/metadata.json" ] && grep -q '"status": "complete"' "$1/metadata.json"; }
log() { tee "$OUTPUT/logs/$1.log"; }

for run in "$BEHAVIOUR/generator_clim" "$BEHAVIOUR/discriminator_clim" "$MODEL" "$COMPARISON/test/normal" \
           "$COMPARISON/test/change_3h" "$COMPARISON/validation/normal" "$COMPARISON/validation/change_3h"; do
    complete "$run" || { echo "missing: $run" >&2; exit 1; }
done
grep -q '"threshold_reference"' "$MODEL/metadata.json" \
    || { echo "$MODEL has no threshold reference: set_era5_threshold_reference.py first" >&2; exit 1; }
mkdir -p "$OUTPUT/logs"
S=$TEST_RUN
E=$MODEL
C=$COMPARISON/test
DCLIM=$OUTPUT/test_discriminator_clim_day

step "validation: maps at two timestamps"
"$PYTHON" "$SCRIPTS/save_era5_instant_maps.py" --run-dir "$VALIDATION_RUN" \
    --correlation "$BEHAVIOUR/correlazioni_g_d/spatial_correlation.csv" --output "$BEHAVIOUR/mappe_istanti.npz" 2>&1 \
    | log mappe_istanti

step "validation: thirds of the flags of G by D(real)"
"$PYTHON" "$SCRIPTS/check_era5_discriminator_confidence.py" --g "$BEHAVIOUR/generator_clim" \
    --judge "D_reale=$VALIDATION_RUN@real" \
    --refs "normale=$COMPARISON/validation/normal" "3ore=$COMPARISON/validation/change_3h" 2>&1 | log fiducia_d_val

if ! complete "$DCLIM"; then
    step "test: climatology of D, from the validation"
    "$PYTHON" "$SCRIPTS/run_era5_stgan.py" climatology --run-dir "$S" --output-dir "$DCLIM" \
        --score-component discriminator --reference validation --reference-run-dir "$VALIDATION_RUN" \
        --normal-masks-dir "$MASKS" --period day --window-days 15 > "$OUTPUT/logs/climatology_test_discriminator.log" 2>&1
fi
"$PYTHON" "$SCRIPTS/set_era5_threshold_reference.py" --run-dir "$DCLIM" --reference-run-dir "$BEHAVIOUR/discriminator_clim"

step "test: correlation of G and D, with and without the climatology"
"$PYTHON" "$SCRIPTS/check_era5_score_correlation.py" --output-dir "$OUTPUT/correlazioni_g_d" \
    --scores "G=$S@generator" "D=$S@discriminator" "G+clim=$E" "D+clim=$DCLIM" 2>&1 | log correlazioni_g_d
step "test: correlation of G with the two outputs of the discriminator"
"$PYTHON" "$SCRIPTS/check_era5_score_correlation.py" --output-dir "$OUTPUT/correlazioni_g_dterms" \
    --scores "G=$S@generator" "Dreale=$S@real" "Dricostruito=$S@generated" "D=$S@discriminator" 2>&1 \
    | log correlazioni_g_dterms

for top in 5 1; do
    step "test: scores in the zones of the events of G+clim, highest $top%"
    "$PYTHON" "$SCRIPTS/check_era5_event_zones.py" --events "G+clim=$E" \
        --scores "G+clim=$E" "G=$S@generator" "Dreale=$S@real" "Dricostruito=$S@generated" "D=$S@discriminator" \
        --inputs-run "$S" --top-percent "$top" --csv "$OUTPUT/zone_g_dterms_${top}pc.csv" 2>&1 | log "zone_g_dterms_${top}pc"
    step "test: where G+clim flags, highest $top%"
    "$PYTHON" "$SCRIPTS/check_era5_score_by_place.py" --run-dir "$E" --score-component combined --top-percent "$top" 2>&1 \
        | log "luogo_g_clim_${top}pc"
done

step "test: does D tell the reconstructions by their smoothness?"
"$PYTHON" "$SCRIPTS/check_era5_discriminator_smoothness.py" --run-dir "$S" --checkpoint "$CHECKPOINT" --samples 240 \
    --csv "$OUTPUT/levigatezza_d.csv" 2>&1 | log levigatezza_d

step "test: the events of G+clim, zones and types"
"$PYTHON" "$SCRIPTS/check_era5_event_zones.py" --events "G+clim=$E" --scores "G+clim=$E" "G=$S@generator" \
    "normale=$C/normal" --inputs-run "$S" --csv "$OUTPUT/zone_5pc.csv" 2>&1 | log zone_5pc
"$PYTHON" "$SCRIPTS/check_era5_event_zones.py" --events "G+clim=$E" --scores "G+clim=$E" "normale=$C/normal" \
    --inputs-run "$S" --csv "$OUTPUT/zone_tipi_5pc.csv" 2>&1 | log zone_tipi_5pc
step "test: the profile of the scores around the events"
"$PYTHON" "$SCRIPTS/check_era5_event_profile.py" --events "G+clim=$E" --scores "G+clim=$E" "G=$S@generator" \
    "normale=$C/normal" --inputs-run "$S" --csv "$OUTPUT/profilo_5pc.csv" 2>&1 | log profilo_5pc

step "test: the largest event of every type, highest 5% and 1%"
"$PYTHON" "$SCRIPTS/check_era5_event_cases.py" --events "G+clim=$E" --scores "G+clim=$E" "G=$S@generator" \
    "normale=$C/normal" --inputs-run "$S" --longest-days 0 --output-dir "$OUTPUT/casi_5pc" 2>&1 | log casi_5pc
"$PYTHON" "$SCRIPTS/check_era5_event_cases.py" --events "G+clim=$E" --scores "G+clim=$E" --inputs-run "$S" \
    --longest-days 0 --top-percent 1 --output-dir "$OUTPUT/casi_1pc" 2>&1 | log casi_1pc
step "test: the example of every type, the largest event lasting at most three days"
"$PYTHON" "$SCRIPTS/check_era5_event_cases.py" --events "G+clim=$E" --scores "G+clim=$E" "G=$S@generator" \
    "normale=$C/normal" --inputs-run "$S" --longest-days 3 --output-dir "$OUTPUT/esempi" 2>&1 | log esempi

step "test: the spread of the dropout of G"
"$PYTHON" "$SCRIPTS/check_era5_dropout_spread.py" --run-dir "$S" --score-component generator 2>&1 \
    | log dropout_rapporto_bande
"$PYTHON" "$SCRIPTS/check_era5_dropout_flags.py" --events "G+clim=$E" --spreads "grezza=$S/scores/generator_std.npy" \
    --normal "normale=$C/normal" --inputs-run "$S" --csv "$OUTPUT/dropout_5pc.csv" 2>&1 | log dropout_5pc
"$PYTHON" "$SCRIPTS/check_era5_dropout_flags.py" --events "G+clim=$E" --spreads "grezza=$S/scores/generator_std.npy" \
    --normal "normale=$C/normal" --raw-error "G=$S@generator" --ratio-of grezza --inputs-run "$S" \
    --csv "$OUTPUT/dropout_rapporto_5pc.csv" 2>&1 | log dropout_rapporto_5pc
mapfile -t instants < <("$PYTHON" -c "import sys, pandas; print('\n'.join(pandas.read_csv(sys.argv[1])['timestamp']))" \
    "$OUTPUT/esempi/cases.csv")
"$PYTHON" "$SCRIPTS/check_era5_dropout_flags.py" --events "G+clim=$E" --spreads "grezza=$S/scores/generator_std.npy" \
    --normal "normale=$C/normal" --raw-error "G=$S@generator" --ratio-of grezza --course "3ore=$C/change_3h" \
    --inputs-run "$S" --instants "${instants[@]}" --csv "$OUTPUT/dropout_tipi_5pc.csv" 2>&1 | log dropout_tipi_5pc

step "done"
