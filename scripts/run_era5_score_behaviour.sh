#!/usr/bin/env bash
# How the components of the STGAN score behave over the validation years, where the score of the detector
# is chosen: the climatology of G and of D (day of year +-15 x hour, one year out, leaving out the unusual
# values), the spatial and temporal correlation of G and D and of the two outputs of the discriminator, the
# scores in the core, at the edge and away from the events of G+clim, where G flags with and without the
# climatology, and whether D tells the reconstructions by their smoothness. Every check at the 95th and,
# where the threshold matters, at the 99th percentile of the same score over the validation years.
#
#   bash /tmp/ana/scripts/run_era5_score_behaviour.sh outputs/comportamento_good2 2>&1 | tee comportamento_good2.log
#
# Run from the checkout, with SCRIPTS pointing at a copy of the scripts when the checkout must not change.
# For another run, set RUN (its scoring over the validation years with the real history) and CHECKPOINT.
# A climatology already complete is skipped; the checks are run again at every start. The full output of
# each step is in its own log under OUTPUT/logs.
set -euo pipefail

OUTPUT=${1:?output root}
SCRIPTS=${SCRIPTS:-$(dirname "$0")}
PYTHON=${PYTHON:-.venv/bin/python}
RUN=${RUN:-outputs/stgan_good_sweep_2_validation}
CHECKPOINT=${CHECKPOINT:-outputs/stgan_wandb/era5/5qlj2lwg/results/model_epoch_6.pt}
MASKS=${MASKS:-outputs/era5_sweep_1999/normal_masks}

step() { echo; echo "=== $(date '+%F %T') $*"; }
complete() { [ -f "$1/metadata.json" ] && grep -q '"status": "complete"' "$1/metadata.json"; }

mkdir -p "$OUTPUT/logs"
for component in generator discriminator; do
    complete "$OUTPUT/${component}_clim" && continue
    step "$component: climatology of the validation, one year out"
    "$PYTHON" "$SCRIPTS/run_era5_stgan.py" climatology --run-dir "$RUN" --output-dir "$OUTPUT/${component}_clim" \
        --score-component "$component" --reference loyo --normal-masks-dir "$MASKS" --period day --window-days 15 \
        > "$OUTPUT/logs/climatology_$component.log" 2>&1
done
G=$RUN@generator
D=$RUN@discriminator
GCLIM=$OUTPUT/generator_clim
DCLIM=$OUTPUT/discriminator_clim

step "correlation of G and D, with and without the climatology"
"$PYTHON" "$SCRIPTS/check_era5_score_correlation.py" --output-dir "$OUTPUT/correlazioni_g_d" \
    --scores "G=$G" "D=$D" "G+clim=$GCLIM" "D+clim=$DCLIM" 2>&1 | tee "$OUTPUT/logs/correlazioni_g_d.log"

step "correlation of G with the two outputs of the discriminator"
"$PYTHON" "$SCRIPTS/check_era5_score_correlation.py" --output-dir "$OUTPUT/correlazioni_g_dterms" \
    --scores "G=$G" "D_reale=$RUN@real" "D_generato=$RUN@generated" "D=$D" 2>&1 \
    | tee "$OUTPUT/logs/correlazioni_g_dterms.log"

for top in 5 1; do
    step "scores in the zones of the events of G+clim, highest $top%"
    "$PYTHON" "$SCRIPTS/check_era5_event_zones.py" --events "G+clim=$GCLIM" \
        --scores "G+clim=$GCLIM" "G=$G" "D_reale=$RUN@real" "D_generato=$RUN@generated" "D=$D" \
        --inputs-run "$RUN" --top-percent "$top" --csv "$OUTPUT/zone_g_dterms_${top}pc.csv" 2>&1 \
        | tee "$OUTPUT/logs/zone_g_dterms_${top}pc.log"
    step "where G flags, without and with the climatology, highest $top%"
    "$PYTHON" "$SCRIPTS/check_era5_score_by_place.py" --run-dir "$RUN" --score-component generator \
        --top-percent "$top" 2>&1 | tee "$OUTPUT/logs/luogo_g_grezzo_${top}pc.log"
    "$PYTHON" "$SCRIPTS/check_era5_score_by_place.py" --run-dir "$GCLIM" --score-component combined \
        --top-percent "$top" 2>&1 | tee "$OUTPUT/logs/luogo_g_clim_${top}pc.log"
done

step "does D tell the reconstructions by their smoothness?"
"$PYTHON" "$SCRIPTS/check_era5_discriminator_smoothness.py" --run-dir "$RUN" --checkpoint "$CHECKPOINT" \
    --samples 240 --csv "$OUTPUT/levigatezza_d.csv" 2>&1 | tee "$OUTPUT/logs/levigatezza_d.log"
