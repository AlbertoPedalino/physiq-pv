#!/usr/bin/env bash
# Every check of the lead of the model on the known events and on the edge, at every level of TOPS (the highest 5%
# and 1% of every score, thresholds from the validation years):
#   - run_era5_known_events_lead.sh: onset of the 39 events, box and rings against random dates (ROOT/anticipo<tag>);
#   - run_era5_edge_lead_events.sh: edge of the model alone, fronts and trails, dropout spread and strength
#     (ROOT/bordo<tag>), with the onsets of the first step;
#   - run_era5_direction_formation.sh: births of the baselines and direction of the edge against the wind
#     (ROOT/direzione<tag>).
# The tag is empty at 5% (the folders already written keep their names) and _1pc at 1%. A step whose last output
# is already there is skipped.
#
#   SCRIPTS=/tmp/lead/scripts EVENTS=/tmp/lead/docs/era5_casi_noti/lista_esatta.csv \
#       bash /tmp/lead/scripts/run_era5_lead_all.sh outputs/optuna27 2>&1 | tee outputs/optuna27_anticipo_tutto.log
set -euo pipefail

ROOT=${1:?output root}
SCRIPTS=${SCRIPTS:-$(dirname "$0")}
TOPS=${TOPS:-"5 1"}
export SCRIPTS

for top in $TOPS; do
    tag=$([ "$top" = 5 ] && echo "" || echo "_${top}pc")
    echo; echo "##### $(date '+%F %T') highest $top% (folders *$tag)"
    if [ -f "$ROOT/anticipo$tag/riepilogo.csv" ]; then
        echo "anticipo$tag: already written, skipped"
    else
        TOP=$top bash "$SCRIPTS/run_era5_known_events_lead.sh" "$ROOT/anticipo$tag"
    fi
    if [ -f "$ROOT/bordo$tag/bordo_eventi_forza.csv" ]; then
        echo "bordo$tag: already written, skipped"
    else
        TOP=$top ONSETS="$ROOT/anticipo$tag/inizi.csv" bash "$SCRIPTS/run_era5_edge_lead_events.sh" "$ROOT/bordo$tag"
    fi
    if [ -f "$ROOT/direzione$tag/direzione_$top.csv" ]; then
        echo "direzione$tag: already written, skipped"
    else
        TOP=$top bash "$SCRIPTS/run_era5_direction_formation.sh" "$ROOT/direzione$tag"
    fi
done
echo; echo "##### $(date '+%F %T') done"
