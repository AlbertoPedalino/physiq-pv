#!/usr/bin/env bash
# The Saharan dust events where they arrive, where they come from and on their way (docs/era5_casi_noti/
# polvere_sahara.csv, rule in make_polvere_sahara.py): the share each score flags in every region, with
# check_era5_known_events.py, one run per score, then one table (rows: region, columns: score; by chance 5%).
#
#   SCRIPTS=/tmp/lead/scripts EVENTS=/tmp/lead/docs/era5_casi_noti/polvere_sahara.csv \
#       bash /tmp/lead/scripts/run_era5_dust_regions.sh outputs/optuna27/polvere_sahara 2>&1 | tee outputs/optuna27_polvere_sahara.log
set -euo pipefail

OUTPUT=${1:?output folder}
SCRIPTS=${SCRIPTS:-$(dirname "$0")}
PYTHON=${PYTHON:-.venv/bin/python}
EVENTS=${EVENTS:-docs/era5_casi_noti/polvere_sahara.csv}
C=${COMPARISON:-outputs/confronti_good2/climatology/test}
SCORES=${SCORES:-"G_n=outputs/optuna27/test_generator_clim_day P_n=outputs/optuna27/previsione/climatology/test/P
normale=$C/normal variazione_3h=$C/change_3h pca=$C/pca cblof=$C/cblof mcd=$C/mcd hbos=$C/hbos copod=$C/copod
iforest=$C/iforest"}
mkdir -p "$OUTPUT"
for score in $SCORES; do
    name=${score%%=*}
    echo "=== $(date '+%F %T') $name"
    "$PYTHON" "$SCRIPTS/check_era5_known_events.py" --events "$EVENTS" --score "$score" \
        --output-dir "$OUTPUT/$name" > "$OUTPUT/$name.log" 2>&1 || { echo "FAILED $name:"; tail -20 "$OUTPUT/$name.log"; exit 1; }
done
"$PYTHON" - "$OUTPUT" $SCORES <<'EOF'
import sys
from pathlib import Path
import pandas as pd
root, names = Path(sys.argv[1]), [item.split("=", 1)[0] for item in sys.argv[2:]]
tables = {name: pd.read_csv(root / name / "eventi_noti.csv").set_index("id") for name in names}
column = "segnalati %"
wide = pd.DataFrame({name: table[column] for name, table in tables.items()})
print(f"\n% of the values of each region flagged by each score (its highest 5%; by chance 5%)")
pd.set_option("display.width", 200)
print(wide.round(0).to_string())
others = [name for name in names if name not in ("G_n", "P_n")]
print("\nregions above 10% (twice chance), by score:")
print((wide > 10).sum().to_string())
wide.to_csv(root / "polvere_sahara_tutti.csv")
EOF
echo "=== $(date '+%F %T') done: $OUTPUT"
