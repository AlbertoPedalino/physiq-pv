#!/usr/bin/env bash
# Does the discriminator, with its sign turned, flag something of its own?
#
# Over the validation years G and D go opposite ways. Here the discriminator component after the climatology of
# its score is read with its sign turned (-D), and its flags, at the highest 5% and 1% of the same years, are
# compared with those of G after the climatology, of D as it is, and of the two references with no model (the
# error of the seasonal normal and of the value three hours before, after the climatology of their score): which
# share of them it has in common with each, and how many no other score flags. No label is used.
#
#   bash /tmp/ana/scripts/check_era5_negated_discriminator.sh outputs/segno_d_good2 2>&1 | tee segno_d_good2.log
#
# BEHAVIOUR is the output of run_era5_score_behaviour.sh for a run (its generator_clim and discriminator_clim);
# COMPARISON that of run_era5_comparisons.sh, whose validation scores of the two references are read. A step
# whose run is already complete is skipped.
set -euo pipefail

OUTPUT=${1:?output root}
SCRIPTS=${SCRIPTS:-$(dirname "$0")}
PYTHON=${PYTHON:-.venv/bin/python}
BEHAVIOUR=${BEHAVIOUR:-outputs/comportamento_good2}
COMPARISON=${COMPARISON:-outputs/confronti_good2}
MASKS=${MASKS:-outputs/era5_sweep_1999/normal_masks}

step() { echo; echo "=== $(date '+%F %T') $*"; }
complete() { [ -f "$1/metadata.json" ] && grep -q '"status": "complete"' "$1/metadata.json"; }

mkdir -p "$OUTPUT/logs"
for name in generator_clim discriminator_clim; do
    complete "$BEHAVIOUR/$name" || { echo "no $BEHAVIOUR/$name: run run_era5_score_behaviour.sh first" >&2; exit 1; }
done
for name in normal change_3h; do
    complete "$COMPARISON/validation/simple/$name" || { echo "run_era5_comparisons.sh has not scored $name yet" >&2; exit 1; }
    if ! complete "$OUTPUT/$name"; then
        step "$name: climatology of the validation, one year out"
        "$PYTHON" "$SCRIPTS/run_era5_stgan.py" climatology --run-dir "$COMPARISON/validation/simple/$name" \
            --output-dir "$OUTPUT/$name" --score-component combined --reference loyo --normal-masks-dir "$MASKS" \
            --period day --window-days 15 2>&1 | tee "$OUTPUT/logs/climatology_$name.log"
    fi
done

if ! complete "$OUTPUT/minus_d"; then
    step "the discriminator component after its climatology, with its sign turned"
    "$PYTHON" -c "
import json, shutil, sys
from pathlib import Path
import numpy as np
source, target = Path(sys.argv[1]), Path(sys.argv[2])
metadata = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
(target / 'scores').mkdir(parents=True)
scores = np.load(source / metadata['anomaly_mean_file'], mmap_mode='r')
turned = np.lib.format.open_memmap(target / 'scores' / 'anomaly_mean.npy', mode='w+', dtype=np.float32, shape=scores.shape)
for start in range(0, len(scores), 256):
    turned[start:start + 256] = -np.asarray(scores[start:start + 256])
turned.flush()
shutil.copy(source / 'test_timestamps.npy', target / 'test_timestamps.npy')
metadata.update(score_kind='discriminator_climatology_sign_turned', source_run_dir=str(source),
                anomaly_mean_file='scores/anomaly_mean.npy', anomaly_std_file=None)
metadata.pop('threshold_reference', None)
(target / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print('written', target)
" "$BEHAVIOUR/discriminator_clim" "$OUTPUT/minus_d"
fi

for top in 5 1; do
    step "flags of -D, D, G and the two references, highest $top%"
    "$PYTHON" "$SCRIPTS/compare_era5_detectors.py" --top-percent "$top" \
        --runs "G+clim=$BEHAVIOUR/generator_clim" "menoD+clim=$OUTPUT/minus_d" "D+clim=$BEHAVIOUR/discriminator_clim" \
               "normale=$OUTPUT/normal" "3ore=$OUTPUT/change_3h" 2>&1 | tee "$OUTPUT/logs/flags_${top}pc.log"
done
