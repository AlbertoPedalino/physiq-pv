# ----------------------------------------------------------------------------
# QS shrinkage diagnostico (post-hoc, non tocca compute_qs originale).
# Esegui DOPO la cell di inferenza, dove sono già definiti:
#   qs (xr.DataArray)         output di compute_qs(ds, debug=True)
#   ds (xr.Dataset)           dataset Sentinel + meteo
#   pred_pv_d, true_pv_d      array daytime
#   qs_d, plant_id_d          array daytime
#   _metrics_by_qs_bin        funzione binning
#
# Idea: sample con poca confidence (finestra rolling sparsa) vengono spinti
# verso il prior fleet-median di QS, invece che verso 0. Nessun discard.
# ----------------------------------------------------------------------------

import numpy as np
import pandas as pd

WINDOW = 720          # stesso di compute_qs
N0 = 360              # metà finestra: confidence 0.5 quando hai 360h valide
SCALE = 90            # transizione graduale
ENERGIA_KEY = "ENERGIA"

print("\n=== QS SHRUNK (Bayesian shrinkage diagnostic) ===")

energia_arr = ds[ENERGIA_KEY].values  # (N_plants, T)
qs_raw_arr = np.asarray(qs.values)    # (N_plants, T)

# Conta osservazioni valide (non-NaN, non-zero) nella finestra rolling 720h.
N_plants, T = energia_arr.shape
valid_count = np.zeros((N_plants, T), dtype=np.float32)
valid_mask = np.isfinite(energia_arr) & (energia_arr > 0)
for p in range(N_plants):
    s = pd.Series(valid_mask[p].astype(float))
    valid_count[p] = s.rolling(WINDOW, min_periods=1).sum().values

# Confidence sigmoid: 0 se finestra vuota, ~1 se finestra piena.
conf = 1.0 / (1.0 + np.exp(-(valid_count - N0) / SCALE))

# Prior = mediana fleet del QS_raw su tutti sample finiti.
qs_prior = float(np.nanmedian(qs_raw_arr))
print(f"  QS prior (fleet median): {qs_prior:.3f}")
print(f"  Mean confidence: {conf.mean():.3f}  (1.0 = full window, 0.0 = empty)")

# Shrinkage: se conf basso, QS converge al prior. Niente discard.
qs_shrunk_arr = (
    conf * np.nan_to_num(qs_raw_arr, nan=qs_prior)
    + (1.0 - conf) * qs_prior
).astype(np.float32)

# Estrai sample-wise e applica daytime mask (riusa quelli già definiti).
qs_shrunk_flat = qs_shrunk_arr[plant_id_flat, time_idx_flat]
qs_shrunk_d = qs_shrunk_flat[day_mask]

# Refit binning con QS shrunk.
print("\n--- Binning QS_RAW (originale) ---")
_ = _metrics_by_qs_bin(pred_pv_d, true_pv_d, qs_d)

print("\n--- Binning QS_SHRUNK ---")
qs_shrunk_rows = _metrics_by_qs_bin(pred_pv_d, true_pv_d, qs_shrunk_d)

# Confidence diagnostics per bin shrunk.
conf_flat = conf[plant_id_flat, time_idx_flat]
conf_d = conf_flat[day_mask]

print("\n--- Confidence per bin (QS_SHRUNK) ---")
bins_def = [
    (0.0, 0.3, "low"),
    (0.3, 0.5, "mid-low"),
    (0.5, 0.7, "medium"),
    (0.7, 0.85, "high"),
    (0.85, 1.01, "very high"),
]
for lo, hi, lab in bins_def:
    m = (qs_shrunk_d >= lo) & (qs_shrunk_d < hi)
    if m.sum() == 0:
        continue
    print(
        f"  {lab:10s}: n={int(m.sum()):>9,}  conf_mean={conf_d[m].mean():.3f}  "
        f"qs_raw_mean={qs_d[m].mean():.3f}  qs_shrunk_mean={qs_shrunk_d[m].mean():.3f}"
    )
