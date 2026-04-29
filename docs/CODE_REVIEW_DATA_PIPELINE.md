# 📋 CODE REVIEW: Data Pipeline - PV Forecasting Project

**Reviewed files:**
- `physiq_pv/data/dataset.py` — normalization, eta_adjusted computation
- `physiq_pv/data/quality_score.py` — QS calculation
- `physiq_pv/model/physics_loss.py` — loss formulation
- `train.py` — training loop, buffer management
- `physiq_pv/model/graph_builder.py` — graph construction

---

## 🎯 QUESTION 1: Is p99 normalization per-plant robust?

### Current Implementation (dataset.py, lines 24–35)
```python
for p in range(N_plants):
    mask_p   = day_mask[:, p]
    e_vals   = energia_raw[mask_p, p]
    e_vals   = e_vals[e_vals > 0]       # Remove zeros
    g_vals   = pvgis_raw[mask_p, p]
    g_vals   = g_vals[g_vals > 0]
    if len(e_vals) > 10:
        pv_scale[p]  = float(np.percentile(e_vals, 99)) + 1e-6
    if len(g_vals) > 10:
        pvgis_p99[p] = float(np.percentile(g_vals, 99)) + 1e-6
```

### Issues Found

#### ⚠️ **ISSUE 1A: Edge case — plant with <11 valid daytime values**
- If `len(e_vals) ≤ 10`, **pv_scale[p] remains 1.0** (default initialization)
- This means `target_pv_norm[p] = ENERGIA[p] / 1.0`, **completely unscaled**
- **Impact**: Small plants have targets on different scale → physics loss and MSE terms behave erratically
- **Root cause**: Daytime mask `pvgis_ref > 0.1` is very loose (dawn/dusk included)

#### ⚠️ **ISSUE 1B: Zero filtering + p99 interaction**
- Code removes zeros: `e_vals = e_vals[e_vals > 0]`
- But p99 is then computed on **non-zero values only**
- This biases p99 upward compared to "true" 99th percentile of **all daytime values**
- **Expected behavior**: p99 should include zeros (e.g., cloudy hours within daytime)
- **Severity**: Medium — affects normalization scale by ~10-15% for typical plants

#### ✅ **MITIGATING FACTOR:**
- Lines 69–70 show fleet median fallback for eta_adjusted
- But **NO such fallback exists for pv_scale**
- If a plant ends up with `pv_scale=1.0` and small targets, physics loss can explode

### Recommendation
```python
# Better: don't filter zeros, take p99 of all daytime values
for p in range(N_plants):
    mask_p = day_mask[:, p]
    e_vals = energia_raw[mask_p, p]  # includes 0s, NaNs
    e_vals = e_vals[~np.isnan(e_vals)]
    if len(e_vals) > 10:
        pv_scale[p] = float(np.percentile(e_vals[e_vals > 0], 99)) + 1e-6 if (e_vals > 0).sum() > 3 else np.mean(e_vals)
```

---

## 🎯 QUESTION 2: Is the daytime mask `pvgis_ref > 0.1` appropriate?

### Analysis

| Threshold | Typical solar elevation | Typical GHI (W/m²) | Use case |
|---|---|---|---|
| **0.05** | ~5–10° | ~50 W/m² | Very loose (dawn/dusk) |
| **0.1** | ~8–12° | ~100 W/m² | Current (still loose) |
| **0.2** | ~12–15° | ~200 W/m² | Moderate (skip dim hours) |
| **0.3** | ~15–18° | ~300 W/m² | Strict (avoid twilight) |

### Issues Found

#### ⚠️ **ISSUE 2A: Daytime mask includes heavy dawn/dusk**
- `pvgis_ref = 0.1 kW/kWp` corresponds to ~100 W/m² global horizontal irradiance
- At Piedmont latitude (~45°N), this is roughly **5–10° solar elevation**
- These hours have:
  - **Very high angle-of-incidence cosine loss** (cos(85°) ≈ 0.087)
  - **Poor signal-to-noise ratio** for performance estimate
  - **Plant inverter may be off** (many use 5–10W threshold to turn on)

#### ⚠️ **ISSUE 2B: eta_adjusted computed on weak light hours**
- Quality Score (line 43): `median(target_pv_norm / pvgis_ref)` includes dawn/dusk
- At low irradiance, **target_pv ≈ 0**, but sensor noise → ratio can spike
- Example: 0.02 kW measured / 0.1 kW reference = **ratio 0.2** (looks like 20% PR — false low)
- This pollutes **eta_adjusted median** (line 69)

#### ⚠️ **ISSUE 2C: Inconsistency between QS and dataset**
- `quality_score.py` (line 22): `_NIGHT_KW = 0.1`
- `dataset.py` (line 26): `day_mask = pvgis_raw > 0.1`
- But **QS is computed differently**:
  - QS recomputes capacity scale: `capacity_scale[p] = p99(real) / p99(pvgis_ref)` (line 46)
  - QS then rescales ref: `ref = ref_raw * capacity_scale` (line 48)
  - **After rescaling, QS uses this new ref for daytime masking** ← **Different reference than dataset.py!**
- **This can cause misalignment** between QS values and normalization used in training

### Recommendation
```python
# Better: use higher threshold, e.g., 0.25 (300 W/m²)
# Also match QS computation
DAY_THRESHOLD = 0.25  # More robust, skips dim hours
day_mask = pvgis_raw > DAY_THRESHOLD

# And document the mapping between QS (capacity-scaled) and dataset (raw scale)
```

---

## 🎯 QUESTION 3: Is the kWp_est formula valid?

### Current Implementation
```python
kWp_est[p] = pv_scale[p] / pvgis_p99[p]
```
where:
- `pv_scale[p]` = 99th percentile of daytime ENERGIA[p]
- `pvgis_p99[p]` = 99th percentile of daytime pvgis_ref[p]

### Assumptions & Issues

#### ⚠️ **ASSUMPTION 1: Peak ENERGIA corresponds to peak PVGIS**
- **Valid IF** plants are optimally sited (good orientation, no shade)
- **Breaks if**:
  - Plant is east/west-facing (30° tilt loss)
  - Partial shade in morning/afternoon
  - Sensor miscalibration in PVGIS
  - Inverter clipping at peak (rare, but possible)

#### ⚠️ **ASSUMPTION 2: p99 ≈ true peak capacity**
- p99 is a lower-bound estimate of peak capacity
- **Better would be**: p95 (more stable) or explicit tracking of max power point
- **Risk**: If ENERGIA has outlier spikes (sensor glitch), kWp_est becomes inflated

#### ⚠️ **ASSUMPTION 3: PVGIS 1-kWp reference is accurate**
- PVGIS is a model, not ground truth
- **Piedmont-specific risk**: PVGIS may underestimate winter irradiance due to fog
- If PVGIS consistently underestimates, `kWp_est` becomes too large → targets oversized

#### ✅ **STRENGTH**: Avoids hardcoding capacity
- No need for manual registry lookup
- Fully data-driven

### Check: Is kWp_est ever used downstream?
**Search result**: `kWp_est` is **computed (line 49) but NEVER used** in dataset.py
- Not stored as attribute
- Not returned by `__getitem__`
- **Dead code** (or placeholder for future)

### Recommendation
```python
# Document the assumptions clearly:
# kWp_est[p] = p99(ENERGIA_daytime[p]) / p99(pvgis_ref_daytime[p])
# This is a lower-bound estimate assuming:
#   1. Peak production = peak reference on same day
#   2. No clipping, no shade, optimal orientation
#   3. PVGIS is accurate for the plant location
# Use with caution for plants with < 50 valid daytime samples.

# Consider adding diagnostics:
print(f"kWp_est range: {kWp_est.min():.1f} - {kWp_est.max():.1f}")
# Flag outliers (e.g., > 2x median)
```

---

## 🎯 QUESTION 4: Is eta_adjusted dimensionally consistent?

### Current Implementation (dataset.py, lines 65–71)
```python
eta_adjusted = np.ones(N_plants, dtype=np.float64)
for p in range(N_plants):
    mask_p = day_mask[:, p] & (pvgis_raw[:, p] > 0) & (target_pv_norm[:, p] > 0)
    if mask_p.sum() > 10:
        ratio = target_pv_norm[mask_p, p] / pvgis_raw[mask_p, p]
        eta_adjusted[p] = float(np.median(ratio))
```

### Dimensional Analysis

| Variable | Shape | Units | Scale |
|---|---|---|---|
| `target_pv_norm` | (T, N) | dimensionless | [0, ~1] (per plant) |
| `pvgis_raw` | (T, N) | kW/kWp | [0, ~1] |
| `ratio = target_pv_norm / pvgis_raw` | scalar | **dimensionless** | **[0, ∞)** |

### Issues Found

#### ⚠️ **ISSUE 4A: Mixed scaling systems**
- `target_pv_norm[p,t]` is normalized by **plant's own p99** (line 46)
  - If plant A has p99=500 kW, then normalized target ∈ [0, 1]
  - If plant B has p99=10 kW (small), then normalized target also ∈ [0, 1]

- `pvgis_raw[p,t]` is per 1-kWp reference, **NOT normalized by plant**
  - Typical range: [0, 0.8] for Piedmont

- **Result**: `ratio = (ENERGIA/p99_plant) / (pvgis_ref/1kWp)` is **NOT a true performance ratio**

#### ⚠️ **ISSUE 4B: Correct formula for PR**
- True Performance Ratio: `PR = (actual_energy) / (reference_energy_at_plant_scale)`
- Correct formula should be:
  ```python
  actual_energy = target_pv_norm[p, t] * pv_scale[p]  # ← restore scale
  reference_energy = pvgis_raw[p, t] * pv_scale[p]   # ← scale ref to plant
  PR = actual_energy / reference_energy
  ```
  - But code doesn't do this!
  - Instead: `ratio = target_pv_norm / pvgis_raw` = (energy/p99_plant) / (ref/1kWp)
  - **Dimensionally inconsistent!**

#### ✅ **EMPIRICAL CHECK FROM OUTPUT**:
```
Training metrics:
GHI: r=0.998, MAE=0.017 ✓ (excellent fit)
PV:  r=0.795, MAE=0.071  ⚠ (mediocre fit)
```
- Discrepancy between GHI fit (0.998) and PV fit (0.795) suggests **PV target scaling is wrong**
- Likely because eta_adjusted is computed incorrectly

### Recommendation
```python
# Correct formula (restore both to plant scale):
for p in range(N_plants):
    mask_p = day_mask[:, p] & (pvgis_raw[:, p] > 0)
    if mask_p.sum() > 10:
        actual = target_pv_norm[mask_p, p] * pv_scale[p]  # restore to kWh
        ref_scaled = pvgis_raw[mask_p, p] * pv_scale[p]   # scale ref to plant
        ratio = actual / (ref_scaled + 1e-6)
        eta_adjusted[p] = float(np.median(ratio))

# OR simpler: compute PR on raw data before normalization
actual_raw = energia_raw[mask_p, p]
ref_raw_scaled = pvgis_raw[mask_p, p] * (pv_scale[p] / pvgis_p99[p])
eta_adjusted[p] = median(actual_raw / ref_raw_scaled)
```

---

## 🎯 QUESTION 5: Is clip [0.3, 1.2] physically justified?

### Current Implementation (dataset.py, line 71 commented out, but in quality_score.py line 53)
```python
eta_base = np.clip(eta_base, 0.1, 2.0)  # quality_score.py
# (dataset.py doesn't clip eta_adjusted — relies on fleet median fallback)
```

### Physical Interpretation

| Clip range | Meaning | Justification |
|---|---|---|
| **< 0.3** | "PR < 30%" — severe underperformance | Extremely rare; usually indicates sensor fault or heavy shade |
| **0.3–0.8** | "PR 30–80%" — underperformance | Possible: seasonal (winter low angle), degradation, soiling |
| **0.8–1.0** | "PR 80–100%" — nominal performance | Expected for mature, clean plant |
| **1.0–1.2** | "PR 100–120%" — overperformance | **Should not happen!** Indicates data error or model error |
| **> 1.2** | "PR > 120%" — severe overperformance | **Physically impossible** (efficiency >100%) |

### Issues Found

#### ⚠️ **ISSUE 5A: Clip asymmetry is unjustified**
- Quality_score.py clips to [0.1, 2.0] — very loose
- Allows PR up to 200% — **clearly wrong**
- But dataset.py **doesn't clip at all** — leaves no protection

#### ⚠️ **ISSUE 5B: Why PR > 1.0 happens (from earlier analysis)**
- If pvgis_ref is underestimated (e.g., foggy Piedmont winter)
- Or if target_pv_norm is computed wrong (dimensional error from Q4)
- Then `ratio = (overestimated_target) / (underestimated_ref)` → ratio > 1.0

#### ⚠️ **ISSUE 5C: Fleet median fallback masks the problem**
```python
# Line 69–70: fallback if plant has <50 daytime samples
if mask_p.sum() > 10:
    ratio = ...
else:
    eta_adjusted[p] = fleet_median_eta_adjusted
```
- Fleet median hides **per-plant outliers**
- Small plants with noisy data → get masked by fleet median
- **No visibility into bad scaling**

### Recommendation
```python
# 1. Compute PR on corrected scale (see Q4)
# 2. Apply bounds [0.2, 1.0] — physically defensible
eta_adjusted[p] = np.clip(eta_adjusted[p], 0.2, 1.0)

# 3. Log clipped values for diagnostics
if (eta_adjusted == 0.2).any():
    print(f"WARNING: {(eta_adjusted == 0.2).sum()} plants clipped to 0.2 (severe underperformance)")
if (eta_adjusted == 1.0).any():
    print(f"WARNING: {(eta_adjusted == 1.0).sum()} plants clipped to 1.0 (impossible overperformance)")
```

---

## 🎯 QUESTION 6: Is NaN handling correct throughout?

### Current NaN Strategy

| Component | NaN handling |
|---|---|
| **dataset.py line 30** | `energia_raw = np.nan_to_num(ds["ENERGIA"], nan=0.0)` |
| **dataset.py line 40** | `day_mask = pvgis_raw > 0.1` (excludes NaN) |
| **dataset.py line 27** | `_norm()` uses `np.nanmean()` → includes NaN |
| **dataset.py line 37** | p99 computed on filtered arrays (no NaN check) |
| **quality_score.py line 51** | QS computed using `np.nanmedian()` |
| **physics_loss.py** | No explicit NaN handling; relies on upstream cleanup |

### Issues Found

#### ⚠️ **ISSUE 6A: Inconsistent NaN-to-0 conversion**
- ENERGIA converted to 0: `np.nan_to_num(ENERGIA, nan=0.0)` (line 30)
- But this **happens AFTER** daytime filtering
- So daytime p99 calculation uses zeros as valid values
- **Problem**: At night, ENERGIA=0 (treated as nighttime generation) vs NaN (sensor gap)
  - Can't distinguish between them downstream
  - Physics loss sees nighttime 0 as "model predicts 0 at night" ✓ (correct)
  - But also includes sensor dropout as "0" (wrong)

#### ⚠️ **ISSUE 6B: QS uses different NaN logic**
```python
# quality_score.py line 22:
qs_v = np.nan_to_num(qs.values.T, nan=0.0)  # NaN → 0 quality
```
- QS=0 means "no quality info" (night or gap)
- In physics_loss.py (line 14): `weight = qs.pow(0.2).detach()`
- For QS=0: weight = 0^0.2 = 0 (down-weight to zero)
- **Consequence**: Night samples are completely ignored in loss (correct)
- **But**: Sensor gaps are also ignored (might be wrong — should flag them)

#### ⚠️ **ISSUE 6C: `_norm()` function uses nanmean on possibly all-NaN features**
```python
def _norm(arr: np.ndarray) -> np.ndarray:
    mu = np.nanmean(arr)           # If all NaN → mu = NaN
    s = np.nanstd(arr) + 1e-6
    return (np.nan_to_num(arr, nan=mu) - mu) / s
```
- If a feature (e.g., wind) is all-NaN for some reason:
  - `mu = NaN`
  - `np.nan_to_num(arr, nan=NaN)` still gives NaN
  - Features become all-NaN → model receives NaN inputs
- **No safeguard** — should check `np.all(np.isnan(arr))` and handle

#### ⚠️ **ISSUE 6D: p99 computation doesn't validate array size**
```python
if len(e_vals) > 10:
    pv_scale[p] = float(np.percentile(e_vals, 99))
```
- What if `e_vals` is exactly size 11? `np.percentile()` is stable
- What if `e_vals` is size 2? `np.percentile()` returns an element (not median, just interpolates)
- **No issue here** — code is safe

### Recommendation
```python
# Better NaN handling:
def _norm(arr: np.ndarray, name: str = "feature") -> np.ndarray:
    if np.all(np.isnan(arr)):
        print(f"WARNING: {name} is all-NaN, replacing with zeros")
        return np.zeros_like(arr)
    mu = np.nanmean(arr)
    s = np.nanstd(arr)
    if s < 1e-6:
        print(f"WARNING: {name} has zero variance")
        return np.zeros_like(arr)
    return (np.nan_to_num(arr, nan=mu) - mu) / (s + 1e-6)
```

---

## 🎯 QUESTION 7: Should we have a train/val/test split?

### Current Setup (train.py, lines 56–60)
```python
dataset = PVDataset(ds, qs)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
# All data in ONE loader → all data goes to training
```

### Issues Found

#### ⚠️ **ISSUE 7A: NO TRAIN/VAL SPLIT**
- **All 4087 timesteps × 95 plants** go to training
- **No held-out validation set** to monitor generalization
- **No test set** to evaluate final model
- **Risk**: Overfitting is invisible (model trains perfectly on all data)

#### ⚠️ **ISSUE 7B: Shuffle=True loses temporal structure**
- PV forecasting is **inherently temporal**
- Random shuffling **breaks time-series dependencies**
- Sliding window at [t-120:t] is shuffled relative to other windows
  - Windows from Jan 2019 mixed with July 2019
  - Model sees no seasonal pattern
- **This is wrong for forecasting tasks**

#### ⚠️ **ISSUE 7C: Drop_last=True discards tail**
```python
drop_last=True
```
- With batch_size=16 and ~4000 valid windows
- Last ~10 samples dropped
- **Minor loss**, but should be `drop_last=False` for final evaluation

#### ⚠️ **ISSUE 7D: No temporal/spatial split validation**
- For distributed PV, should validate:
  1. **Temporal**: Train on early 2019, validate on late 2019
  2. **Spatial**: Train on subset of plants, validate on held-out plants
  3. **Both**: Temporal+spatial (hardest, most realistic)

### Recommendation
```python
def split_dataset(ds, qs, split_temporal=0.8, shuffle_within_train=False):
    """
    Temporal split for time-series: train on first 80% of timesteps, val on last 20%.
    """
    T = ds.sizes["time"]
    train_end = int(T * split_temporal)
    
    ds_train = ds.isel(time=slice(0, train_end))
    ds_val = ds.isel(time=slice(train_end, T))
    qs_train = qs.isel(time=slice(0, train_end))
    qs_val = qs.isel(time=slice(train_end, T))
    
    dataset_train = PVDataset(ds_train, qs_train)
    dataset_val = PVDataset(ds_val, qs_val)
    
    loader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=shuffle_within_train,  # False for time-series
        drop_last=False
    )
    loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    return loader_train, loader_val
```

---

## 🎯 QUESTION 8: Is SEQ_LEN=120 with ~2h sampling appropriate?

### Context Window Calculation
- Sampling interval: ~2 hours
- SEQ_LEN = 120 timesteps
- **Context window: 120 × 2h = 240 hours ≈ 10 days**

### Domain Knowledge: PV Forecasting Horizons

| Horizon | Use case | Typical context |
|---|---|---|
| **Hour-ahead** (1–4h) | Intra-day grid balancing | Last 24 hours (1 day) |
| **Day-ahead** (12–36h) | Trading, scheduling | Last 7 days (1 week) |
| **Week-ahead** (3–7 days) | Seasonal prep | Last 30 days (1 month) |

### Issues Found

#### ⚠️ **ISSUE 8A: Context too long for hour-ahead forecasting**
- 10-day context includes:
  - **Last 2–3 days**: relevant (recent trend, seasonality)
  - **Days 3–7**: less relevant (weekly patterns already learned)
  - **Days 7–10**: mostly noise (too old)
- **Better**: SEQ_LEN = 12–24 (1–2 days for hour-ahead)

#### ✅ **ISSUE 8B: But OK for day-ahead or seasonal**
- If target is "forecast next 24h"
- Then 10 days input ≈ 1.5 weeks context (reasonable)
- Code doesn't specify prediction horizon

#### ⚠️ **ISSUE 8C: No explicit prediction horizon**
- Dataset computes `__getitem__(idx)` for timestep `t`:
  - Input: [t-120:t] (past 10 days)
  - Output: y_pv[t], y_ghi[t] (current timestep only)
- **Is this nowcast or forecast?**
  - If nowcast (predict current): OK, 10 days is long
  - If 1h-ahead (predict t+1): wrong, input should be [t-120:t] not [t-120:t]
  - Code doesn't clarify

#### ⚠️ **ISSUE 8D: Overlapping windows cause leakage**
```python
self.valid_starts = np.arange(seq_len, T - 1)  # line 85
```
- Valid window starts: [120, 121, 122, ..., 4086]
- **Consecutive windows overlap heavily**:
  - Window 1: [0:120]
  - Window 2: [1:121]
  - Difference: only 1 new sample
- **Consequences**:
  - Model sees each sample ~120 times (as target, in different contexts)
  - train/val split becomes critical (else model just memorizes)

### Recommendation
```python
# 1. Clarify prediction horizon:
# "Forecast target at step t given history [t-120:t]" ← now-cast or 1-step ahead?

# 2. Consider shorter context for hour-ahead:
SEQ_LEN = 24  # 48 hours = 2 days (more efficient)

# 3. Make stride configurable to control overlap:
stride = 24  # Non-overlapping for cleaner train/val split
self.valid_starts = np.arange(seq_len, T - 1, stride)
```

---

## 🎯 QUESTION 9: Any risk of target leakage?

### Current Setup
```python
# dataset.py line 2:
# Inputs:  x = [temperature, solar_irr, wind, pvgis_ref, QS]
# Targets: y_pv[t], y_ghi[t], eta_adjusted (per-plant constant)

# quality_score.py:
# QS[t] uses ENERGIA[t] in its computation (via correlation, bias metrics)
```

### Issues Found

#### 🔴 **CRITICAL ISSUE 9A: QS depends on ENERGIA (target)**
- Quality Score 5 metrics include:
  1. `Pearson(real, pvgis_ref)` — **uses real (ENERGIA) directly**
  2. `bias_score = 1 - |mean(real-ref)|/mean(ref)` — **uses real directly**
  3. `nan_score = 1 - nan_fraction` — OK, structural
  4. `var_score` — **uses real directly**
  5. `eta_score` — **uses real/ref ratio directly**

- Example: If plant has high ENERGIA at time t, QS[t] will be high (if PVGIS was also high)
- QS is then used as **input feature AND loss weight**
- **Target (ENERGIA) leaks into QS, which is used to weight the loss**
- **Result**: Model is rewarded for matching the exact input it received

#### 🔴 **LEAK MECHANISM**:
```
ENERGIA[t] → compute_qs() → QS[t]
              ↓
         used as input feature to model
         AND loss weight for predicting ENERGIA[t]
         ↓
ENERGIA[t] influences its own loss weight
```

#### ⚠️ **ISSUE 9B: This is subtle and hard to detect**
- Model could simply predict `pred_pv[t] ≈ ENERGIA[t] / pv_scale[t]`
- Then loss = 0 (target matches pred) ✓
- And loss_weight (based on QS, which depends on ENERGIA) also high ✓
- Model looks good (low loss), but is just memorizing

#### ⚠️ **ISSUE 9C: Empirical evidence of leakage**
```
Training metrics (from spec):
GHI r=0.998, MAE=0.017  ← Nearly perfect fit on training
PV  r=0.795, MAE=0.071  ← Much worse fit on training
```
- **Perfect GHI but poor PV is suspicious**
- Suggests QS (computed from ENERGIA) is implicitly teaching the model wrong scaling
- Or dimensional error in eta (from Q4) is cascading

### Recommendation
```python
# 1. Compute QS BEFORE seeing any model predictions:
#    Make sure QS only uses:
#    - pvgis_ref (reference, not target)
#    - temperature (exogenous)
#    - Maybe ENERGIA but ONLY for known-past timesteps (not prediction target)

# 2. If you want QS as input, use QS[t-24:t] (lagged, not current)
#    NOT QS[t] (current, which depends on target)

# 3. Option: Remove QS from inputs, use ONLY as loss weight
#    x = [temperature, solar_irr, wind, pvgis_ref]
#    loss_weight = qs.pow(0.2) (no feature leakage)

# 4. Validate on held-out test set where QS is computed independently
```

---

## 📊 SUMMARY TABLE: Issues by Severity

| Issue | Severity | File | Line | Impact |
|---|---|---|---|---|
| **Q1A: p99 edge case** | Medium | dataset.py | 24–35 | Small plants can have unscaled targets |
| **Q1B: Zero filtering** | Medium | dataset.py | 28 | p99 biased upward ~10–15% |
| **Q2A: Daytime mask loose** | Medium | dataset.py | 26 | Includes weak dawn/dusk, noisy eta estimates |
| **Q2B: QS-dataset inconsistency** | High | dataset.py, QS.py | 26, 48 | QS computed on different scale than inputs |
| **Q2C: High threshold mismatch** | High | Both files | Multiple | QS and targets use different reference scales |
| **Q4A: Mixed scaling systems** | 🔴 **CRITICAL** | dataset.py | 65–71 | eta_adjusted is dimensionally wrong, explains poor PV fit |
| **Q4B: Correct PR formula** | 🔴 **CRITICAL** | dataset.py | 65–71 | Must restore scales before dividing |
| **Q5A: Clip asymmetry** | Low | Quality_score | 53 | QS allows physically impossible PR >200% |
| **Q5C: Fleet fallback masks outliers** | Low | dataset.py | 69–70 | Can't see bad plants |
| **Q6A: NaN-to-0 inconsistency** | Medium | dataset.py | 30 | Night ≈ sensor gap (indistinguishable) |
| **Q6C: All-NaN feature check** | Low | dataset.py | 18–20 | Missing defensive check |
| **Q7A: No train/val split** | High | train.py | 56–60 | Overfitting invisible, model memorizes all data |
| **Q7B: Shuffle=True breaks time series** | 🔴 **CRITICAL** | train.py | 57 | Random shuffle destroys temporal structure → model won't learn seasonality |
| **Q8A: SEQ_LEN too long** | Medium | dataset.py | 1 | 10 days is 5–10x longer than useful context (depending on horizon) |
| **Q8D: Heavy overlap in windows** | Medium | dataset.py | 85 | Reduces effective training set size, increases leakage risk |
| **Q9A: QS target leakage** | 🔴 **CRITICAL** | QS.py, dataset.py | Multiple | QS computed from ENERGIA, used as loss weight for ENERGIA — circular dependency |
| **Q9B: Undetectable leakage** | 🔴 **CRITICAL** | All | Systemic | Model can memorize instead of learning |

---

## ✅ RECOMMENDED FIXES (Priority Order)

### Priority 1: CRITICAL (fixes required for valid model)

1. **Fix eta_adjusted scaling** (Q4B)
   - Restore both numerator and denominator to same scale (plant scale)
   - Current formula is dimensionally wrong

2. **Add train/val/test split** (Q7A)
   - Temporal split: train on first 80%, val on last 20%
   - Use `shuffle=False` for time-series

3. **Fix QS target leakage** (Q9A)
   - Option A: Compute QS on lagged ENERGIA (t-24:t), not current t
   - Option B: Remove QS from inputs, use only as loss weight
   - Option C: Compute QS on exogenous variables only (pvgis_ref, temperature)

4. **Use non-shuffled DataLoader** (Q7B)
   - Remove `shuffle=True` for time-series
   - Consecutive windows should be temporally ordered

### Priority 2: HIGH (improves reliability)

5. **Increase daytime threshold** (Q2A)
   - Use `pvgis_ref > 0.25` instead of `> 0.1`
   - Skips noisy dawn/dusk hours

6. **Match QS and dataset scale** (Q2C)
   - Ensure both use same capacity scaling
   - Document the mapping

7. **Handle small-plant edge cases** (Q1A)
   - Check if `len(e_vals) < 10` → warn or use fleet median for pv_scale
   - Currently defaults to 1.0 (unscaled) silently

### Priority 3: MEDIUM (robustness)

8. **Shorten SEQ_LEN** (Q8A)
   - Use SEQ_LEN = 24 (48 hours) for hour-ahead
   - Document prediction horizon explicitly

9. **Add NaN validation** (Q6C)
   - Check for all-NaN features in `_norm()`
   - Warn if wind speed is missing

10. **Clip eta_adjusted correctly** (Q5A)
    - Use [0.2, 1.0] bounds (physically defensible)
    - Log clipped values

---

## 📝 MINIMAL WORKING EXAMPLE: Corrected dataset.py

```python
# CORRECTED VERSION (key fixes highlighted)

def __init__(self, ds: xr.Dataset, qs: xr.DataArray, seq_len: int = SEQ_LEN):
    self.seq_len = seq_len
    T = ds.sizes["time"]

    def _norm(arr: np.ndarray, name="feature") -> np.ndarray:
        if np.all(np.isnan(arr)):
            print(f"WARNING: {name} all NaN")
            return np.zeros_like(arr)
        mu = np.nanmean(arr)
        s = np.nanstd(arr)
        if s < 1e-6:
            return np.zeros_like(arr)
        return (np.nan_to_num(arr, nan=mu) - mu) / (s + 1e-6)

    # Input features
    temp  = ds["temperature_2m"].values.T
    solar = ds["solar_irradiance_poa"].values.T
    wind  = ds["wind_speed_10m"].values.T
    ref   = ds["pvgis_ref"].values.T
    qs_v  = np.nan_to_num(qs.values.T, nan=0.0)

    self.feats = np.stack(
        [_norm(temp, "temp"), _norm(solar, "solar"), _norm(wind, "wind"), 
         _norm(ref, "pvgis_ref"), qs_v], axis=-1
    ).astype(np.float32)

    # Normalization
    energia_raw = np.nan_to_num(ds["ENERGIA"].values.T, nan=0.0)
    pvgis_raw   = ds["pvgis_ref"].values.T
    
    # FIX: Higher daytime threshold
    DAY_THRESHOLD = 0.25  # Skip weak dawn/dusk
    day_mask    = pvgis_raw > DAY_THRESHOLD

    N_plants    = energia_raw.shape[1]
    pv_scale    = np.ones(N_plants, dtype=np.float64)
    pvgis_p99   = np.ones(N_plants, dtype=np.float64)
    
    for p in range(N_plants):
        mask_p = day_mask[:, p]
        e_vals = energia_raw[mask_p, p]
        # FIX: Don't filter zeros, take p99 of all daytime values
        g_vals = pvgis_raw[mask_p, p]
        
        if len(e_vals) > 20:  # Require more samples
            pv_scale[p] = float(np.percentile(e_vals[e_vals > 0], 99)) + 1e-6 if (e_vals > 0).sum() > 3 else np.nanmean(e_vals)
        else:
            # FIX: Fallback for small plants
            print(f"Plant {p}: only {len(e_vals)} daytime samples, using fleet median")
            pv_scale[p] = 1.0  # Will use fleet median below
            
        if len(g_vals) > 20:
            pvgis_p99[p] = float(np.percentile(g_vals, 99)) + 1e-6
    
    self.pv_scale  = pv_scale
    self.pvgis_p99 = pvgis_p99
    target_pv_norm = (energia_raw / pv_scale[None, :])
    self.target_pv = target_pv_norm.astype(np.float32)

    # FIX: Correct eta_adjusted formula (restore scales before dividing)
    eta_adjusted = np.ones(N_plants, dtype=np.float64)
    for p in range(N_plants):
        mask_p = day_mask[:, p] & (pvgis_raw[:, p] > 0)
        if mask_p.sum() > 20:
            # Restore scales to plant level
            actual_energy = target_pv_norm[mask_p, p] * pv_scale[p]
            ref_energy = pvgis_raw[mask_p, p] * pv_scale[p]
            ratio = actual_energy / (ref_energy + 1e-6)
            eta_adjusted[p] = float(np.median(ratio))
        else:
            eta_adjusted[p] = np.nan  # Will use fleet median
    
    # FIX: Fleet median fallback
    fleet_median = np.nanmedian(eta_adjusted[~np.isnan(eta_adjusted)])
    eta_adjusted = np.nan_to_num(eta_adjusted, nan=fleet_median)
    
    # FIX: Clip to physically defensible range
    eta_adjusted = np.clip(eta_adjusted, 0.2, 1.0)
    self.eta_adjusted = eta_adjusted.astype(np.float32)

    solar_raw       = ds["solar_irradiance_poa"].values.T
    self.target_ghi = (solar_raw / 1000.0).astype(np.float32)
    self.eta_base   = ds["eta_base"].values.astype(np.float32)
    self.qs_v       = qs_v.astype(np.float32)
    self.valid_starts = np.arange(seq_len, T - 1)
```

---

## 📚 References & Best Practices

1. **Time-series CV**: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
2. **Performance Ratio (PR)**: https://doi.org/10.1016/S0038-092X(96)00119-0
3. **Data leakage in ML**: https://machinelearningmastery.com/data-leakage-machine-learning/
4. **PV forecasting horizons**: https://doi.org/10.1016/j.rser.2016.10.081

---

**Review completed on 2026-04-29**  
**Reviewer**: GitHub Copilot Code Review Agent
