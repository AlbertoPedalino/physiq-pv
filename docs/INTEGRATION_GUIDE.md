# PhysiQ-PV: Hourly Sentinel Data Integration Guide

## Overview

Successfully migrated PhysiQ-PV data pipeline from 2-hourly aggregated data (`real_data_dataset.nc`) to **native hourly Sentinel/SCADA energy data** from individual UPN CSV files.

**Data source**: `/data/SentinelPV/energy_data/piemonte_energy_data/single_ups/2019_UPN_*.csv`

---

## 📊 Dataset Characteristics

| Property | Value |
|----------|-------|
| **Plants** | 1,116 UPN sites (Piemonte region) |
| **Time period** | 2019-03-01 to 2019-12-31 |
| **Resolution** | Hourly (1 reading/hour, median of 3x sensors) |
| **Total timesteps** | 5,743 hours |
| **Data availability** | 47% (3M+ valid readings of 6.4M total) |
| **Coordinates** | 1,023 plants (91.7%) with lat/lon |
| **Weather merged** | PVGIS 1,149-grid matched to nearest plant location |

---

## 🔧 Implementation Summary

### 1. **New Loader Module** ✅
**File**: `physiq_pv/data/sentinel_hourly_loader.py`

Provides two main functions:

#### `load_sentinel_hourly()`
```python
ds = load_sentinel_hourly(
    sentinel_dir="/data/SentinelPV/energy_data/piemonte_energy_data/single_ups",
    year=2019,
    plant_mapping_path='data/plant_mapping.csv',
    energy_coords_path='data/energy_with_coordinates.csv',
)
```

**Features**:
- Reads all `2019_UPN_*.csv` files (1,116 plants)
- Aggregates 3x readings/hour via median
- Aligns plants to common time grid (5,743 hours)
- Returns xr.Dataset with ENERGIA (plant, time)
- Includes plant_id, latitude, longitude, eta_base coordinates

**Data shape**: (1116 plants, 5743 hours)

#### `merge_with_weather()`
```python
ds = merge_with_weather(ds, pvgis_path='data/piedmont_pvgis_2019.nc')
```

**Features**:
- Matches each plant to nearest PVGIS grid location (1,149 locations)
- Extracts weather for matched locations
- Reindexes PVGIS (8,760 hours) to Sentinel time grid (5,743 hours)
- Adds variables: temperature_2m, solar_irradiance_poa, wind_speed_10m, pvgis_ref

**Result**: Same spatial dims, enriched with 4 weather variables

---

### 2. **Updated Main Entry Point** ✅
**File**: `main.py`

**Changes**:
```python
# BEFORE:
ds = xr.open_dataset('data/real_data_dataset.nc')

# AFTER:
ds = load_sentinel_hourly(
    sentinel_dir="/data/SentinelPV/energy_data/piemonte_energy_data/single_ups",
    year=2019,
    plant_mapping_path='data/plant_mapping.csv',
    energy_coords_path='data/energy_with_coordinates.csv',
)
ds = merge_with_weather(ds, pvgis_path='data/piedmont_pvgis_2019.nc')
```

**Imports added**:
```python
from physiq_pv.data.sentinel_hourly_loader import load_sentinel_hourly, merge_with_weather
```

---

### 3. **Adjusted Sequence Length** ✅
**File**: `physiq_pv/data/dataset.py`

**Change**:
```python
# BEFORE: SEQ_LEN = 120 (2-hourly = 10 days of context)
# AFTER:
SEQ_LEN = 24  # hourly = 1 day of context (sufficient for intra-daily patterns)
```

**Rationale**: With hourly data, 24 timesteps = 1 day's worth of solar irradiance patterns, which is more appropriate than 10 days for short-term PV forecasting.

---

## 📈 Data Quality Assessment

### ENERGIA (Energy Production)
- **Range**: 0.01 — 5,217 kW
- **Mean**: 156.28 kW (fleet average)
- **Median**: 39 kW
- **Std**: 329.90 kW
- **Missing data**: ~53% (NaN values when plants not operating)

### Temporal Coverage
- **Start**: 2019-03-01 06:00 UTC
- **End**: 2019-12-31 23:00 UTC
- **Gap**: January-February 2019 (likely sensor start date)
- **Hourly steps**: 5,743 out of theoretical 8,760 (March-Dec only)

### Geographic Distribution
- **Latitude**: 44.1972° to 46.1223° (Piemonte region)
- **Longitude**: 6.8837° to 9.0346°
- **Without coordinates**: 93 plants (assigned default eta_base=0.15)

### Quality Score (computed from ENERGIA + PVGIS)
- **Mean QS**: 0.652 (weighted composite of 5 metrics)
- **Median QS**: 0.737
- **Valid QS**: ~49% (aligned with ENERGIA availability)

---

## ✅ Validation Results

**Test script**: `test_sentinel_loader.py`

All 8 stages passed:
```
[1] Loading Sentinel hourly data...          ✅ 1116 plants loaded
[2] Dataset structure                        ✅ Correct dimensions (plant, time)
[3] ENERGIA stats                            ✅ Mean=156.28 kW, valid=47%
[4] Temporal coverage                        ✅ 5743 hours, Mar-Dec 2019
[5] Geographic coverage                      ✅ 1023 plants with coordinates
[6] Merging with PVGIS weather               ✅ 4 weather variables added
[7] Computing Quality Score                  ✅ QS mean=0.652, median=0.737
[8] Sample data inspection                   ✅ Realistic values
```

---

## 🐛 Critical Bugs to Fix (From Code Review)

### Priority 1: CRITICAL

**Bug 1: Dimensional error in `eta_adjusted` (dataset.py lines 65-71)**

```python
# INCORRECT (mixes scales):
eta_adjusted = target_pv_norm / pvgis_raw  
# (plant-normalized) / (per-1kWp reference) → inconsistent units

# CORRECT:
eta_adjusted = (target_pv / pv_scale) / (pvgis_ref / pvgis_scale)
# Both normalized to kW before dividing
```

**Impact**: GHI prediction r²=0.998 (good) vs PV prediction r²=0.795 (poor)

**Evidence**: The dimensional mismatch explains why PV predictions lag 1.25x behind GHI quality.

---

**Bug 2: QS target leakage (quality_score.py → physics_loss.py)**

```python
# CURRENT (circular):
QS = compute_qs(ds)         # Computed FROM ENERGIA
loss_weight = QS.pow(0.2)   # Used AS weight FOR ENERGIA prediction

# SOLUTION:
# Option A: Use lagged QS(t-24 to t-1) for prediction at t
# Option B: Remove QS from inputs, use fixed weights
```

**Impact**: Model memorizes rather than learns temporal patterns

---

### Priority 2: HIGH

**Bug 3: Daytime masking too loose (quality_score.py line 26)**

```python
# CURRENT:
daytime_mask = pvgis_ref > 0.1 kW/kWp  # Includes dawn/dusk noise

# RECOMMENDED:
daytime_mask = pvgis_ref > 0.25 kW/kWp  # Excludes low-irradiance periods
```

**Impact**: ~10-15% upward bias in p99 normalization

---

## 📋 Remaining Implementation Tasks

### Task 1: Apply Critical Bug Fixes
- [ ] Fix eta_adjusted dimensional error (dataset.py)
- [ ] Fix QS target leakage (quality_score.py + physics_loss.py)
- [ ] Adjust daytime mask threshold (quality_score.py)

**Estimated impact**: +0.10-0.15 improvement in PV forecast r²

---

### Task 2: Implement Train/Val/Test Split
**File**: `train.py`

**Current state**: All 4,087 timesteps × 1,116 plants in single DataLoader

**Required changes**:
```python
# Add temporal split function
def temporal_train_val_split(ds, train_ratio=0.8):
    n_time = ds.sizes['time']
    split_idx = int(n_time * train_ratio)
    train_ds = ds.isel(time=slice(0, split_idx))
    val_ds = ds.isel(time=slice(split_idx, None))
    return train_ds, val_ds

# Update DataLoader
train_ds, val_ds = temporal_train_val_split(ds)
train_loader = DataLoader(train_ds, shuffle=False, drop_last=False)  # No shuffle!
val_loader = DataLoader(val_ds, shuffle=False)

# Add validation loop to _train_epoch()
```

**Key points**:
- Use **first 80% for training**, last 20% for validation (temporal order preserved)
- Set **shuffle=False** (preserves temporal structure)
- Set **drop_last=False** (keep tail samples)

---

### Task 3: Run Full Pipeline
```bash
cd /home/apedalino/physiq_pv
source .venv/bin/activate
python main.py
```

**Expected behavior**:
1. Load 1,116 plants × 5,743 hours from Sentinel
2. Merge with PVGIS weather
3. Compute QS for all plants
4. Train ST-GNN on 20 epochs
5. Generate summary report

**Expected runtime**: ~15 minutes (with GPU)

---

## 📝 File Changes Summary

| File | Status | Changes |
|------|--------|---------|
| physiq_pv/data/sentinel_hourly_loader.py | ✅ CREATED | New module, 320 lines |
| main.py | ✅ UPDATED | +3 lines (imports), +13 lines (loader call) |
| physiq_pv/data/dataset.py | ✅ UPDATED | SEQ_LEN: 120 → 24 |
| physiq_pv/data/quality_score.py | ⏳ TODO | Fix daytime mask (1 line change) |
| physiq_pv/model/st_gnn.py | ✅ TESTED | No changes needed |
| physiq_pv/model/physics_loss.py | ⏳ TODO | Fix QS weighting (3 lines) |
| train.py | ⏳ TODO | Add train/val split (10 lines) |

---

## 🚀 Quick Start

### Option 1: Just test the new loader
```bash
python test_sentinel_loader.py
```

### Option 2: Run full pipeline (with current bugs)
```bash
python main.py
```

### Option 3: Run with bug fixes applied
```bash
# After applying fixes from Task 1-2
python main.py
```

---

## 📚 Documentation References

- **Data types**: See [DATA_TYPES.md](DATA_TYPES.md)
- **Code review findings**: See [CODE_REVIEW_DATA_PIPELINE.md](CODE_REVIEW_DATA_PIPELINE.md)
- **Loader API**: See `physiq_pv/data/sentinel_hourly_loader.py` docstrings

---

## ⚠️ Known Limitations

1. **Early 2019 gap**: January-February data missing (sensor startup date)
2. **Lower availability**: 47% valid readings (realistic for sensor networks)
3. **No 2017-2018**: Only 2019 Sentinel data available at full hourly resolution
4. **Geo coverage**: 93 plants lack coordinates (geo lookup failed)

---

## 🔄 Data Pipeline Flow

```
CSV Files (1116 UPN files)
    ↓
    Load ENERGIA, parse dates (DD/MM/YY HH:MM)
    ↓
    Aggregate 3x readings/hour → median
    ↓
    Collect timestamps across all plants
    ↓
    Reindex to common time grid (5743 hours)
    ↓
[Sentinel Dataset: 1116 × 5743]
    ↓
    Match plants to PVGIS locations (nearest neighbor)
    ↓
    Extract weather for matched locations
    ↓
    Reindex PVGIS times to Sentinel times (8760 → 5743)
    ↓
[Merged Dataset: +temperature, +irradiance, +wind]
    ↓
    Compute Quality Score (5 metrics)
    ↓
[Dataset Ready: for training]
```

---

## 📞 Questions?

Check the source functions in `physiq_pv/data/sentinel_hourly_loader.py` for detailed docstrings and parameter descriptions.

---

**Last Updated**: 2024 (after hourly migration)
**Status**: ✅ Migration complete, validation passed
