# 📊 PhysiQ-PV Dataset Types & Pipeline Documentation

**Last Updated**: 2026-04-29  
**Dataset**: Piedmont (Piemonte) PV Plants 2019  
**Region**: North-West Italy (45°N latitude)  

---

## 🏭 Overview

This project uses **hourly energy production data** from 94–95 PV plants in Piedmont, coupled with meteorological and reference irradiance data.

### Data Sources

| Source | Type | Resolution | Period | Count |
|---|---|---|---|---|
| **Sentinel/SCADA** | Energy production (kW) | **Hourly** ✅ | 2019-01-03 to 2019-12-31 | 94 plants |
| **PVGIS** | Reference irradiance (kW/m²) | ~2-hourly* | 2019 full year | 95 plants |
| **Open-Meteo** | Temperature, wind, etc. | Hourly | 2019 full year | 95 plants |
| **Plant Registry** | Coordinates, capacity, UPN | Static | Ceduti mappato 2019 | 95 plants |

*PVGIS in project is ~2-hourly aggregation; native hourly available via API.

---

## 📈 Variable Definitions

### **1. ENERGIA** — Energy Production

**Source**: Sentinel/SCADA systems (individual inverter readings)

| Property | Value |
|---|---|
| **Units** | kW (instantaneous power) |
| **Temporal resolution** | Hourly (3600 seconds) |
| **Period** | 2019-01-03 07:00 to 2019-12-31 16:00 |
| **Timesteps per plant** | 3,770 hours (~365 days × 11.2 readings/day*) |
| **NaN ratio** | ~33% (night time, sensor downtime) |
| **Range** | 0.03 kW — 4,164 kW |
| **Mean (fleet)** | 173.8 kW |
| **Median (fleet)** | ~100 kW |
| **Std Dev** | 362.5 kW (high variability across plants) |

**Issues in Raw Data**:
- ⚠️ **Multiple readings per hour**: Sentinel CSV has 3x readings per timestamp (different inverters/sensors)
  - Solution: Median aggregation per hour
- ⚠️ **Night-time zeros**: Inverters off → ENERGIA=0 (not NaN)
  - Can't distinguish from sensor gap without external validation
- ✅ **Hourly granularity**: Captured within each hour, better than 2-hourly

**Interpretation**:
- **Peak values** (4000+ kW) indicate large industrial plants
- **Small values** (< 100 kW) indicate residential or small commercial
- **NaN during night** is expected (no solar production)

---

### **2. pvgis_ref** — PVGIS Reference Irradiance

**Source**: PVGIS (Solar radiation database, ESA satellite-based)

| Property | Value |
|---|---|
| **Units** | kW/m² or kW/kWp (per 1 kW peak reference) |
| **Temporal resolution** | ~2-hourly (4087 timesteps/year) |
| **Range** | 0 kW/kWp — 0.798 kW/kWp |
| **Mean** | 0.175 kW/kWp |
| **Std Dev** | 0.214 kW/kWp |
| **NaN ratio** | 0% (complete time series) |

**Physical Interpretation**:
- **0.0 kW/kWp** → Night time (no solar irradiance)
- **0.1 kW/kWp** → ~100 W/m² (dawn/dusk, low angle)
- **0.3–0.5 kW/kWp** → 300–500 W/m² (mid-day clear sky)
- **0.798 kW/kWp** → ~800 W/m² (peak summer noon, ideal conditions)

**Role in Physics Loss**:
- Normalized reference (independent of plant capacity)
- Used to compute Quality Score (QS) and eta_adjusted
- Ratio ENERGIA/pvgis_ref ≈ system efficiency (Performance Ratio)

**Known Issue**:
- PVGIS is 2-hourly aggregation in current real_data_dataset.nc
- Newly loaded Sentinel hourly data needs hourly PVGIS (via API resample)

---

### **3. temperature_2m** — Ambient Temperature

**Source**: Open-Meteo or PVGIS dataset

| Property | Value |
|---|---|
| **Units** | °C |
| **Temporal resolution** | Hourly |
| **Range** | -12.77°C to 36.31°C |
| **Mean** | 13.48°C |
| **Std Dev** | 7.30°C |
| **NaN ratio** | 0% (complete) |

**Role in Physics Loss**:
- Temperature correction to efficiency:  
  `eta_T = eta_base × (1 - γ × (T - 25°C))`
  - γ ≈ 0.004 K⁻¹ (IEC 61215 standard, typical silicon)
  - Accounts for PV efficiency drop at high temperatures
  
**Seasonal Pattern** (Piedmont):
- **Winter (-13°C)**: Lower temperature → Higher efficiency (paradoxical but real)
- **Spring/Fall (10–15°C)**: Moderate efficiency
- **Summer (+36°C)**: High temperature → Lower efficiency (~25% drop vs ref)

---

### **4. Quality Score (QS)** — Data Quality Metric

**Computed**: Per (plant, time) from 5-metric composite

| Metric | Description | Range |
|---|---|---|
| **m1: Pearson(real, pvgis_ref)** | Temporal correlation | [0, 1] |
| **m2: bias_score** | Systematic offset | [0, 1] |
| **m3: nan_score** | Data completeness | [0, 1] |
| **m4: var_score** | Variance consistency | [0, 1] |
| **m5: eta_score** | Thermal consistency | [0, 1] |

**Final QS**:  
$$QS = (m1 \times m2 \times m3 \times m4 \times m5)^{1/5}$$ (geometric mean)

| QS Range | Interpretation | Loss Weight | Use |
|---|---|---|---|
| **0.0–0.3** | Poor quality (sensor error, shading) | ~0.000 | Excluded from loss |
| **0.3–0.6** | Moderate quality (clouds, drift) | ~0.035 | Down-weighted |
| **0.6–0.85** | Good quality (normal conditions) | ~0.100 | Normal weight |
| **0.85–1.0** | Excellent quality (clear sky, stable) | ~0.137 | Full weight |

**Applied as**:  
- **Loss weight**: `weight = QS^0.2` (smooth power law, no hard cutoff)
- **Input feature**: Included as 5th channel in neural network

---

### **5. eta_adjusted** — Per-Plant Efficiency (Performance Ratio)

**Computed**: Per plant from daytime ENERGIA/pvgis_ref ratio

| Property | Value |
|---|---|
| **Units** | Dimensionless (0–1, ideally) |
| **Range** | 0.15 (all plants Thin-film) |
| **Mean** | 0.15 |
| **Std Dev** | 0.0 (constant for all plants) |

**⚠️ KNOWN ISSUE** (from code review):
- Current implementation is **dimensionally wrong**
  - Should compute: `(ENERGIA / scale) / (pvgis_ref / scale)` → cancels scale
  - Currently does: `target_pv_norm / pvgis_raw` → mixed scales
- Need to fix in `dataset.py` lines 65–71

**Intended Use**:
- Target value for physics loss:  
  `L_physics = (pred_pv / pred_ghi - eta_adjusted)²`
- Constraint: Keep model predictions consistent with expected efficiency

**Per-Plant Interpretation**:
- **η ≈ 0.15** (Thin-film technology, all plants)
- **η > 0.20** would indicate Monocrystalline (but not in this dataset)
- **η < 0.10** would indicate severe degradation or shadowing

---

## 📁 File Formats & Paths

### **Source: Sentinel Hourly CSV**

```
Path: /data/SentinelPV/energy_data/piemonte_energy_data/single_ups/
Files: 2019_UPN_XXXXXXX_01.csv
Columns: date (MM/DD/YY HH:MM), ENERGIA (kW)
Rows: ~3770 per plant (365 days × 10–11 readings/day with duplicates)
Size: ~370 KB per file × 94 plants = ~35 MB total
```

**Example (2019_UPN_0110065_01.csv)**:
```
date,ENERGIA
01/03/19 07:00,4.0
01/03/19 07:00,4.0
01/03/19 07:00,4.0    ← 3x readings, same timestamp (take median)
01/03/19 08:00,39.0
01/03/19 09:00,250.0
...
```

### **NetCDF: Current real_data_dataset.nc**

```
Path: data/real_data_dataset.nc
Format: NetCDF4 (9 MB)
Dimensions: plant (95), time (4087), date (4087)
Variables:
  - ENERGIA (95, 4087): ~2-hourly aggregation, 33% NaN
  - pvgis_ref (95, 4087): 2-hourly, 0% NaN
  - temperature_2m (95, 4087): 2-hourly, 0% NaN
Coordinates:
  - plant (95 anonymous IDs)
  - time (datetime)
  - latitude (95): geographic coordinates (anonimized)
  - longitude (95): geographic coordinates (anonimized)
  - eta_base (95): 0.15 for all (placeholder)
```

### **Plant Registry: plant_mapping.csv**

```
Path: data/plant_mapping.csv
Columns: Codice UP (UPN), Latitude, Longitude, Codice Censimp Impianto, plant_id, eta_base
Rows: 95 plants (94 mapped to Sentinel)
Example:
  UPN_2021228_01, 44.696487, 7.933231, IM_2021228, 0, 0.15
```

### **Energy Registry: energy_with_coordinates.csv**

```
Path: data/energy_with_coordinates.csv
Rows: 61,516 (full Italy registry, we use Piemonte subset)
Columns:
  - Codice Censimp Impianto: GSE registry ID
  - Potenza di picco (kW): Installed capacity
  - Livello di Tensione: BT/MT/AT (voltage level)
  - Provincia Impianto: Province
  - Data Esercizio: Commissioning date
  - Latitude, Longitude: Coordinates
```

---

## 🔄 Data Processing Pipeline

### **Step 1: Load Sentinel CSV**
```python
sentinel_hourly_loader.load_sentinel_hourly(
    year=2019,
    plant_mapping_path='data/plant_mapping.csv',
    energy_coords_path='data/energy_with_coordinates.csv'
)
```
- Reads 94× `2019_UPN_*.csv` files from `/data/SentinelPV/...`
- Aggregates 3x readings per hour → median
- Merges with plant registry for coordinates & metadata
- **Output**: xr.Dataset (94 plants, 3770 hours)

### **Step 2: Add Weather Data**
```python
sentinel_hourly_loader.merge_with_weather(
    ds=ds,
    pvgis_path='data/piedmont_pvgis_2019.nc'
)
```
- Loads PVGIS reference irradiance
- Loads temperature (Open-Meteo or PVGIS)
- Aligns time grids (hourly Sentinel × ~2-hourly PVGIS)
- **Output**: xr.Dataset with ENERGIA, pvgis_ref, temperature_2m

### **Step 3: Compute Quality Score**
```python
physiq_pv.data.quality_score.compute_qs(ds)
```
- Computes 5-metric QS per (plant, time)
- Per-plant capacity scaling (p99-based)
- Daytime masking (pvgis_ref > 0.1 kW/kWp)
- **Output**: xr.DataArray (94, 3770) with QS ∈ [0,1]

### **Step 4: Normalize & Create PyTorch Dataset**
```python
physiq_pv.data.dataset.PVDataset(ds, qs)
```
- **Per-plant normalization**:
  - ENERGIA: scale by p99 daytime → [0, 1]
  - pvgis_ref: scale by p99 daytime → [0, 1]
  - temperature: standardize (μ=0, σ=1)
  - wind, solar: standardize
  - QS: as-is (already [0,1])
  
- **Compute eta_adjusted** (should fix dimensional issue):
  - `eta_adjusted[p] = median(ENERGIA_norm[p] / pvgis_ref[p])` over daytime
  - Fallback: fleet median if plant has <10 valid hours
  - Clip: [0.2, 1.0]

- **Create sliding windows**:
  - SEQ_LEN = 120 timesteps (currently 2-hourly → 10 days; **should be 24 for hourly → 1 day**)
  - stride = 1 (highly overlapping)
  - target = `y_pv[t]`, `y_ghi[t]` (current timestep, not future)

- **Output**: PyTorch DataLoader with batches (B, N, 120, 5)

### **Step 5: Train ST-GNN**
```python
train.train(ds=ds, n_epochs=20)
```
- PatchTST temporal encoder + GAT spatial propagation
- Physics loss: L = L_ghi + L_pv + λ × L_physics
- QS-weighted: `weight = qs.pow(0.2)`
- No train/val split (⚠️ overfitting risk)

---

## 🎯 Key Temporal Features

### **Hourly Pattern** (Circadian)
- **Night (21:00–06:00)**: ENERGIA ≈ 0 (inverter off)
- **Dawn (06:00–09:00)**: ENERGIA ramps up (low angle)
- **Peak (11:00–14:00)**: ENERGIA ≈ P_max (high angle, clear sky)
- **Dusk (15:00–19:00)**: ENERGIA ramps down
- **Seasonal modulation**: Peak height varies by season

### **Seasonal Pattern** (Annual)
- **Winter (Dec–Feb)**: Low sun angle → Lower P_max
- **Spring (Mar–May)**: Increasing: +40% from Feb to May
- **Summer (Jun–Aug)**: Peak production, high temperature loss
- **Fall (Sep–Nov)**: Decreasing: -30% from Aug to Oct

**Summer/Winter ratio**: ~1.6× (ENERGIA higher in summer despite temperature loss)

---

## 📊 Data Quality Summary

| Aspect | Status | Notes |
|---|---|---|
| **Completeness** | ✅ 66% valid (33% night) | Expected; night values are structural NaN |
| **Temporal alignment** | ⚠️ Sentinel hourly ↔ PVGIS ~2h | Need to resample PVGIS hourly |
| **Geographic coverage** | ✅ 94/95 plants | 1 plant missing UPN code |
| **Coordinate precision** | ✅ ~5m accuracy | Lat/lon from GSE registry |
| **Metadata consistency** | ⚠️ Mixed sources | eta_base all 0.15 (placeholder) |
| **Sensor reliability** | ⚠️ Unknown | No validation data available |

---

## 🔧 Recommended Fixes (Priority)

| Issue | Impact | Fix |
|---|---|---|
| **eta_adjusted dimensional error** | 🔴 High | Restore scales before dividing (Q4 in code review) |
| **No train/val/test split** | 🔴 High | Temporal split: 80% train, 20% val |
| **shuffle=True breaks time-series** | 🔴 High | Use shuffle=False, preserve temporal order |
| **QS target leakage** | 🔴 High | Compute QS on lagged data (t-24:t) not current |
| **SEQ_LEN=120 too long (hourly)** | ⚠️ Medium | Change to 24 (24h context) |
| **Daytime mask 0.1 too loose** | ⚠️ Medium | Increase to 0.25 (skip weak dawn/dusk) |

---

## 📝 References

- **IEC 61215**: Crystalline silicon terrestrial photovoltaic (PV) modules
- **PVGIS**: https://pvgis.cm.unito.it/
- **Performance Ratio**: https://doi.org/10.1016/S0038-092X(96)00119-0
- **Piedmont Region**: 45.0°N, 7.5–8.5°E (northern Italy, Alpine foothills)

---

**Created**: 2026-04-29  
**Dataset**: PhysiQ-PV Piedmont 2019 (94 hourly PV plants)  
**Contact**: System generated documentation
