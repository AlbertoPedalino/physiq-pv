# PhysiQ-PV — Data Reference

**Dataset**: Piedmont (Piemonte) PV plants, 2019  
**Region**: North-West Italy, 44.2°–46.1°N, 6.9°–9.0°E  
**Last updated**: 2026-04-29

---

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| Plants | 1,116 UPN sites |
| Timesteps | 5,743 hours |
| Period | 2019-03-01 to 2019-12-31 |
| Resolution | Hourly |
| Data availability | ~47% (structural NaN at night) |
| Plants with coordinates | 1,023 / 1,116 (91.7%) |

**Note**: January–February 2019 absent from Sentinel CSV files (sensor start date).

---

## 2. Data Sources

| Source | File/Path | Variables | Units |
|--------|-----------|-----------|-------|
| Sentinel/SCADA | `/data/SentinelPV/.../single_ups/2019_UPN_*.csv` | `ENERGIA` | kW |
| PVGIS | `data/piedmont_pvgis_2019.nc` | `pvgis_ref` | kW/kWp |
| PVGIS | `data/piedmont_pvgis_2019.nc` | `solar_irradiance_poa` | W/m² |
| PVGIS | `data/piedmont_pvgis_2019.nc` | `temperature_2m` | °C |
| PVGIS | `data/piedmont_pvgis_2019.nc` | `wind_speed_10m` | m/s |
| GSE Registry | `data/energy_with_coordinates.csv` | `Potenza di picco (kW)` | kWp |
| Plant mapping | `data/plant_mapping.csv` | lat, lon, `Codice UP` | — |

---

## 3. Variables

### 3.1 ENERGIA — Energy Production

- **Source**: Sentinel/SCADA per-inverter readings
- **Units**: kW (instantaneous power)
- **Raw format**: 3 readings per hour (separate sensors) → aggregated via median
- **Range**: 0 – ~5,200 kW (varies by plant size)
- **Fleet mean**: ~156 kW, median ~39 kW, std ~330 kW (high variance across plants)
- **NaN**: ~53% total (night: inverter off; sensor gaps: indistinguishable from night without PVGIS cross-check)
- **Night**: ENERGIA = 0 or NaN when `pvgis_ref = 0`

**Normalization in training**:
```
pv_scale[p]       = p99(ENERGIA_daytime[p])   # pvgis_ref > 0.1
target_pv_norm[p] = clip(ENERGIA[p] / pv_scale[p], 0.0, 1.5)
```
Clip to 1.5 removes sensor spikes. After normalization target ∈ [0, ~1].

---

### 3.2 pvgis_ref — PVGIS Reference PV Output

- **Source**: PVGIS satellite model, 1 kWp reference system
- **Units**: kW/kWp (production per unit peak capacity)
- **Range**: 0.0 – ~0.80 kW/kWp
- **NaN**: 0% (complete time series)
- **Physical meaning**:
  - 0.0 → night
  - 0.1 → ~100 W/m² dawn/dusk
  - 0.3–0.5 → mid-day, partial cloud
  - ~0.80 → peak summer clear-sky noon

**Role**: Reference for normalization, eta_adjusted computation, Quality Score, GHI target.  
**Spatial assignment**: nearest-neighbor match from 1,149 PVGIS grid points to each plant lat/lon.

---

### 3.3 solar_irradiance_poa — Plane-of-Array Irradiance

- **Source**: PVGIS
- **Units**: W/m²
- **Target in training**: `target_ghi = solar_irradiance_poa / 1000.0` [kW/m²]

---

### 3.4 temperature_2m — Ambient Temperature

- **Source**: PVGIS
- **Units**: °C
- **Range**: –13°C to +36°C (Piedmont annual)
- **Role**: Input feature (z-score normalized). Captures temperature-efficiency coupling (PV efficiency drops ~0.4%/K above 25°C).

---

### 3.5 wind_speed_10m — Wind Speed

- **Source**: PVGIS
- **Units**: m/s
- **Role**: Input feature (z-score normalized). Minor effect on panel cooling.

---

## 4. Derived Quantities

### 4.1 eta_adjusted — Per-Plant Performance Ratio

Estimated from data during training (`dataset.py`):

```
pvgis_norm    = pvgis_ref[p] / p99(pvgis_ref_daytime[p])
ratio         = target_pv_norm[p] / pvgis_norm        # daytime only: pvgis_ref > 0.25
eta_adjusted[p] = median(ratio)
```

Clip: `[0.1, 1.0]`  
Fleet mean: ~0.757  
Fallback: fleet median for plants with < 50 valid daytime samples (or no real kWp).

**Interpretation**: dimensionless Performance Ratio — fraction of reference irradiance actually converted to electricity by each plant. Encodes orientation, shading, degradation, inverter efficiency.

---

### 4.2 Quality Score (QS)

Computed per (plant, time) in `quality_score.py`:

```
QS = (m1 × m2 × m3 × m4 × m5)^(1/5)    QS ∈ [0, 1]
```

| Metric | Formula | Captures |
|--------|---------|----------|
| m1 corr_score | Pearson(real, pvgis_ref) rolling 720h | Daily shape correlation |
| m2 bias_score | `1 - |mean(real-ref)| / mean(ref)` | Systematic offset |
| m3 nan_score | `1 - nan_fraction` | Data completeness |
| m4 var_score | `clip(std_real / std_ref, 0, 1)` | Frozen sensor detection |
| m5 eta_score | `1 - mean(max(0, 1 - PR/eta_T))` | Thermal physics consistency |

Night (pvgis_ref < 0.1) → QS = NaN → set to 0.0 in dataset.py.

Fleet statistics: mean QS ≈ 0.652, median ≈ 0.737, valid (non-NaN) ≈ 49%.

**Use in training**:
- Loss weight: `weight = QS^0.2`
- 5th input feature to model (as-is, already ∈ [0, 1])

---

## 5. Sentinel CSV Format

```
Path:    /data/SentinelPV/energy_data/piemonte_energy_data/single_ups/
Pattern: 2019_UPN_XXXXXXX_01.csv
Columns: date (DD/MM/YY HH:MM),  ENERGIA (kW)
```

Example:
```
date,ENERGIA
03/01/19 07:00,4.0
03/01/19 07:00,4.0
03/01/19 07:00,4.0   ← 3 readings same hour → take median
03/01/19 08:00,39.0
```

---

## 6. Plant Registry Files

### plant_mapping.csv
```
Columns: Codice UP, Latitude, Longitude, plant_id, eta_base
Rows:    1,116 plants (UPN ↔ coordinates mapping)
```

### energy_with_coordinates.csv
```
Columns: Codice Censimp Impianto, Potenza di picco (kW), Latitude, Longitude,
         Provincia Impianto, Data Esercizio, Livello di Tensione
Rows:    61,516 (full Italy GSE registry — Piemonte subset used)
```
`Potenza di picco (kW)` = installed peak capacity (kWp). Used as real kWp where available.

---

## 7. Loading Pipeline

```python
# Step 1 — Sentinel CSV → xr.Dataset
ds = load_sentinel_hourly(
    sentinel_dir="/data/SentinelPV/energy_data/piemonte_energy_data/single_ups",
    year=2019,
    plant_mapping_path="data/plant_mapping.csv",
    energy_coords_path="data/energy_with_coordinates.csv",
)
# Output: ENERGIA (1116, 5743), coords: plant_id, latitude, longitude, eta_base

# Step 2 — Add weather
ds = merge_with_weather(ds, pvgis_path="data/piedmont_pvgis_2019.nc")
# Adds: temperature_2m, solar_irradiance_poa, wind_speed_10m, pvgis_ref

# Step 3 — Quality Score
qs = compute_qs(ds)    # xr.DataArray (1116, 5743), ∈ [0, 1]

# Step 4 — PyTorch Dataset
dataset = PVDataset(ds, qs, kwp=kwp)
# Yields: (x, y_ghi, y_pv, qs, eta)
#   x      (N=1116, seq_len=24, C=5)
#   y_ghi  (N,)   — solar_irradiance_poa / 1000
#   y_pv   (N,)   — clip(ENERGIA / pv_scale, 0, 1.5)
#   qs     (N,)   — quality score at prediction step
#   eta    (N,)   — eta_adjusted per plant
```

---

## 8. Seasonal and Temporal Patterns

### Daily
- Night (21:00–06:00): ENERGIA ≈ 0, pvgis_ref = 0
- Dawn/dusk (06:00–09:00, 16:00–19:00): ramp up/down
- Peak (11:00–14:00): max production, highest pvgis_ref

### Annual (Piedmont)
- Winter (Dec–Feb): low sun angle, short days — absent from dataset (data starts March)
- Spring (Mar–May): increasing production
- Summer (Jun–Aug): peak production, high temperature → slight efficiency loss
- Fall (Sep–Nov): decreasing, October onwards significant drop

Summer/winter ratio ≈ 1.6× in production.

---

## 9. Known Limitations

| Limitation | Detail |
|------------|--------|
| Jan–Feb missing | Sentinel files start 2019-03-01 |
| ~53% NaN | Expected: night + sensor gaps combined |
| 93 plants without coordinates | Default eta_base=0.15, no PVGIS match |
| kWp available for subset | `energy_with_coordinates.csv` covers most Piemonte plants |
| PVGIS ≠ ground truth | Satellite model; may underestimate winter irradiance (Piedmont fog) |
