# PhysiQ-PV — Registro Modifiche

Tutte le modifiche rispetto alla baseline originale (pre-fix).

---

## 1. Fix critici al data pipeline

### 1.1 `pvgis_ref` unità errata — `sentinel_hourly_loader.py:324`

**Problema**: PVGIS `pv_power_output` è in W/kWp nativamente. Il codice non divideva per 1000, quindi `pvgis_ref` risultava in W/kWp invece di kW/kWp.

**Impatto**: soglie `pvgis_ref > 0.1` e `pvgis_ref > 0.25` (usate per day_mask e p99_mask) erano applicate su valori 1000x troppo grandi → quasi tutte le ore classificate come "notte" → `eta_adjusted` calcolato su campioni sbagliati → vincolo fisico `L_physics` distorto.

**Fix**:
```python
# sentinel_hourly_loader.py:324
pvgis_ref_array[i, :] = pv_reindexed['pv_power_output'].values / 1000.0  # W/kWp → kW/kWp
```

**Effetto sui risultati**:
| Metrica | Pre-fix | Post-fix |
|---------|---------|----------|
| PV Pearson r | 0.638 | 0.885 |
| PV MAE | 0.2050 | 0.1117 |
| PV bias | -0.127 | +0.001 |
| GHI Pearson r | 0.858 | 0.926 |
| `eta_adjusted` fleet mean | 0.757 | 0.925 |

---

## 2. Sostituzione feature `pvgis_ref` con geometria solare

### 2.1 Motivazione

`pvgis_ref` come feature di input aveva due problemi:

1. **Dipendenza da sorgente esterna a runtime**: a inferenza (continual learning) PVGIS non è disponibile in real-time. Richiederebbe un proxy (Open-Meteo + pvlib) con distribution shift rispetto ai dati di training.
2. **Ridondanza parziale**: `pvgis_ref ≈ solar_irradiance_poa/1000 × eta_system × (1 - temp_effect)`. L'informazione meteo era già contenuta nelle feature `solar_irradiance_poa` e `temperature_2m`.

### 2.2 Soluzione: geometria solare deterministica

Feature `pvgis_ref` rimossa e sostituita con due feature deterministiche calcolate via pvlib:

```
sin(solar_elevation)   — altezza del sole [0..1], 0=notte, 1=sole a picco
cos(solar_elevation)   — componente complementare [0..1]
```

Entrambe calcolate da `lat`, `lon`, `timestamp` — **nessuna dipendenza da sorgente dati esterna**.

**N_FEATURES**: 5 → 6

**Feature layout attuale** (canali 0-5):
| Idx | Feature | Sorgente | Normalizzazione |
|-----|---------|----------|-----------------|
| 0 | `temperature_2m` | PVGIS (train) / Open-Meteo (inference) | z-score |
| 1 | `solar_irradiance_poa` | PVGIS (train) / Open-Meteo (inference) | z-score |
| 2 | `wind_speed_10m` | PVGIS (train) / Open-Meteo (inference) | z-score |
| 3 | `sin(solar_elevation)` | pvlib (deterministico) | as-is [0,1] |
| 4 | `cos(solar_elevation)` | pvlib (deterministico) | as-is [0,1] |
| 5 | `QS` | calcolato | as-is [0,1] |

**Nota**: `pvgis_ref` rimane nel dataset `xr.Dataset` per uso interno in `eta_adjusted` e `day_mask`. Non è più una feature di input al modello.

### 2.3 File modificati

**`physiq_pv/data/dataset.py`**:
- `N_FEATURES = 6`
- Aggiunta funzione `_solar_geometry(times, lats, lons)` → loop pvlib per-plant
- Piante senza coordinate → fallback fleet-mean lat/lon

**`pyproject.toml`**:
- Aggiunta dipendenza `pvlib>=0.11.0`

**`main.py`**:
- `model_config.json`: `n_features: 6`

---

## 3. Data augmentation durante training

**`train.py` — `_train_epoch()`**:

Perturbazione ±5% moltiplicativa sui canali weather (0,1,2) durante training:

```python
noise = 1.0 + 0.05 * torch.randn(..., 3, device=device)
x = torch.cat([x[..., :3] * noise, x[..., 3:]], dim=-1)
```

Canali 3,4 (geometria, deterministici) e 5 (QS) non perturbati.

**Motivazione**: rende il modello robusto a differenze di scala tra PVGIS (training) e Open-Meteo (inference). Letteratura mostra che perturbazione delle feature meteorologiche migliora transfer cross-domain.

---

## 4. Fix notebook di analisi (`model_results.ipynb`)

| Cella | Problema | Fix |
|-------|----------|-----|
| `d08c19f6` (per-plant table) | `pr_actual = eta_adj * pvgis_p99` (dimensionalmente errato) | `pr_actual = eta_adj` (PR è già adimensionale) |
| `3a3b2890` (QS report mensile) | `qs_fleet_hourly` includeva QS notturni (= 0.0 spurio) → medie mensili contaminate (~0.05-0.25) | `qs_day = np.where(pvgis>0.1, qs_arr, NaN)` — solo ore diurne |
| `0845835b` (spatial QS map) | `np.argsort` con NaN → Plant 896 (all-NaN) appariva come top-1 | `valid_mask = ~np.isnan(qs_plant_mean)` — escludi NaN da ranking |

---

## 5. Documentazione aggiunta

| File | Contenuto |
|------|-----------|
| `docs/DATA_TYPES.md` | Riferimento completo variabili, unità, range, limitazioni dataset |
| `docs/DATA_FLOW.md` | Come PVGIS, Sentinel, GSE partecipano al modello; stima kWp; eta_adjusted |
| `docs/TRAINING_ARCHITECTURE.md` | Architettura completa ST-GNN, parametri, metriche pre/post fix |
| `docs/CONTINUAL_LEARNING.md` | Pipeline online (Open-Meteo + pvlib) vs batch (ERA5); QS senza PVGIS |
| `docs/CHANGES.md` | Questo file |

---

## 6. Cosa NON è cambiato

- Architettura modello (PatchTST + GAT, 2.1M parametri) — invariata
- Formula loss: `L_ghi + L_pv + 0.1×L_physics` — invariata
- `compute_qs()` — invariato (usa ancora `pvgis_ref` dal dataset)
- `eta_adjusted` computation — invariata
- Split stratificato mensile 80/20 — invariato
- Iperparametri: LR=1e-3, batch=16, 20 epoche, AdamW — invariati

---

## 7. Prossimi step

1. Installare pvlib sul server: `uv add pvlib`
2. Pushare modifiche: `git push`
3. Retrainare: `python main.py`
4. Ottenere dati giugno/agosto 2019 dal prof → aggiungere CSV → retrain completo
5. Per continual learning: implementare fetch Open-Meteo + pvlib per inference senza PVGIS
