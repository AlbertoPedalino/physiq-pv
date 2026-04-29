# PhysiQ-PV — Training Architecture

**Dataset**: Piedmont 2019, 1116 PV plants, hourly  
**Model**: ST-GNN (PatchTST + GAT)  
**Entry point**: `main.py` → `train.py`

---

## 1. Data Pipeline

### 1.1 Input Data
| Source | File | Variabile | Unità |
|--------|------|-----------|-------|
| Sentinel/SCADA | `2019_UPN_*.csv` | `ENERGIA` | kW |
| PVGIS | `piedmont_pvgis_2019.nc` | `pvgis_ref` | kW/kWp |
| PVGIS | `piedmont_pvgis_2019.nc` | `solar_irradiance_poa` | W/m² |
| PVGIS | `piedmont_pvgis_2019.nc` | `temperature_2m` | °C |
| PVGIS | `piedmont_pvgis_2019.nc` | `wind_speed_10m` | m/s |
| GSE Registry | `energy_with_coordinates.csv` | `Potenza di picco (kW)` | kWp |

**Loader**: `sentinel_hourly_loader.load_sentinel_hourly()`
- 1116 file CSV caricati, 3 letture/ora aggregate via mediana
- Match piante → griglia PVGIS via nearest-neighbor (lat/lon)
- Output: `xr.Dataset` (1116 piante × 5743 ore)

### 1.2 Normalizzazione (`dataset.py`)

**Feature input** `x` — normalizzazione z-score per-feature globale:
```
x = (x - mean) / (std + 1e-6)
```
Applicata a: `temperature_2m`, `solar_irradiance_poa`, `wind_speed_10m`, `pvgis_ref`  
QS incluso as-is (già in [0,1]).

**Target PV** — normalizzazione per-pianta:
```
target_pv_norm[p] = clip(ENERGIA[p] / pv_scale[p], 0.0, 1.5)
pv_scale[p]       = p99(ENERGIA_daytime[p])   # p99_mask: pvgis_ref > 0.1
```
Clip a 1.5 elimina spike da sensori difettosi.

**Target GHI**:
```
target_ghi = solar_irradiance_poa / 1000.0   [W/m² → kW/m²]
```

### 1.3 Efficienza per pianta — `eta_adjusted`
Performance Ratio stimato dai dati:
```
pvgis_norm = pvgis_ref / p99(pvgis_ref_daytime)
ratio      = target_pv_norm / pvgis_norm         # daytime only (pvgis_ref > 0.25)
eta_adjusted[p] = median(ratio)                  # ≈ 0.757 fleet mean
```
Clip finale: `[0.1, 1.0]`  
Fallback: mediana fleet per piante con < 50 campioni diurni validi.

---

## 2. Quality Score (`quality_score.py`)

**Formula**: media geometrica di 5 metriche, `QS ∈ [0,1]`
```
QS = (m1 × m2 × m3 × m4 × m5)^(1/5)
```

| Metrica | Formula | Misura |
|---------|---------|--------|
| `m1` corr_score | `Pearson(real, pvgis_ref)` rolling 720h | Forma profilo giornaliero |
| `m2` bias_score | `1 - |mean(real-ref)| / mean(ref)` | Offset sistematico |
| `m3` nan_score | `1 - nan_fraction` | Completezza dati |
| `m4` var_score | `clip(std_real / std_ref, 0, 1)` | Sensore bloccato |
| `m5` eta_score | `1 - mean(max(0, 1 - PR/eta_T))` | Consistenza fisica termica |

**Notte**: `pvgis_ref < 0.1` → QS = NaN → convertito a 0.0 in `dataset.py`  
**Uso in loss**: `weight = QS^0.2` — pesa i campioni senza azzerarli

---

## 3. Grafo Spaziale (`graph_builder.py`)

**Tipo**: grafo non diretto, archi per distanza Haversine  
**Soglia**: `max_dist_km = 20.0` km  
**Peso arco**: `w = 1 / dist_km` (piante vicine = accoppiamento forte)  
**Fallback**: se nessun arco → nearest-neighbor garantito

```
1116 nodi, ~113,312 archi (con max_dist_km=20)
```

---

## 4. Modello ST-GNN (`st_gnn.py`)

### Architettura
```
Input (B, N, 24, 5)
    ↓
PatchTSTEncoder  — encoding temporale channel-independent
    ↓
Linear Projection + GELU + LayerNorm  → (B, N, 256)
    ↓
GATLayer × 2  — propagazione spaziale su grafo
    ↓
Head GHI: Linear(256→128) + GELU + Linear(128→1) + softplus → pred_ghi (B, N)
Head PV:  Linear(256→128) + GELU + Linear(128→1) + softplus → pred_pv  (B, N)
```

### 4.1 PatchTST Encoder (`patchtst_encoder.py`)

| Parametro | Valore | Significato |
|-----------|--------|-------------|
| `seq_len` | 24 | 24 ore di contesto |
| `patch_len` | 4 | patch di 4 ore |
| `stride` | 2 | sovrapposizione 50% |
| `n_patches` | 11 | `(24-4)//2 + 1` |
| `d_model` | 128 | dimensione embedding |
| `n_heads` | 4 | teste attention |
| `n_layers` | 2 | layer transformer |
| `dropout` | 0.0 | disabilitato (abilita flash SDP) |

**Channel-independent**: ogni feature processata separatamente.  
**Output**: `(B×N, n_features × d_model)` = `(B×N, 640)`

### 4.2 GAT Layer

| Parametro | Valore |
|-----------|--------|
| `gat_dim` | 256 |
| `gat_heads` | 4 |
| `gat_layers` | 2 |
| `dropout` | 0.0 |

Attention score scalato per `log(1 + edge_weight)` — piante vicine pesano di più.  
Residual connection + LayerNorm per stabilità.

### Parametri totali modello
`~2.1M` parametri

---

## 5. Loss Function (`physics_loss.py`)

```
L = L_ghi + L_pv + λ × L_physics

L_ghi     = mean(weight × (pred_ghi - true_ghi)²)
L_pv      = mean(weight × (pred_pv  - true_pv)²)
L_physics = mean(weight × (pred_pv / pred_ghi - eta_adjusted)²)

weight    = QS^0.2
λ         = 0.1
```

**`L_physics`** forza la consistenza fisica: il rapporto produzione/irraggiamento predetto deve approssimare il PR stimato per pianta.  
**`weight = QS^0.2`**: funzione potenza smooth — QS=0 → weight=0, QS=1 → weight=1, QS=0.5 → weight≈0.87.

---

## 6. Training Loop (`train.py`)

### Split dataset
**Stratified monthly split** — 80% di ogni mese → train, 20% → val.  
Garantisce tutte le stagioni in entrambi i set.  
Implementato via `torch.utils.data.Subset` sugli indici delle finestre.

```
Train: 4573 finestre  (80% × 12 mesi)
Val:   1145 finestre  (20% × 12 mesi)
```

### Iperparametri

| Parametro | Valore |
|-----------|--------|
| `BATCH_SIZE` | 16 |
| `LR` | 1e-3 |
| `weight_decay` | 1e-4 |
| `optimizer` | AdamW |
| `n_epochs` | 20 |
| `lam` (λ physics) | 0.1 |
| `shuffle` (train) | True |
| `num_workers` | 4 |
| `pin_memory` | True |

### Best model
Salvato il checkpoint con **minima val loss** — ricaricato alla fine del training.

### Continual learning
`ReplayBuffer(capacity=1000)` + `QualityGatedUpdater` — struttura per online loop (attualmente in standby).

---

## 7. Configurazione GPU

**GPU**: NVIDIA RTX PRO 6000 Blackwell (94.97 GB VRAM)  
**Flash SDP**: abilitato (`dropout=0.0` rimuove il limite 65535 batch)  
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — riduce frammentazione memoria

---

## 8. Output

```
checkpoints/
  model.pt           — state dict best val epoch
  loss_history.json  — {"train": [...], "val": [...]}
  model_config.json  — iperparametri architettura
```

---

## 9. Metriche

### Post-fix (run corrente — tutti i fix applicati)

| Metrica | GHI | PV |
|---------|-----|----|
| Pearson r | 0.926 | 0.885 |
| MAE | 0.0791 | 0.1117 |
| bias | +0.036 | +0.001 |
| amp_ratio | — | 0.85 |

Best val loss: **0.0097** @ epoch 18 (20 epoche totali).  
n campioni scatter: 3,061,186 (solo ore diurne).

Fix applicati: `patch_len` 1→4 (11 patch), `pvgis_ref` W/kWp→kW/kWp (`/1000`), `eta_adjusted` ricalcolato (~0.757 fleet mean).

### Baseline pre-fix (storico)

| Metrica | GHI | PV |
|---------|-----|----|
| Pearson r | 0.858 | 0.638 |
| MAE | 0.1416 | 0.2050 |
| bias | +0.094 | -0.127 |
| amp_ratio | — | 0.79 |
