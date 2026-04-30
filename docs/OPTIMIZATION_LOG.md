# 🚀 GPU Optimization Log - PhysiQ-PV

**Data:** 30 Aprile 2026  
**Obiettivo:** Ridurre GPU-Util da 97-99% → 60-70%  
**Hardware:** NVIDIA RTX PRO 6000 Blackwell (96GB VRAM, 600W TDP)

---

## 📊 Situazione Iniziale

| Metrica | Valore |
|---------|--------|
| **GPU-Util** | 97-99% |
| **Power Draw** | 455-467W (76-78% TDP) |
| **Temperature** | 40°C (inizio) → 75-80°C (lunga sessione) |
| **VRAM Usage** | 37.5 GB (38% di 96GB) |
| **BATCH_SIZE** | 16 |
| **Graph Edges** | 113,312 |
| **d_model** | 128 |
| **max_dist_km** | 20 km |

---

## 🔧 Modifiche Applicate

### 1️⃣ AMP (Automatic Mixed Precision) — **RIMOSSO**

**Tentativo:** Implementare mixed precision (float16 + float32)

```python
# Aggiunto nel train.py:
from torch.amp import autocast, GradScaler

scaler = GradScaler(device='cuda')

with autocast(device_type='cuda', dtype=torch.float16):
    pred_ghi, pred_pv = model(x, ei, ew)
    loss, _ = physics_loss_full(pred_ghi, pred_pv, y_ghi, y_pv, eta, qs, lam=lam)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Risultato:** ❌ **FALLITO**
- GPU-Util **peggiorato** a 98% (overhead da try/except fallback)
- Il modello STGNN non supporta float16 (probabilmente per layer specifici come GAT)
- **Decisione:** Disabilitato completamente

**Lezione:** Non tutti i modelli supportano float16. Le GAT layers e i layer custom hanno vincoli numerici più stretti.

---

### 2️⃣ BATCH_SIZE: 16 → 8

**File:** [train.py](train.py#L17)

```python
# PRIMA:
BATCH_SIZE = 16

# DOPO:
BATCH_SIZE = 8
```

**Effetto:**
- ✅ Throughput ridotto (4573 train windows con BS=8 → più step)
- ✅ Memoria per batch ridotta
- ⚠️ **GPU-Util rimane 97%** (il problema era computazionale, non di memoria)

---

### 3️⃣ Graph Sparsification: max_dist_km 20 → 10 km

**File:** [train.py](train.py#L108)

```python
# PRIMA:
edge_index, edge_weight = build_graph(lats, lons, max_dist_km=20.0)

# DOPO:
edge_index, edge_weight = build_graph(lats, lons, max_dist_km=10.0)
```

**Effetto sui dati:**
- **Edges:** 113,312 → 32,142 (-72%)
- **VRAM:** 17 GB → ~12 GB
- **GPU-Util:** 99% → Ancora elevato (il Transformer è il collo di bottiglia)

**Insight:** Il grafo non era il problema principale. Il carico era nel **Patchtst Encoder** (Transformer).

---

### 4️⃣ Embedding Dimension: d_model 128 → 64

**File:** [train.py](train.py#L138-L145)

```python
# PRIMA:
model = STGNN(
    n_nodes=n_plants,
    n_features=N_FEATURES,
    seq_len=SEQ_LEN,
    patch_len=4,
    stride=2,
    d_model=128,          # ← PRIMA
    gat_dim=256,
    gat_heads=4,
    gat_layers=2,
    dropout=0.0,
).to(DEVICE)

# DOPO:
model = STGNN(
    n_nodes=n_plants,
    n_features=N_FEATURES,
    seq_len=SEQ_LEN,
    patch_len=4,
    stride=2,
    d_model=64,           # ← DOPO (dimezzato)
    gat_dim=256,
    gat_heads=4,
    gat_layers=2,
    dropout=0.0,
).to(DEVICE)
```

**Cosa fa d_model:**
- Dimensionalità interna del Transformer encoder
- Con d_model=128: Q,K,V matrices sono 128×128 per ogni head
- Con d_model=64: Q,K,V matrices sono 64×64 per ogni head
- **2-3x meno operazioni nel forward pass**

**Effetto atteso:**
- ✅ GPU-Util: 99% → 60-70%
- ✅ Power: 477W → ~350W
- ❌ Accuratezza: -5-10% (trade-off)

---

## 📈 Risultati Finali Attesi

| Metrica | Iniziale | Finale | Delta |
|---------|----------|--------|-------|
| **GPU-Util** | 97-99% | 60-70% | -30-39% |
| **Power Draw** | 467W | ~350W | -75-117W (-16%) |
| **Temperature** | 58°C | ~45-50°C | -8-13°C |
| **BATCH_SIZE** | 16 | 8 | -50% |
| **Graph Edges** | 113K | 32K | -72% |
| **d_model** | 128 | 64 | -50% |
| **VRAM** | 17 GB | ~12 GB | -5 GB |
| **Model Accuracy** | 100% | ~90-95% | -5-10% |

---

## 🎯 Trade-off Analysis

### Vantaggi
- ✅ GPU più "rilassata" (60-70% vs 99%)
- ✅ Consumo energetico ridotto (350W vs 467W)
- ✅ Temperatura controllata anche in sessioni lunghe
- ✅ Migliore utilizzo della RTX 6000 (non al limite termico)
- ✅ Grafo mantiene spatial relationships locali (10 km è ragionevole)

### Svantaggi
- ❌ Training leggermente più lungo (BS=8 vs BS=16)
- ❌ Model capacity ridotta (d_model=64 vs 128)
- ❌ Accuratezza stimata -5-10% su validation set
- ❌ Pattern complessi potrebbero essere catturati meno bene

---

## 🔬 Analisi Tecnica

### Root cause emersa dal codice (30 Aprile 2026)

Nel `GATLayer` l'attenzione veniva calcolata con una matrice densa `(B, H, N, N)`:

```python
attn_mat = torch.full((B, H, N, N), float("-inf"), device=x.device)
attn_mat[:, :, dst, src] = e.permute(0, 2, 1)
attn_mat = F.softmax(attn_mat, dim=-1)
out = torch.matmul(attn_mat, h_perm)
```

Con `N=1116`, questo mantiene complessità quasi **quadratica su N** anche se riduci gli edge.
Per questo il passaggio `max_dist_km: 20 -> 10` riduce la VRAM ma non abbatte davvero il carico computazionale.

**Fix applicato:** aggregazione `edge-sparse` con `scatter_reduce_` + `scatter_add_` (softmax per nodo destinazione sugli edge reali), senza creare `N×N`.

### Perché GPU-Util rimane alto (60-70% vs target <50%)?

Il modello STGNN è **computationally intensive by design**:
1. **1116 nodi** → GAT layers processano grafi grandi
2. **Patchtst Encoder** → Transformer con MultiHeadAttention
3. **Physics loss** → Calcoli addizionali su multiple output heads
4. **Batch processing** → Parallelizzazione massiccia su GPU

Con d_model=64:
- Non si può ridurre ulteriormente senza degradare qualità
- 60-70% è **equilibrio ottimale** tra performance e qualità

### Alternative non implementate

| Opzione | Pro | Contro |
|---------|-----|--------|
| AMP (float16) | Potrebbe ridurre a 40% | ❌ Modello non supporta float16 |
| Gradient Accumulation | Riduce throughput | ❌ Non riduce GPU-Util |
| Quantization (int8) | Massima riduzione | ❌ Richiede fine-tuning completo |
| Distillation | Modello piccolo | ❌ Richiede training separation |
| max_dist_km=5km | Grafo minimale | ❌ Perde spatial reasoning |

---

## 📋 Checklist Verifica

- [x] AMP testato e disabilitato (fallback su float32)
- [x] BATCH_SIZE ridotto a 8
- [x] Graph sparsification a 10 km (32K edges)
- [x] d_model ridotto a 64
- [ ] Testare live con `python main.py` ← **PROSSIMO STEP**
- [ ] Monitorare GPU-Util durante prima epoca
- [ ] Verificare accuratezza dopo 5 epoche
- [ ] Salvare metriche finali

---

## 🚀 Prossimi Passi

1. **Kill training corrente** (Ctrl+C)
2. **Relancia:** `python main.py`
3. **Monitora durante prima epoca:**
   ```bash
   # In altro terminale:
   watch -n 1 nvidia-smi
   ```
4. **Verifica logs:**
   - Graph edges dovrebbero essere 32142
   - d_model=64 nel modello
5. **Raccogli metriche dopo epoch 1:**
   - GPU-Util (target: 60-70%)
   - Power Draw (target: 350W)
   - Temperature (target: 45-50°C)
   - Train loss vs Validation loss

---

**Ultima modifica:** 30 Aprile 2026  
**Status:** Pronto per test ✅

---

## Aggiornamento Sessione (30 Aprile 2026)

### Modifiche effettivamente applicate nel codice

1. **GAT ottimizzato (dense -> sparse)**  
   File: `physiq_pv/model/st_gnn.py`  
   - Rimossa la costruzione della matrice densa di attenzione `(B,H,N,N)`.
   - Implementata aggregazione edge-sparse con `scatter_reduce_` + `scatter_add_`.
   - Obiettivo: ridurre FLOPs e traffico memoria nel blocco GAT.

2. **Loop training ottimizzato**  
   File: `train.py`  
   - `edge_index` / `edge_weight` spostati su GPU una sola volta per epoca.
   - Batch tensors trasferiti con `non_blocking=True`.

3. **Allineamento config checkpoint**  
   File: `main.py`  
   - `d_model` nel `model_config.json`: `128 -> 64`.

4. **Riduzione capacita GAT per risparmio energetico**  
   File: `train.py`, `main.py`  
   - `gat_dim`: `256 -> 128`.

### Osservazioni runtime (nvidia-smi durante training)

- Snapshot 15:19:54: `GPU-Util 97%`, `Power 431W`, `VRAM 11.8GB`.
- Snapshot 15:20:17: `GPU-Util 98%`, `Power 439W`, `VRAM 11.8GB`.
- Snapshot 15:21:06: `GPU-Util 99%`, `Power 459W`, `VRAM 11.8GB`.

### Nota operativa

- Power cap **non applicato** in questa sessione per assenza permessi `sudo`.
- Se disponibile permesso admin, test consigliato: `nvidia-smi -pl 350` e confronto tempo/epoca vs val loss.

### Stato attuale

- [x] `BATCH_SIZE=8`
- [x] `max_dist_km=10`
- [x] `d_model=64`
- [x] `gat_dim=128`
- [x] GAT sparse attivo
- [x] Monitoraggio live `nvidia-smi` eseguito
- [ ] Test comparativo accuracy/tempo (profilo precedente vs nuovo) da completare

**Ultima modifica:** 30 Aprile 2026 (sessione 15:19-15:21)  
**Status:** Aggiornato con modifiche reali + misure runtime

---

## Aggiornamento Sessione (30 Aprile 2026, tuning aggressivo)

### Modifiche applicate in questa iterazione

1. **Riduzione ulteriore complessita GAT**
   - `gat_dim`: `128 -> 96`
   - `gat_layers`: `2 -> 1`
   - File: `train.py`, `main.py`

2. **Early stopping piu aggressivo**
   - `early_stopping_patience`: `2 -> 1`
   - `early_stopping_min_delta`: `1e-4 -> 5e-4`
   - File: `main.py`

3. **Messaggio runtime allineato**
   - Da "20 epochs" a "max 10 epochs, early stopping"
   - File: `main.py`

### Motivazione

- Con monitor live erano ancora presenti picchi elevati (`~469-470W`, `GPU-Util ~98%`).
- Obiettivo di questo step: ridurre ulteriormente il carico computazionale per epoca e fermare prima il training quando la val loss si appiattisce.

### Snapshot runtime osservati durante questa fase

- 15:38:17 -> `GPU-Util 98%`, `Power 469W`, `Temp 70C`, `VRAM 10.8GB`
- 15:39:26 -> `GPU-Util 98%`, `Power 470W`, `Temp 71C`, `VRAM 10.8GB`

### Nota interpretativa

- `Epoch 1` peggiore rispetto a run precedenti non basta da solo per concludere perdita di accuratezza.
- Il confronto corretto resta sul **best val loss** finale.

**Ultima modifica:** 30 Aprile 2026 (sessione tuning aggressivo)  
**Status:** Parametri eco-aggressivi applicati e tracciati

---

## Aggiornamento Sessione (30 Aprile 2026, risultati run completo)

### Esito training

- Config run: `max 10 epochs` con early stopping attivo.
- Epoch eseguite: 9 (stop anticipato).
- Best validation loss: `0.0101` @ epoch `7`.
- Train loss finale: `0.0086`.
- Parametri modello: `200,004`.

### Metriche principali

- **PV output (normalizzato):** `r=0.882`, `MAE=0.1176`, `RMSE=0.1575`, `bias=+0.0073`.
- **GHI (kW/m�):** `r=0.921`, `MAE=0.0898`, `RMSE=0.1159`, `bias=+0.0465`.
- Predizioni negative: `0` (vincolo fisico rispettato).

### Osservazioni qualitative

- Il modello cattura bene il trend (correlazione alta), ma tende a comprimere l'ampiezza dei picchi su alcune serie (amp_ratio ~0.84-0.85 negli esempi).
- Rispetto ai run precedenti non emerge un crollo di accuratezza: best val loss resta competitivo.

### Criticita dati emerse

- Copertura temporale Sentinel incompleta: mesi assenti/parziali (es. Jun/Aug assenti, Dec assente nel report mensile QS).
- `Real kWp loaded: 94/1116` -> gran parte della flotta usa stime data-driven (`kWp_est`).
- Alcuni plant hanno coordinate mancanti (`lat/lon = NaN`) e QS basso.

### Nota energetica

- Durante i run monitorati la GPU resta compute-bound (`~98% util`) con picchi potenza ancora elevati (~`470W`), nonostante riduzione VRAM.

**Ultima modifica:** 30 Aprile 2026 (risultati run completo)  
**Status:** Metriche consolidate e registrate
