# PhysiQ-PV — Model Reference

Riferimento tecnico unificato: tipi di dato, pipeline, architettura, loss, Quality Score, glossario parametri.

## 1. Dataset

| Campo | Valore |
|---|---|
| Area | Piemonte, Italia |
| Risoluzione | oraria |
| Impianti | 1116 UPN |
| Anno | 2019 |
| Produzione | Sentinel/SCADA |
| Meteo | PVGIS reanalysis (ERA5) o fallback pvlib clear-sky |

## 2. Variabili raw

Shape comune: `(plant, time)`.

| Variabile | dtype | Unità | Sorgente | Note |
|---|---|---|---|---|
| `ENERGIA` | `float64` | kW | Sentinel CSV | Mediana se più letture/ora |
| `solar_irradiance_poa` | `float64` | W/m² | NetCDF PVGIS | `/1000` → kW/m² |
| `temperature_2m` | `float64` | °C | NetCDF PVGIS | Correzione termica |
| `wind_speed_10m` | `float64` | m/s | NetCDF PVGIS | Solo feature |
| `lat`, `lon` | `float64` | gradi | `energy_with_coordinates.csv` | Grafo + geometria solare |
| `time` | `datetime64[ns]` | UTC orari | indice allineato | |
| `plant` | `int` | 0..N-1 | `plant_mapping.csv` | |
| `eta_base` | `float64` | adim | seed PR | Riscritto in `compute_qs` |

## 3. Variabili derivate

| Variabile | Calcolata in | Formula |
|---|---|---|
| `sin/cos_solar_elev` | `PVDataset` | pvlib da `lat/lon/time` |
| `ghi_cs` | `PVDataset` | pvlib Ineichen + Linke turbidity climatology |
| `pv_scale[p]` | `PVDataset` | `p99(ENERGIA[p] daytime)` |
| `solar_p99[p]` | `PVDataset` | `p99(solar_irradiance_poa[p]/1000 daytime)` |
| `eta_adjusted[p]` | `PVDataset` | WLS through origin (peso=GHI), clip `[0.1, 0.98]` |
| `capacity_scale[p]` | `compute_qs` | `p99(ENERGIA_day) / p99(ref_day)` |
| `eta_T[p, t]` | `compute_qs` | `eta_base * (1 - 0.004*(T-25))` |
| `m1..m5[p, t]` | `compute_qs` | rolling 720h (vedi sezione 7) |
| `pv_lag[t, p]` | `PVDataset` | `target_pv_norm[t]`, slice causale `t-seq_len:t` |

## 4. Pipeline end-to-end

```text
[2019_UPN_*.csv]      [piedmont_pvgis_2019.nc]      [plant_mapping.csv]
       │                       │              [energy_with_coordinates.csv]
       ▼                       ▼                       ▼
load_sentinel_hourly()    merge_with_weather()      load_kwp() (opz)
       │                       │                       │
       └──── xr.Dataset ───────┘                       │
                  │                                    │
                  ▼                                    │
          _normalize_dataset()                         │
                  │                                    │
                  ▼                                    │
            compute_qs() → QS DataArray + m1..m5       │
                  │                                    │
                  ▼                                    ▼
              PVDataset(ds, m_components, kwp=kwp, eta_max=0.98)
                  │
                  ▼
       (x, y_ghi, y_pv, eta, ghi_cs)
                  │
                  ▼
                STGNN
                  │
                  ▼
        pred_ghi, pred_pv
```

### Step 1 — `load_sentinel_hourly`

Legge tutti CSV `2019_UPN_*.csv`, allinea a indice orario comune, aggrega con mediana se multiple letture/ora. Output: `xr.Dataset` con `ENERGIA(plant, time)`.

### Step 2 — `merge_with_weather`

Apre NetCDF PVGIS, matching spaziale (cella più vicina via lat/lon), allineamento temporale. Aggiunge `solar_irradiance_poa`, `temperature_2m`, `wind_speed_10m`. Fallback pvlib clear-sky se NetCDF manca.

### Step 3 — `_normalize_dataset` (in `main.py`)

Aggiunge `eta_base` come seed PR. Imposta attributi (`pv_scale_method`).

### Step 4 — `compute_qs`

Calcola `m1..m5` rolling 720h + QS aggregato `(m1·m2·m3·m4·m5)^0.2` per `(plant, time)` su ore diurne. Notte gestita separatamente (vedi sezione 7).

### Step 5 — `PVDataset`

- Geometria solare + `ghi_cs` via pvlib Ineichen (Linke turbidity climatology, fallback simplified_solis)
- `pv_scale[p]`, `solar_p99[p]`, `eta_adjusted[p]` per impianto
- `pv_lag` causale (slice `feats[t-seq_len:t]` espone solo `target_pv_norm[t-seq_len..t-1]`)
- z-score globale su meteo
- Finestre temporali `seq_len=24`
- Output 5-tuple per sample

### Step 6 — `STGNN.forward`

Vedi sezione 5.

## 5. Architettura

```text
(B, N, 24, 11)
    → PatchTSTEncoder        encoder temporale, channel-independent
    → proiezione lineare → GELU → LayerNorm
    → GAT (1 layer, geographic graph)  edge ≤ 20km, weight = 1/dist_km
    → dual head:
        head_ghi → kt = sigmoid(·) * 1.2 → pred_ghi = kt * ghi_cs
        head_pv  → softplus(·) → pred_pv
```

Configurazione operativa (`train.py`):

| Parametro | Valore |
|---|---|
| `seq_len` | 24 |
| `patch_len` | 4 |
| `stride` | 2 |
| `d_model` | 64 |
| `gat_dim` | 96 |
| `gat_heads` | 4 |
| `gat_layers` | 1 |
| `dropout` | 0.0 |

`KT_MAX = 1.2` (snow albedo edge): bound fisico hard sul clear-sky index.

Quando `ghi_cs ≈ 0` (notte), `pred_ghi → 0` automaticamente. `pred_pv` rimane non-negativo via softplus.

Grafo: `max_dist_km=10.0` (in `build_graph`), edge_weight=`1/dist_km`. ~113k edge totali su 1116 impianti.

## 6. Input al modello

Shape sample: `(N, 24, 11)`. Vedi tabella in `README.md` per descrizione canali.

Note importanti:
- `pv_lag` (canale 10) è il segnale autoregressivo dominante. Slice causale `feats[t-seq_len:t]` espone solo `target_pv_norm[t-seq_len..t-1]`.
- `m1..m5` sono i 5 componenti separati del QS, non l'aggregato.
- Nessun data leakage: `m1..m5` usano rolling 720h fino a `t-1`.

Target:

```text
y_ghi = solar_irradiance_poa / 1000.0          [kW/m²]
y_pv  = clip(ENERGIA / pv_scale, 0.0, 1.5)    [normalizzato]
```

Ore diurne: `sin_elev > 0.05` AND `solar_irradiance_poa/1000 > 0.03`.

`pv_scale[p]` = p99 produzione diurna. `solar_p99[p]` = p99 irradianza diurna.

## 7. Quality Score

`QS ∈ [0, 1]` per `(plant, time)` calcolato in `physiq_pv/data/quality_score.py`.

### Riferimento fisico

```text
ref[p, t] = (solar_irradiance_poa[p, t] / 1000) * capacity_scale[p]
capacity_scale[p] = p99(ENERGIA_day[p]) / p99(ref_raw_day[p])
```

`ref` ha unità `kW`, comparabile direttamente con `ENERGIA`. Non usa `pvgis_ref` come reference power.

### `eta_base` (riscritto in `compute_qs`)

```text
eta_base[p] = median(ENERGIA[p] / ref[p])  daytime
              clip([0.1, 2.0])
```

PR mediano stimato dai dati. `eta_T[p, t] = eta_base[p] * (1 - 0.004*(T-25))` corregge per temperatura (IEC 61215).

### Definizione daytime in QS

```text
daytime[p, t]  ⟺  solar_irradiance_poa[p, t] / 1000 > 0.05
```

Notte:
- `irr ≈ 0` AND `ENERGIA ≤ 1% di p99(ENERGIA_day)` → QS = 1.0
- `irr ≈ 0` AND `ENERGIA > soglia` → QS = 0.0 (produzione in buio = guasto)
- `0 < irr < 0.05` (alba/tramonto) → QS = NaN

### Formula

**Step 1 — Aggregazione raw (compute_qs):**

```text
QS_raw = (m1 * m2 * m3 * m4 * m5) ^ 0.2
```

Media geometrica → "AND fuzzy". Una metrica a zero tira giù QS. Window 720h (≈ 30gg), `min_periods = 180` (= window/4).

**Step 2 — Shrinkage bayesiano (apply_qs_shrinkage):**

```text
n_valid = count(ENERGIA finite & >0) in window 720h
conf    = sigmoid((n_valid - n0) / scale)             # n0=360, scale=90
qs_prior = median_fleet(QS_raw)                       # ~ 0.74
QS      = conf * QS_raw + (1 - conf) * qs_prior
```

Sample con finestra rolling sparsa (low confidence) → spinti verso prior fleet-median, **non verso zero**. Distingue "non so" da "qualità bassa". Nessun discard.

QS finale è quello che entra nel binning diagnostico, mappa spaziale, framework CL gating. Le feature m1..m5 al modello restano invariate (canali 5..9 sono i componenti grezzi, non l'aggregato).

### Componenti m1..m5

| Metrica | Formula | Cosa rileva |
|---|---|---|
| `m1` corr_score | `clip(Pearson(real, ref), 0, 1)` rolling 720h | Coupling temporale produzione-sole |
| `m2` bias_score | `clip(1 - |mean(real-ref)| / mean(ref), 0, 1)` rolling | Bias sistematico medio |
| `m3` nan_score | `1 - nan_fraction(real)` rolling | Disponibilità dato SCADA |
| `m4` var_score | `clip(std(real)/std(ref), 0, 1)` asimmetrico | Sensore stuck su valore costante |
| `m5` eta_score | `clip(1 - mean(max(0, 1 - PR_obs/eta_T)), 0, 1)` rolling | Sottoproduzione vs efficienza attesa |

`m4` clip asimmetrico: `var_ratio > 1` → 1 (no penalità), `var_ratio < 1` abbassa.

`m5` penalizza solo quando `PR_obs < eta_T` (sottoproduzione cronica).

### Uso operativo del QS nel modello

I 5 componenti `m1..m5` entrano come feature canali 5..9.

**QS aggregato `(m1·m2·m3·m4·m5)^0.2` NON è usato dal modello:**
- non è feature input
- non è gate
- non pesa la loss (no soft weighting)
- non entra in `physics_loss_full`

QS aggregato è calcolato solo per **diagnostica post-hoc nel notebook** (binning errori per fascia di qualità) e per il framework **Continual Learning** (gating updater, drift detection, replay weighting). Vedi `CONTINUAL_LEARNING.md`.

### `eta_adjusted` vs `eta_base` vs `eta_T`

| Variabile | Cosa è | Dove vive | Uso |
|---|---|---|---|
| `eta_base[p]` | PR mediano dai dati | `compute_qs` (riscritto) | Seed + atteso per m5 |
| `eta_T[p, t]` | `eta_base * (1 - γ(T-25))` | `compute_qs` | Target istantaneo m5 |
| `eta_adjusted[p]` | WLS through origin, clip `[0.1, 0.98]` | `PVDataset` | Target del vincolo `L_physics` |

```text
eta_adjusted[p]: solar_norm = solar_kwm2 / solar_p99
                 pv_norm = ENERGIA / pv_scale
                 w = solar_norm
                 eta = sum(w * solar_norm * pv_norm) / sum(w * solar_norm^2)
```

WLS pesato in irradianza: punti ad alto SNR dominano, basso GHI rumoroso non distorce.

## 8. Loss

```text
L = L_base + peak_loss_weight * L_peak
```

### `L_base`

```text
L_base = L_ghi + L_pv + lam * L_physics
L_ghi      = mean((pred_ghi - true_ghi)^2)
L_pv       = mean((pred_pv  - true_pv)^2)
L_physics  = mean((pred_pv - eta_T * pred_ghi)^2)     # multiplicativo
```

**Multiplicativo** (non `(pred_pv/pred_ghi - eta)^2`): evita div per zero quando `pred_ghi → 0` di notte (forzato a zero da `kt*ghi_cs` con `ghi_cs ≈ 0`).

`eta_T` qui = `eta_adjusted` per impianto.

### `L_peak` — penalità asimmetrica picchi

```text
w_peak = 1 + peak_alpha * true_pv^peak_gamma
err    = pred_pv - true_pv
asym   = under_penalty * |err|  se err < 0
         |err|                   se err >= 0
L_peak = mean(w_peak * asym)
```

Configurazione: `peak_alpha=2.0`, `peak_gamma=2.0`, `under_penalty=2.0`, `peak_loss_weight=0.25`.

Tunato da 0.5 → 0.25 per evitare peak overshoot.

### Note loss

- **No QS-weighting** sulla loss base (rimosso quando lagged power introdotto).
- **No `quality_over_loss`** (rimosso).
- Forward train aggiunge perturbazione meteo `±5%` solo su canali 0–2.

## 9. Split e checkpoint

Split: stratificato per mese, 80/20 train/val. Validation indices: ultimi 20% per ogni mese.

Checkpoint: `best_state` aggiornato solo quando migliora `val_loss`. A fine training il modello ricarica `best_state`. `main.py` salva `checkpoints/model.pt` da best validation epoch.

## 10. Calibrazione PV

```text
pred_cal = slope * pred_pv + intercept
pred_cal = clip(pred_cal, 0.0, None)
```

Abilitata solo se migliora il KPI selezionato:

| `calibration_kpi` | Condizione |
|---|---|
| `rmse` | RMSE dopo < RMSE prima |
| `mae` | MAE dopo < MAE prima |
| `both` | entrambi migliorano |
| `none` | disabilitata |

Configurazione corrente: `none`, per non comprimere i picchi. KPI calcolati dopo floor a zero.

## 11. Metriche errore

```python
diff = pred - true
mae  = mean(|diff|)
rmse = sqrt(mean(diff^2))
```

| KPI | Target | Unità |
|---|---|---|
| `mae_pv`, `rmse_pv` | `y_pv` normalizzato | adim (0–1) |
| `mae_ghi`, `rmse_ghi` | `y_ghi` | kW/m² |

Baseline: `NaiveBaseline` scala POA al p99 di produzione per impianto.

## 12. Outlier filter

`_filter_outlier_plants` in `main.py`. Drop plant se:
- `qs_daytime_mean < 0.30` OR
- `n_valid_daytime_samples < 200`

Toggle `APPLY_OUTLIER_FILTER`. Nel run corrente: **attivo**. Nel default committed: disabilitato (filtrare contraddice narrativa CL — CL deve gate-are non scartare).

## 13. Glossario parametri

### Costanti fisiche

| Simbolo | Valore | Unità | Cosa rappresenta |
|---|---|---|---|
| `_GAMMA` | 0.004 | K⁻¹ | Coeff. termico potenza c-Si (IEC 61215) |
| `T_STC` | 25 | °C | Temperatura standard |
| `1000` | 1000 | W/m² | Irradianza STC |
| `_NIGHT_KW` | 0.05 | kW/m² | Soglia giorno/notte |
| `KT_MAX` | 1.2 | adim | Bound fisico clear-sky index |
| `_EPS` | 1e-6 | — | Numerico anti div-zero |

### Parametri stimati per impianto

| Simbolo | Definizione | Range |
|---|---|---|
| `pv_scale[p]` | `p99(ENERGIA_day)` | impianto-specifico |
| `solar_p99[p]` | `p99(solar_kwm2_day)` | ≈ 0.8–1.0 |
| `capacity_scale[p]` | `pv_scale / solar_p99` | impianto-specifico |
| `eta_base[p]` | PR mediano | `[0.1, 2.0]` |
| `eta_T[p, t]` | `eta_base * (1 - 0.004*(T-25))` | varia con T |
| `eta_adjusted[p]` | WLS clip `[0.1, 0.98]` | `[0.1, 0.98]` |

### `p99` perché

Robusto a outlier (sensori spike, errori SCADA puntuali). Più stabile di `max`. Cattura plateau di produzione/irradianza ignorando top 1% di rumore. `p95` sotto-stima.

### Performance Ratio

```text
PR = produzione_reale / produzione_teorica_attesa
```

Adimensionale, tipico 0.7–0.9 per impianti sani. Cala per soiling, degradazione moduli, mismatch elettrico, perdite.

`m5` confronta `PR_obs = ENERGIA/ref` con `eta_T` atteso → bassa coerenza = degradazione fisica.

### Parametri training

| Parametro | Valore | Significato |
|---|---|---|
| `lam` | 0.1 | Moltiplicatore `L_physics` |
| `peak_alpha` | 2.0 | Ampiezza pesatura picchi |
| `peak_gamma` | 2.0 | Esponente pesatura picchi |
| `peak_loss_weight` | 0.25 | Moltiplicatore `L_peak` |
| `under_penalty` | 2.0 | Penalità sottostima |
| `eta_max` | 0.98 | Cap superiore `eta_adjusted` |
| `lr` | 1e-3 | AdamW learning rate |
| `weight_decay` | 1e-4 | AdamW |
| `batch_size` | 8 | |
| `n_epochs` | 10 | |
| `early_stopping_patience` | 3 | |
| `calibration_kpi` | `none` | Calibrazione PV disabilitata |

### Parametri QS rolling

| Parametro | Valore | Significato |
|---|---|---|
| `window` | 720h (≈30gg) | Lunghezza rolling m1..m5 |
| `min_periods` | 180 (= window/4) | Min osservazioni per metrica |

### Parametri grafo

| Parametro | Valore |
|---|---|
| `max_dist_km` | 10.0 (in `build_graph`) — nota: `seq_len=24` finestra |
| `edge_weight` | `1 / dist_km` |

### Mappa parametri → cosa controllano

| Cambia... | Modifica... |
|---|---|
| Quanto pesa vincolo fisico | `lam` |
| Quanto cattura picchi | `peak_loss_weight`, `peak_alpha`, `peak_gamma` |
| Lunghezza finestra modello | `seq_len` |
| Capacità rappresentativa | `d_model`, `gat_dim`, `gat_layers` |
| Cap fisico PR | `eta_max` |
| Lunghezza storia QS | `window` (in `compute_qs`) |
| Connettività grafo | `max_dist_km` |
| Filtro outlier on/off | `APPLY_OUTLIER_FILTER` in `main.py` |

## 14. File su disco

| File | Formato | Contenuto |
|---|---|---|
| `2019_UPN_*.csv` | CSV | Letture orarie ENERGIA |
| `piedmont_pvgis_2019.nc` | NetCDF | Meteo orario PVGIS |
| `plant_mapping.csv` | CSV | UPN → plant idx |
| `energy_with_coordinates.csv` | CSV | lat/lon/kWp per UPN |
| `checkpoints/model.pt` | PyTorch state_dict | Best val epoch |
| `checkpoints/model_config.json` | JSON | Iperparametri architettura |
| `checkpoints/training_config.json` | JSON | `eta_max`, `calibration_kpi` |
| `checkpoints/pv_calibration.json` | JSON | Slope/intercept + KPI |
| `checkpoints/loss_history.json` | JSON | Train/val per epoch + best_epoch |
