# PhysiQ-PV — Riferimento Modello

## Tipi di dato

### Variabili raw del dataset (`xr.Dataset`)

Shape comune: `(plant, time)` — 1116 impianti × ore (2019, Piemonte).

| Variabile | Tipo | Unità | Sorgente | Note |
|---|---|---|---|---|
| `ENERGIA` | `float64` | kW | Sentinel/SCADA (`2019_UPN_*.csv`) | Mediana se più letture nella stessa ora |
| `solar_irradiance_poa` | `float64` | W/m² | NetCDF PVGIS o fallback pvlib clear-sky | Convertito a kW/m² via `/1000` |
| `temperature_2m` | `float64` | °C | NetCDF meteo o fallback stagionale | Usata per correzione termica |
| `wind_speed_10m` | `float64` | m/s | NetCDF meteo o fallback costante 3 m/s | Solo feature, no fisica |
| `lat`, `lon` | `float64` | gradi decimali | `energy_with_coordinates.csv` | Grafo + geometria solare |
| `time` | `datetime64[ns]` | UTC orari | indice comune allineato | 5743 ore in 2019-03 → 2019-12 |
| `plant` | `int` | indice 0..N-1 | UPN id mappato | `plant_mapping.csv` |

### Variabili derivate (calcolate in pre-processing)

| Variabile | Calcolata in | Formula |
|---|---|---|
| `sin_solar_elev`, `cos_solar_elev` | `PVDataset` (pvlib) | da `lat/lon/time` |
| `pv_scale[p]` | `PVDataset` | `p99(ENERGIA[p] daytime)` |
| `solar_p99[p]` | `PVDataset` | `p99(solar_irradiance_poa[p]/1000 daytime)` |
| `eta_base[p]` | pre-processing | seed PR per impianto, riscritto in `compute_qs` |
| `eta_adjusted[p]` | `PVDataset` | `median(pv_norm/solar_norm)` clip `[0.1, 0.98]` |
| `QS[p, t]` | `compute_qs()` | media geometrica m1·m2·m3·m4·m5 |
| `m1_past[p, t]` | `PVDataset` | corr rolling causale `pv_norm` vs `solar_norm` |

### Tensori PyTorch (output `PVDataset.__getitem__`)

| Campo | Shape | dtype | Range |
|---|---|---|---|
| `x` | `(N, 24, 7)` | `float32` | z-score per canali 0–2, `[0, 1]` per 3–6 |
| `y_ghi` | `(N,)` | `float32` | kW/m² (≈ `[0, 1]`) |
| `y_pv` | `(N,)` | `float32` | normalizzato `[0, 1.5]` |
| `qs` | `(N,)` | `float32` | `[0, 1]` |
| `eta` | `(N,)` | `float32` | `[0.1, 0.98]` |

### File su disco

| File | Formato | Contenuto |
|---|---|---|
| `2019_UPN_*.csv` | CSV | letture orarie ENERGIA per impianto |
| `piedmont_pvgis_2019.nc` | NetCDF | meteo orario PVGIS (opzionale) |
| `plant_mapping.csv` | CSV | mapping UPN → plant idx |
| `energy_with_coordinates.csv` | CSV | lat/lon/kWp per UPN |
| `checkpoints/model.pt` | PyTorch state_dict | pesi modello best validation epoch |
| `checkpoints/model_config.json` | JSON | iperparametri architettura |
| `checkpoints/training_config.json` | JSON | config QS/loss/eta |
| `checkpoints/pv_calibration.json` | JSON | slope/intercept calibrazione + KPI |
| `checkpoints/loss_history.json` | JSON | train/val loss per epoch |

---

## Pipeline dati end-to-end

Sequenza di trasformazioni dai file su disco fino ai tensori in input al modello.

```text
[2019_UPN_*.csv]            [piedmont_pvgis_2019.nc]      [plant_mapping.csv]
       │                              │                  [energy_with_coordinates.csv]
       │                              │                              │
       ▼                              ▼                              ▼
load_sentinel_hourly()    merge_with_weather()              load_kwp() (opzionale)
       │                              │                              │
       └──────► xr.Dataset ◄──────────┘                               │
                    │                                                 │
                    ▼                                                 │
            _normalize_dataset()                                      │
                    │                                                 │
                    ▼                                                 │
              compute_qs() ──► QS DataArray                           │
                    │                                                 │
                    ▼                                                 ▼
                 PVDataset(ds, qs, kwp=kwp)
                    │
                    ▼
            (x, y_ghi, y_pv, qs, eta) tensori
                    │
                    ▼
                  STGNN
                    │
                    ▼
              pred_ghi, pred_pv
```

### Step 1 — `load_sentinel_hourly`

**Cosa fa:** legge tutti i CSV `2019_UPN_*.csv`, allinea a indice temporale orario comune, aggrega con mediana se più letture nella stessa ora.

**Output:** `xr.Dataset` con variabili `ENERGIA(plant, time)` e coordinate `time`, `plant`.

**Da dove:** `/data/SentinelPV/energy_data/piemonte_energy_data/single_ups/`.

### Step 2 — `merge_with_weather`

**Cosa fa:** apre il NetCDF PVGIS 2019, fa matching spaziale (per ogni impianto trova la cella PVGIS più vicina via `lat/lon`), allinea temporalmente e aggiunge:
- `solar_irradiance_poa` (W/m²)
- `temperature_2m` (°C)
- `wind_speed_10m` (m/s)

**Fallback:** se NetCDF manca, calcola clear-sky GHI con pvlib + temperatura stagionale + vento costante 3 m/s. Solo per demo, non per training affidabile.

### Step 3 — `_normalize_dataset` (in `main.py`)

**Cosa fa:**
- aggiunge `eta_base` come seed PR per impianto (calcolato da kWp reale se disponibile, altrimenti fallback fleet)
- imposta attributi del dataset (`pv_scale_method`, ecc.)

**Output:** stesso `xr.Dataset` con variabili extra.

### Step 4 — `compute_qs`

**Cosa fa:** calcola QS per ogni `(plant, time)` usando ENERGIA + meteo. Vedi sezione **Quality Score** sotto per dettaglio.

**Output:** `xr.DataArray` di shape `(plant, time)` con valori in `[0, 1]` o NaN.

### Step 5 — `PVDataset`

**Cosa fa:**
- calcola geometria solare (`sin_solar_elev`, `cos_solar_elev`) con pvlib da `lat/lon/time`
- calcola `pv_scale[p]`, `solar_p99[p]`, `eta_adjusted[p]` per impianto
- calcola `m1_past[p, t]` rolling causale
- normalizza meteo con z-score globale
- crea finestre temporali di lunghezza `seq_len=24`
- restituisce sample come `(x, y_ghi, y_pv, qs, eta)`

**Output:** PyTorch Dataset con tensori `float32`.

### Step 6 — `STGNN.forward`

Tensori passano attraverso encoder PatchTST → proiezione → GAT → softplus heads. Output: `pred_ghi, pred_pv` per ogni `(batch, plant)`.

---

## Architettura

```
(B, N, 24, 7)
    → PatchTSTEncoder        encoder temporale, channel-independent
    → proiezione lineare
    → GAT (grafo geografico)  aggregazione spaziale tra impianti vicini
    → softplus heads
    → pred_ghi, pred_pv
```

| Componente | Dettaglio |
|---|---|
| Encoder temporale | PatchTST: sequenza patchata, Transformer per canale |
| Aggregazione spaziale | Graph Attention Network su grafo geografico lat/lon |
| Output | due teste `softplus` → valori non negativi |

Configurazione operativa (`train.py`):

| Parametro | Valore |
|---|---|
| `seq_len` | 24 ore |
| `patch_len` | 4 |
| `stride` | 2 |
| `d_model` | 64 |
| `gat_dim` | 96 |
| `gat_heads` | 4 |
| `gat_layers` | 1 |
| `dropout` | 0.0 |

---

## Input al modello

Ogni sample ha shape `(N, 24, 7)`: N impianti, finestra 24 ore, 7 canali.

| Canale | Variabile | Trasformazione | Sorgente |
|---|---|---|---|
| 0 | `temperature_2m` | z-score globale | meteo orario |
| 1 | `solar_irradiance_poa` | z-score globale | meteo orario |
| 2 | `wind_speed_10m` | z-score globale | meteo orario |
| 3 | `sin_solar_elev` | pvlib, in `[0, 1]` | geometria solare |
| 4 | `cos_solar_elev` | pvlib, in `[0, 1]` | geometria solare |
| 5 | `QS` | in `[0, 1]` | `compute_qs()` su dati t-window:t-1 |
| 6 | `m1_past` | correlazione rolling causale | calcolato su dati t-window:t-1 |

Note importanti:
- `pvgis_ref` **non** è una feature.
- La produzione passata **non** entra direttamente come feature.
- `QS` e `m1_past` usano solo dati fino a `t-1`: nessun data leakage.
- Il QS del timestep target è disponibile **solo dopo** osservazione di `ENERGIA(t)`.

### `m1_past`

```text
m1_past(t) = corr(pv_norm[t-window:t-1], solar_norm[t-window:t-1])
```

Rende esplicita la coerenza storica tra produzione PV e irradianza.

---

## Target

```text
y_ghi = solar_irradiance_poa / 1000.0          [kW/m2]
y_pv  = clip(ENERGIA / pv_scale, 0.0, 1.5)    [normalizzato]
```

`pv_scale[p]` = p99 della produzione diurna osservata per impianto.

Le ore diurne sono definite da:
- elevazione solare positiva (`sin_elev > 0.05`)
- irradianza minima (`solar_irradiance_poa / 1000 > 0.03`)

---

## Quality Score (QS)

`QS ∈ [0, 1]` per coppia `(plant, time)`. Calcolato da `compute_qs()` (`physiq_pv/data/quality_score.py`) usando `solar_irradiance_poa / 1000.0` come riferimento fisico scalato per impianto.

### Dati in ingresso a `compute_qs()`

| Variabile nel dataset | Unità originale | Trasformazione interna | Sorgente |
|---|---|---|---|
| `ds["ENERGIA"]` | kW (produzione SCADA/Sentinel) | nessuna | Sentinel/SCADA |
| `ds["solar_irradiance_poa"]` | W/m² | `/1000` → kW/m², poi `× capacity_scale[p]` | meteo orario |
| `ds["temperature_2m"]` | °C | nessuna | meteo orario |
| `ds["eta_base"]` | adimensionale | riscritto dentro QS come `median(ENERGIA/irr_scaled)` | stimato in pre-processing |

**`capacity_scale[p]`** è calcolato internamente:

```text
capacity_scale[p] = p99(ENERGIA_day[p]) / p99(irr_day[p])
```

Allinea il riferimento irradiance-based alla scala reale dell'impianto (kW prodotti).
Dopo la scalatura: `ref[p] = irr_raw[p] × capacity_scale[p]`, con `ref ~ ENERGIA/PR`.

**`eta_base[p]`** viene letto da `ds["eta_base"]` come seed, poi **riscritto** dentro `compute_qs`:

```text
eta_base[p] = median(ENERGIA[p] / ref[p])   sulle ore diurne
              clip([0.1, 2.0])
```

Rappresenta il Performance Ratio mediano stimato dai dati stessi, non da PVGIS.

**`eta_T[p, t]`** — efficienza corretta per temperatura (IEC 61215):

```text
eta_T[p, t] = eta_base[p] × (1 - 0.004 × (temperature_2m[p,t] - 25))
```

### Definizione di ore diurne in QS

```text
daytime[p, t]  ⟺  solar_irradiance_poa[p, t] / 1000 > 0.05 kW/m²
```

Ore notturne gestite separatamente:
- `irr ≈ 0` e `ENERGIA ≈ 0` → QS = 1.0 (sensore corretto)
- `irr ≈ 0` e `ENERGIA > soglia` → QS = 0.0 (produzione in buio fisico = guasto)
- `0 < irr < 0.05` (alba/tramonto marginali) → QS = NaN

### Formula

```text
QS = (m1 * m2 * m3 * m4 * m5) ^ 0.2
```

Media geometrica delle 5 metriche (ciascuna clippata a `[0, 1]`): se una vale zero, QS → 0.

Finestra rolling: **720 ore** (≈ 30 giorni), `min_periods = 180`.

### Metriche componenti — formula esatta e sorgente dati

#### `m1` — correlazione temporale

```text
m1[p, t] = clip(Pearson(ENERGIA[p, t-720:t], ref[p, t-720:t]), 0, 1)
```

- Sorgente: `ENERGIA` (Sentinel) + `solar_irradiance_poa` scalata (meteo)
- Basso m1 = produzione non segue il sole → guasto / ombreggiamento / dato SCADA corrotto

#### `m2` — bias relativo

```text
diff_mean = rolling_mean(ENERGIA - ref, 720h)
ref_mean  = rolling_mean(ref, 720h)
m2[p, t]  = clip(1 - |diff_mean| / ref_mean, 0, 1)
```

- Sorgente: `ENERGIA` + `ref` scalato
- Basso m2 = impianto sistematicamente sopra o sotto il riferimento irradiance-based

#### `m3` — completezza

```text
m3[p, t] = clip(1 - nan_fraction(ENERGIA[p, t-720:t]), 0, 1)
```

- Sorgente: `ENERGIA` raw (tutte le ore, non solo diurne)
- Basso m3 = molte ore mancanti nei dati SCADA → gap di lettura o disconnessione

#### `m4` — rapporto di varianza

```text
var_ratio = std(ENERGIA[p, t-720:t]) / std(ref[p, t-720:t])
m4[p, t]  = clip(var_ratio, 0, 1)
```

- Sorgente: `ENERGIA` + `ref` scalato
- Penalizzazione **asimmetrica**: `var_ratio > 1` viene clippato a 1 (non penalizzato), `var_ratio < 1` abbassa m4
- Basso m4 = sensore bloccato su valore costante / produzione piatta rispetto al variare dell'irradianza

#### `m5` — coerenza fisica con efficienza termica

```text
PR_obs[p, t]   = ENERGIA[p, t] / ref[p, t]
pr_ratio[p, t] = PR_obs[p, t] / eta_T[p, t]
pr_err[p, t]   = max(0, 1 - pr_ratio)
m5[p, t]       = clip(1 - rolling_mean(pr_err, 720h), 0, 1)
```

- Sorgente: `ENERGIA` + `solar_irradiance_poa` + `temperature_2m`
- Penalizza solo quando `PR_obs < eta_T` (sottoproduzione rispetto all'efficienza attesa)
- Basso m5 = impianto produce meno di quanto previsto dalla fisica (temperatura + PR mediana)

### Interpretazione fisica delle 5 metriche

Riassunto: **cosa rileva ogni metrica quando va a zero**.

| Metrica | Cosa controlla | Caso patologico tipico |
|---|---|---|
| `m1` | Coupling temporale produzione vs sole | Impianto disconnesso, forte ombreggiamento, sensori SCADA congelati |
| `m2` | Bias sistematico medio | Inverter mal calibrato, errore di scala persistente |
| `m3` | Disponibilità del dato SCADA | Disconnessione modem, gap di lettura, manutenzioni |
| `m4` | Variabilità del segnale produzione | Sensore stuck su valore costante, contatore guasto |
| `m5` | Coerenza con efficienza termica attesa | Sottoproduzione cronica oltre quella spiegabile da temperatura |

Le 5 metriche misurano **cose diverse**: la media geometrica garantisce che basta un fallimento per abbassare QS. Equivalente a "AND fuzzy" tra i 5 controlli.

### Riferimento fisico utilizzato

QS confronta produzione reale `ENERGIA[p, t]` con **riferimento data-driven**:

```text
ref[p, t] = (solar_irradiance_poa[p, t] / 1000) × capacity_scale[p]
```

- Numeratore: irradianza in `kW/m²` (proxy "quanto sole disponibile per m²")
- `capacity_scale[p]`: fattore di conversione da `kW/m²` a `kW prodotti` per quell'impianto
- `ref` ha quindi unità di `kW`, comparabile direttamente con `ENERGIA`

**Importante:** non viene usato `pvgis_ref` (la "produzione attesa" PVGIS) come riferimento. Il riferimento è **costruito dai dati stessi**, dall'irradianza misurata + capacità stimata.

Questo è il core fisico: assumiamo che produzione e irradianza debbano essere proporzionali, con costante di proporzionalità (`capacity_scale × eta_T`) data dalla teoria fotovoltaica. QS misura quanto questa relazione è rispettata nei dati.

### Distinzione `eta_base` vs `eta_adjusted` vs `eta_T`

| Variabile | Cosa è | Dove vive | Uso |
|---|---|---|---|
| `eta_base[p]` | PR mediano stimato dai dati di un impianto | `ds["eta_base"]` → riscritto in `compute_qs` | seed + atteso per m5 |
| `eta_T[p, t]` | `eta_base` corretto per temperatura ora per ora | calcolato in `compute_qs` | target istantaneo per m5 |
| `eta_adjusted[p]` | `eta_base` clippato `[0.1, 0.98]` | `dataset.eta_adjusted` in `PVDataset` | target del vincolo fisico `L_physics` |

Tutti convergono allo stesso concetto (Performance Ratio dell'impianto), ma calcolati in contesti diversi.

### Uso operativo del QS

QS **non è un gate**: abbassa il contributo dei dati di bassa qualità ma non li elimina.

**Come feature** (canale 5): rappresenta la qualità storica osservata nella finestra input.

**Come peso loss** (soft weighting):

```text
weight = qs_weight_floor + (1 - qs_weight_floor) * QS ^ qs_weight_exponent
       = 0.2 + 0.8 * QS ^ 0.2        (configurazione corrente)
```

Effetto:
- QS = 1.0 → weight = 1.0
- QS = 0.5 → weight ≈ 0.75
- QS = 0.0 → weight = 0.2 (floor: il campione contribuisce comunque)

---

## Loss

```text
L = L_base + peak_loss_weight * L_peak + quality_over_loss_weight * L_quality_over
```

### `L_base`

```text
L_base     = L_ghi + L_pv + lam * L_physics

L_ghi      = mean(weight * (pred_ghi - true_ghi)^2)
L_pv       = mean(weight * (pred_pv  - true_pv)^2)
L_physics  = mean(weight * (pred_pv / |pred_ghi| - eta_adjusted)^2)
```

`eta_adjusted` è la proxy di Performance Ratio per impianto:

```text
eta_adjusted[p] = median(pv_norm / solar_norm)  sulle ore diurne
                  clip([0.1, eta_max])  con eta_max = 0.98
```

### `L_peak` — penalità sottostima picchi PV

```text
w_peak = 1 + peak_alpha * true_pv ^ peak_gamma
err    = pred_pv - true_pv
asym   = 2 * |err|  se err < 0
         |err|      se err >= 0
L_peak = mean(weight * w_peak * asym)
```

Obiettivo: ridurre la sottostima sistematica nei picchi senza cambiare architettura.

### `L_quality_over` — penalità sovrastima quality-aware

```text
risk          = (1 - QS) * (1 - m1_past)
over          = max(pred_pv - true_pv, 0)
L_quality_over = mean(risk * over^2)
```

Configurazione: `quality_over_loss_weight = 0.02`.

Obiettivo: penalizzare la sovrastima nei campioni con QS basso e bassa coerenza PV-irradianza.

---

## Metriche di errore

Calcolate su array `pred` e `true` con `NaN` esclusi:

```python
diff = pred - true
mae  = mean(|diff|)
rmse = sqrt(mean(diff^2))
```

Il post-processing (calibrazione lineare + floor a zero) viene applicato **prima** del calcolo dei KPI, coerentemente con l'inferenza operativa.

### Definizioni

| Metrica | Formula | Unità | Interpretazione |
|---|---|---|---|
| **MAE** (Mean Absolute Error) | `mean(|pred - true|)` | stessa del target | Errore medio assoluto. Robusto agli outlier. |
| **RMSE** (Root Mean Square Error) | `sqrt(mean((pred - true)^2))` | stessa del target | Penalizza errori grandi più del MAE. Sensibile agli outlier. |

### KPI per target

| KPI | Target | Unità |
|---|---|---|
| `mae_pv` | `y_pv` normalizzato | adimensionale (0–1) |
| `rmse_pv` | `y_pv` normalizzato | adimensionale (0–1) |
| `mae_ghi` | `y_ghi` in kW/m2 | kW/m2 |
| `rmse_ghi` | `y_ghi` in kW/m2 | kW/m2 |

### Differenza MAE vs RMSE

- MAE = media errori assoluti → peso uguale a tutti gli errori
- RMSE = radice della media degli errori quadrati → errori grandi pesano di più
- RMSE ≥ MAE sempre; se RMSE >> MAE, il modello ha errori grandi concentrati (es. picchi)
- Per l'energia solare i picchi sono critici: RMSE è il KPI più diagnostico

### Baseline di confronto

`NaiveBaseline` scala l'irradianza POA al p99 di produzione osservata per impianto. Se il modello non batte questa baseline su MAE e RMSE, la capacità predittiva è da verificare.

---

## Calibrazione post-training (opzionale)

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
| `none` | disabilitata (configurazione corrente) |

Configurazione corrente: `none`, per non comprimere i picchi con la regressione lineare.

---

## Glossario parametri

Spiegazione di ogni simbolo/parametro che appare nel codice e nei file di config.

### Costanti fisiche

| Simbolo | Valore | Unità | Cosa rappresenta |
|---|---|---|---|
| `_GAMMA` | `0.004` | K⁻¹ | Coefficiente termico di potenza per moduli c-Si (riferimento IEC 61215). Indica quanto cala l'efficienza per ogni grado sopra 25°C: ~0.4% per K. |
| `T_STC` | `25` | °C | Temperatura standard test conditions. Sotto STC il modulo eroga la potenza nominale. |
| `1000` | `1000` | W/m² | Irradianza STC. Convenzione globale per dichiarare la potenza nominale dei pannelli. Da qui il `/1000` per convertire `solar_irradiance_poa` in unità STC. |
| `_NIGHT_KW` | `0.05` | kW/m² | Soglia operativa giorno/notte. Sotto questa irradianza l'impianto non può produrre significativamente. |
| `_EPS` | `1e-6` | — | Costante numerica per evitare divisioni per zero. |

### Parametri stimati per impianto

| Simbolo | Definizione | Unità | Range tipico | Significato |
|---|---|---|---|---|
| `pv_scale[p]` | `p99(ENERGIA_day[p])` | kW | impianto-specifico | Capacità nominale data-driven. Usato per normalizzare la produzione: `pv_norm = ENERGIA / pv_scale ∈ [0, ~1.5]`. |
| `solar_p99[p]` | `p99(solar_irradiance_poa_day[p] / 1000)` | kW/m² | ≈ 0.8–1.0 | Irradianza massima osservata per quell'impianto. Riferimento per normalizzazione irradianza. |
| `capacity_scale[p]` | `pv_scale[p] / solar_p99[p]` | kW / (kW/m²) = m² equivalenti | impianto-specifico | Fattore di conversione che permette di esprimere irradianza in unità di potenza dell'impianto. Concettualmente: "area attiva equivalente × efficienza". |
| `eta_base[p]` | `median(ENERGIA / ref)` daytime | adimensionale | `[0.1, 2.0]` | Performance Ratio mediano. Frazione di potenza realmente erogata rispetto a quella attesa data l'irradianza. Tipico 0.7–0.9. |
| `eta_T[p, t]` | `eta_base × (1 - 0.004 × (T - 25))` | adimensionale | varia con T | Performance Ratio corretto per temperatura ora per ora. A 35°C ≈ `eta_base × 0.96`. |
| `eta_adjusted[p]` | `clip(eta_base, 0.1, 0.98)` | adimensionale | `[0.1, 0.98]` | Versione clippata di `eta_base` usata come target nel vincolo fisico `L_physics`. Cap a 0.98 evita saturazione fisica e sovrastima. |

### Cos'è `p99`

**Definizione:** 99° percentile di una distribuzione = valore sotto cui cade il 99% delle osservazioni.

**Perché `p99` e non `max`:** il massimo è dominato da outlier (spike sensori, errori SCADA puntuali). `p99` è un proxy robusto del valore "tipicamente massimo" ignorando il top 1% di rumore.

**Perché non `p95`:** `p99` cattura meglio il vero plateau di produzione/irradianza. `p95` sotto-stima la capacità.

**Esempio numerico:** se un impianto ha 5000 letture diurne con `ENERGIA` in kW e `p99 = 250 kW`, vuol dire che 4950 letture sono sotto 250 kW e quelle sopra sono trattate come anomale.

### Parametri rolling/finestra QS

| Parametro | Valore | Unità | Significato |
|---|---|---|---|
| `window` | `720` | ore (≈ 30 giorni) | Lunghezza finestra rolling per calcolo m1, m2, m3, m4, m5. 30 giorni cattura un mese di pattern (ciclo soiling, stagionalità di breve termine). |
| `min_periods` | `180` (= window/4) | ore | Numero minimo di osservazioni nella finestra per calcolare la metrica. Sotto questa soglia → NaN. |

### Performance Ratio (PR) — concetto chiave

**Definizione fisica:**

```text
PR = produzione_reale / produzione_teorica_attesa
```

**Adimensionale**, valore tipico `0.7–0.9` per impianti sani, scende per:
- Soiling (sporco sui pannelli)
- Degradazione moduli
- Mismatch elettrico
- Perdite cavi/inverter

**Nel codice:**

```text
PR_obs[p, t] = ENERGIA[p, t] / ref[p, t]
```

`m5` confronta `PR_obs` con `eta_T` (PR atteso data temperatura): bassa coerenza → produzione cronica sotto le aspettative → degradazione fisica.

### Parametri training (in `train.py` / `training_config.json`)

| Parametro | Valore | Range valido | Significato |
|---|---|---|---|
| `qs_weight_floor` | `0.2` | `[0, 1]` | Peso minimo del campione anche se QS=0. Evita che dati di bassa qualità vengano ignorati del tutto. A 0 = QS è gate. |
| `qs_weight_exponent` | `0.2` | `[0, ∞)` | Esponente nella curva `weight = floor + (1-floor) × QS^exp`. Con esponente piccolo (`0.2`), la curva è concava: anche QS=0.5 ha peso ≈ 0.7. Avvicina QS=0 a QS=1. |
| `eta_max` | `0.98` | `[0.1, 1.0]` | Cap superiore di `eta_adjusted`. Sopra a 1.0 implicherebbe efficienza > 100% (non fisico). 0.98 lascia margine ai migliori impianti reali. |
| `lam` | dipende da config | — | Moltiplicatore della componente `L_physics` nel `L_base`. |
| `peak_loss_weight` | configurabile | — | Moltiplicatore della componente `L_peak` nel totale. |
| `peak_alpha` | configurabile | — | Ampiezza della pesatura amplificata sui picchi. `w_peak = 1 + alpha × true_pv^gamma`. |
| `peak_gamma` | configurabile | — | Esponente della pesatura amplificata sui picchi. Più grande → più aggressiva sui picchi alti. |
| `quality_over_loss_weight` | `0.02` | — | Moltiplicatore di `L_quality_over` nel totale. |
| `calibration_kpi` | `none` | `{rmse, mae, both, none}` | Quale KPI deve migliorare per attivare la calibrazione lineare PV post-training. |

### Parametri architettura modello (in `model_config.json`)

| Parametro | Valore | Significato |
|---|---|---|
| `n_nodes` | 1116 | Numero di impianti nella flotta. |
| `n_features` | 7 | Canali in input per timestep (vedi tabella "Input al modello"). |
| `seq_len` | 24 | Lunghezza finestra temporale in input al modello (24 ore = 1 giorno). |
| `patch_len` | 4 | Lunghezza di ogni patch in PatchTST. La sequenza viene divisa in patch di 4 ore. |
| `stride` | 2 | Step tra patch consecutive. Con `stride=2` e `patch_len=4`: overlap del 50%. Numero patch = `(seq_len - patch_len)/stride + 1 = 11`. |
| `d_model` | 64 | Dimensione embedding del Transformer interno a PatchTST. Ogni patch viene proiettata in vettore 64-d. |
| `gat_dim` | 96 | Dimensione hidden della Graph Attention. Più grande = più capacità ma più parametri. |
| `gat_heads` | 4 | Numero di teste di attention parallele nel GAT. Multi-head permette di attendere diversi pattern relazionali simultaneamente. |
| `gat_layers` | 1 | Numero di strati GAT. Con 1 strato ogni nodo riceve info dai vicini diretti (1-hop nel grafo). |
| `dropout` | 0.0 | Probabilità di dropout nel Transformer. `0.0` = disattivato nella config baseline. |

### Costruzione del grafo geografico

| Parametro | Valore | Significato |
|---|---|---|
| `max_dist_km` | `20.0` | Distanza massima per creare un edge tra due impianti. Sotto questa soglia → connessione, sopra → no. |
| `edge_weight` | `1 / dist_km` | Peso dell'edge inversamente proporzionale alla distanza geografica. Vicini pesano di più. |

Edge totali su 1116 impianti: `~113k`.

### Riepilogo mappa parametri → cosa controllano

| Vuoi cambiare... | Modifica... |
|---|---|
| Quanto i dati di bassa qualità contribuiscono | `qs_weight_floor`, `qs_weight_exponent` |
| Quanto pesa il vincolo fisico nella loss | `lam` |
| Quanto il modello cerca di catturare i picchi | `peak_loss_weight`, `peak_alpha`, `peak_gamma` |
| Quanto è aggressiva la penalità sovrastima quality-aware | `quality_over_loss_weight` |
| Lunghezza finestra di osservazione modello | `seq_len` |
| Capacità rappresentativa del modello | `d_model`, `gat_dim`, `gat_layers` |
| Cap fisico del Performance Ratio | `eta_max` |
| Quanto è "lunga" la storia per QS | `window` (in `compute_qs`) |
| Quali impianti sono connessi nel grafo | `max_dist_km` (in `build_graph`) |
