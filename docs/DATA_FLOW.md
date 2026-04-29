# PhysiQ-PV — Flusso dei Dati e Ruolo di Ogni Sorgente

---

## 1. Le tre sorgenti dati

| Sorgente | File | Variabili | Scala |
|----------|------|-----------|-------|
| Sentinel/SCADA | `2019_UPN_*.csv` | `ENERGIA` | per impianto |
| PVGIS (satellite) | `piedmont_pvgis_2019.nc` | `pvgis_ref`, `solar_irradiance_poa`, `temperature_2m`, `wind_speed_10m` | per griglia geografica |
| Registro GSE | `energy_with_coordinates.csv` | `Potenza di picco (kW)`, lat/lon | per impianto |

PVGIS copre il Piemonte con una griglia di 1149 punti. Ogni impianto viene associato al punto più vicino (nearest-neighbor su lat/lon).

---

## 2. Come partecipano al modello

### 2.1 Le 5 feature di input

Il modello riceve per ogni impianto le ultime **24 ore** di 5 variabili:

```
x[impianto, t-23..t, :] = [temperature_2m, solar_irradiance_poa, wind_speed_10m, pvgis_ref, QS]
```

Tutte normalizzate con z-score globale (media e std calcolate sull'intero dataset), tranne QS che è già in [0,1].

| Feature | Sorgente | Ruolo fisico nel modello |
|---------|----------|--------------------------|
| `temperature_2m` | PVGIS | Efficienza cala ~0.4%/K sopra 25°C; la temperatura delle ultime 24h cattura l'inerzia termica dei pannelli |
| `solar_irradiance_poa` | PVGIS | Irradianza sul piano del pannello: driver principale della produzione; il profilo orario delle 24h precedenti informa sulla tendenza meteo |
| `wind_speed_10m` | PVGIS | Raffreddamento pannelli; contributo minore |
| `pvgis_ref` | PVGIS | Produzione di riferimento per 1 kWp in condizioni reali satellite; sintetizza irradianza + temperatura + perdite spettrali in un unico numero fisicamente interpretabile |
| `QS` | calcolato | Qualità del sensore di questo impianto nelle 24h precedenti; il modello impara a fidarsi meno di impianti con QS basso |

**Nessuna feature è la produzione ENERGIA passata** — il modello impara la fisica, non l'autocorrelazione del sensore.

### 2.2 I due target

Il modello predice simultaneamente due quantità al timestep t+1:

```
pred_pv  (B, N)   — produzione PV normalizzata per impianto
pred_ghi (B, N)   — irradianza GHI in kW/m²
```

`pred_ghi` viene supervisionato da `solar_irradiance_poa / 1000`. Predire anche l'irradianza (e non solo la produzione) forza il modello a rappresentare esplicitamente lo stato dell'irraggiamento, che è il vincolo fisico dominante.

### 2.3 La loss e il vincolo fisico

```
L = L_ghi + L_pv + 0.1 × L_physics

L_ghi      = mean(w × (pred_ghi - target_ghi)²)
L_pv       = mean(w × (pred_pv  - target_pv)²)
L_physics  = mean(w × (pred_pv / pred_ghi - eta_adjusted)²)

w = QS^0.2   ← peso per impianto/timestep
```

`L_physics` è il termine che lega le due uscite: il rapporto `pred_pv / pred_ghi` deve approssimare il Performance Ratio stimato per quell'impianto. Questo impedisce al modello di produrre previsioni fisicamente inconsistenti (es. alta produzione con bassa irradianza).

Il peso `QS^0.2` riduce il contributo alla loss degli impianti con sensore difettoso o dati mancanti.

---

## 3. Come si stima la dimensione dell'impianto

Il registro GSE (`energy_with_coordinates.csv`) fornisce la potenza di picco nominale (kWp) per molti impianti, ma non per tutti. Il sistema calcola una **stima data-driven** indipendente dal registro.

### 3.1 Stima kWp dai dati

```
pv_scale[p]   = p99(ENERGIA[p]  per ore diurne)   [kW]      ← picco produzione osservata
pvgis_p99[p]  = p99(pvgis_ref[p] per ore diurne)  [kW/kWp]  ← picco riferimento PVGIS

kWp_est[p] = pv_scale[p] / pvgis_p99[p]   [kWp]
```

**Interpretazione**: `pvgis_p99` è la produzione massima che farebbe un sistema da 1 kWp in quella posizione geografica (giorno estivo soleggiato). Se l'impianto reale produce al picco `pv_scale = 820 kW` e un 1 kWp produce al picco `pvgis_p99 = 0.72 kW/kWp`, allora l'impianto equivale a circa `820 / 0.72 ≈ 1139 kWp`.

Questa stima è **puramente osservativa** — non assume nulla sulla tecnologia, orientamento o configurazione dell'impianto. Cattura automaticamente perdite di cablaggio, ombreggiamento parziale, degradazione.

Valori tipici per la flotta Piemonte 2019:

| Statistica | kWp_est |
|-----------|---------|
| Minimo | 1.4 kWp |
| Mediana | 222 kWp |
| Massimo | 7327 kWp |
| Media | 570 kWp |

Il range molto ampio (residenziale 1-5 kWp fino a utility-scale 5-7 MW) riflette la natura eterogenea del registro Piemonte.

### 3.2 Performance Ratio per impianto (eta_adjusted)

```
pvgis_norm[p, t]    = pvgis_ref[p, t] / pvgis_p99[p]       ← pvgis normalizzato [0..1]
target_pv_norm[p, t] = clip(ENERGIA[p, t] / pv_scale[p], 0, 1.5)  ← produzione normalizzata

ratio[p, t] = target_pv_norm[p, t] / pvgis_norm[p, t]      ← solo ore diurne (pvgis > 0.25 kW/kWp)

eta_adjusted[p] = median(ratio[p, :])
```

`eta_adjusted` è il **Performance Ratio** dell'impianto: la frazione della produzione di riferimento effettivamente realizzata. Incorpora orientamento, inclinazione, ombreggiamento permanente, degradazione dei pannelli, efficienza inverter.

```
eta_adjusted = 1.0  →  impianto produce esattamente come il riferimento PVGIS
eta_adjusted = 0.85 →  impianto produce il 15% meno del riferimento (es. orientamento sub-ottimale)
eta_adjusted = 0.30 →  impianto gravemente sottoperformante (guasto / sensore rotto)
```

Valori flotta attuale: media **0.925**, range 0.24–1.00. Molti impianti sono clipgati a 1.0 (il clip è [0.1, 1.0]).

### 3.3 Uso in training

`kWp_est` non entra direttamente nel modello come feature. Viene usato **indirettamente** tramite:
- `pv_scale` → normalizzazione del target `target_pv_norm`
- `eta_adjusted` → vincolo fisico `L_physics`

Il modello quindi impara a predire produzione **normalizzata per impianto**: un valore di 1.0 significa "l'impianto sta producendo al suo massimo storico", indipendentemente dalla sua taglia in kWp. Questo rende il modello applicabile a impianti di dimensioni molto diverse senza riscalamento manuale.

---

## 4. Schema riassuntivo del flusso

```
Sentinel CSV                    PVGIS NetCDF
  ENERGIA (kW)                    pvgis_ref (kW/kWp)
      │                           solar_poa (W/m²)
      │                           temperature_2m (°C)
      │                           wind_speed_10m (m/s)
      │                                │
      ├──────────────────────────────►QS computation
      │                           │   (5 metriche fisiche)
      │                           │        │
      ▼                           │        ▼
  pv_scale[p] = p99(ENERGIA_day)  │    weight = QS^0.2
  pvgis_p99[p] = p99(pvgis_day) ◄─┘
      │
      ├──► kWp_est[p] = pv_scale / pvgis_p99
      │
      ├──► eta_adjusted[p] = median(ENERGIA_norm / pvgis_norm)
      │
      ├──► target_pv_norm = clip(ENERGIA / pv_scale, 0, 1.5)
      │
      └──► target_ghi = solar_poa / 1000

Features x (N, 24, 5):
  [temp, solar_poa_norm, wind_norm, pvgis_ref_norm, QS]
                │
         PatchTST Encoder
                │
         GAT × 2 (grafo geografico 20km)
                │
        ┌───────┴───────┐
    pred_ghi          pred_pv
        │                │
        └───── Loss ──────┘
    L_ghi + L_pv + 0.1×L_physics
    L_physics = (pred_pv/pred_ghi - eta_adjusted)²
    tutti pesati per QS^0.2
```
