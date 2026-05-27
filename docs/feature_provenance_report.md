# Feature Provenance Report — PhysiQ-PV

## 1. Feature list reale

| # | Feature | File/Funzione | Sorgente | Unita | Normalizzazione | Dipendenze |
|---|---------|---------------|----------|-------|-----------------|------------|
| 0 | `temperature_2m` | `dataset.py:104` `_norm(temp)` | PVGIS NetCDF (`piedmont_pvgis_2019.nc`) | C | z-score globale | PVGIS |
| 1 | `solar_irradiance_poa` | `dataset.py:105,204` `_norm(solar)` | PVGIS NetCDF | W/m2 | z-score globale | PVGIS |
| 2 | `wind_speed_10m` | `dataset.py:106,205` `_norm(wind)` | PVGIS NetCDF | m/s | z-score globale | PVGIS |
| 3 | `sin_solar_elev` | `dataset.py:118,206` | pvlib `get_solarposition` | [0,1] | nessuna (gia bounded) | lat/lon + timestamp |
| 4 | `cos_solar_elev` | `dataset.py:118,207` | pvlib `get_solarposition` | [0,1] | nessuna | lat/lon + timestamp |
| 5 | `m1` | `dataset.py:109,208` | `quality_score.py` | [0,1] | nan_to_num(0) | ENERGIA + solar_irradiance_poa |
| 6 | `m2` | `dataset.py:110,208` | `quality_score.py` | [0,1] | nan_to_num(0) | ENERGIA + solar_irradiance_poa |
| 7 | `m3` | `dataset.py:111,208` | `quality_score.py` | [0,1] | nan_to_num(0) | ENERGIA (solo NaN fraction) |
| 8 | `m4` | `dataset.py:112,208` | `quality_score.py` | [0,1] | nan_to_num(0) | ENERGIA + solar_irradiance_poa |
| 9 | `m5` | `dataset.py:113,208` | `quality_score.py` | [0,1] | nan_to_num(0) | ENERGIA + solar_irradiance_poa + temperature_2m + eta_base |
| 10 | `pv_lag` | `dataset.py:164,209` | ENERGIA / pv_scale | [0,1.5] | per-plant (kwp o p99) | ENERGIA + pv_scale |
| 11 | `kt` | `dataset.py:169,210` | solar_raw_kwm2 / ghi_cs | [0,1.5] | nessuna (gia ratio) | solar_irradiance_poa + pvlib clear-sky |
| 12 | `kt_std_3h` | `dataset.py:173-176,211` | rolling(3h).std(kt) | [0,~0.5] | nessuna | kt (-> solar_irradiance_poa) |
| 13 | `dghi_dt` | `dataset.py:179-180,212` `_norm(dghi)` | diff(solar_raw_kwm2) | kW/m2/h | z-score globale | solar_irradiance_poa |
| 14 | `dni_norm` | `dataset.py:192-197,213` | pvlib.irradiance.erbs(solar_poa_as_ghi) | kW/m2 | clip [0,1.5] | solar_irradiance_poa + zenith + DOY |
| 15 | `dhi_norm` | `dataset.py:192-198,214` | pvlib.irradiance.erbs(solar_poa_as_ghi) | kW/m2 | clip [0,1.0] | solar_irradiance_poa + zenith + DOY |

Feature array costruito a `dataset.py:202-216`, stacked in `self.feats` shape `(T, N, 16)`.

Ordine **hardcoded** nella lista `feature_arrays` — non esiste registry centralizzato, solo commento a `dataset.py:9-18`.

Normalizzazione: `_norm()` = z-score globale `(x - mean) / (std + 1e-6)` applicato a temp, solar, wind, dghi. Le altre feature sono gia bounded o ratio.

---

## 2. Dipendenza da `solar_irradiance_poa`

**Dipendenza diretta** (6 feature su 16):

| Feature | Tipo dipendenza |
|---------|-----------------|
| `solar_irradiance_poa` (feat 1) | **sorgente primaria** |
| `kt` (feat 11) | `solar_raw_kwm2 / ghi_cs` |
| `kt_std_3h` (feat 12) | `rolling_std(kt)` |
| `dghi_dt` (feat 13) | `diff(solar_raw_kwm2)` |
| `dni_norm` (feat 14) | Erbs decomposition di `solar_raw_kwm2 * 1000` |
| `dhi_norm` (feat 15) | Erbs decomposition di `solar_raw_kwm2 * 1000` |

**Dipendenza indiretta** (via quality score):

| Feature | Dipendenza |
|---------|------------|
| `m1` | Pearson(ENERGIA, solar_irradiance_poa) |
| `m2` | bias ENERGIA vs solar_irradiance_poa |
| `m4` | variance ratio ENERGIA vs solar_irradiance_poa |
| `m5` | physical consistency con eta_base e temperatura |

**Indipendenti** da solar_irradiance_poa:

| Feature | Sorgente |
|---------|----------|
| `temperature_2m` (feat 0) | PVGIS NetCDF (indipendente da POA) |
| `wind_speed_10m` (feat 2) | PVGIS NetCDF (indipendente da POA) |
| `sin_solar_elev` (feat 3) | pvlib (geometria solare) |
| `cos_solar_elev` (feat 4) | pvlib (geometria solare) |
| `m3` | solo NaN fraction di ENERGIA |
| `pv_lag` (feat 10) | solo ENERGIA passata |

**In sintesi**: 6/16 feature dipendono direttamente da `solar_irradiance_poa`, 4/16 indirettamente via QS, 6/16 indipendenti.

---

## 3. GHI/DNI/DHI: grezzi o derivati?

**GHI grezza: NON ESISTE come variabile separata.**

Il dataset PVGIS contiene una variabile chiamata `solar_irradiance_poa` (W/m2). Questa viene trattata come GHI proxy in tutto il codice:
- `dataset.py:185-186`: "Treat solar_irradiance_poa as the GHI proxy consistent with the rest of the pipeline."
- `README.md:163`: "PVGIS reanalysis ERA5-derived"
- `sentinel_hourly_loader.py:269`: nel fallback pvlib, `solar_irradiance_poa` viene riempita con `cs["ghi"]` (clear-sky GHI orizzontale)

**DNI/DHI grezzi: NON ESISTONO nel dataset.**

Sono derivati interamente da `solar_irradiance_poa` via Erbs:
- `dataset.py:192-198`: `pvlib.irradiance.erbs(ghi=solar_raw_kwm2*1000, zenith=zenith_deg, datetime_or_doy=doy)`

**Nessun file NetCDF contiene variabili GHI/DNI/DHI separate.**

Il PVGIS NetCDF (`piedmont_pvgis_2019.nc`) contiene solo: `solar_irradiance_poa`, `temperature_2m`, `wind_speed_10m`.

---

## 4. Uso di Erbs: diagnosi coerenza fisica

### Come viene chiamato

```python
erbs_out = pvlib.irradiance.erbs(
    ghi=ghi_wm2[:, p],          # solar_irradiance_poa * 1000 (W/m2)
    zenith=zenith_deg[:, p],     # apparent zenith from pvlib
    datetime_or_doy=doy,         # day of year
)
```

### Modello Erbs: cosa si aspetta

`pvlib.irradiance.erbs` stima la **frazione diffusa** (DHI/GHI) a partire da:
- **GHI** (Global Horizontal Irradiance) — irradianza su **piano orizzontale**
- zenith angle
- giorno dell'anno

Poi calcola: `DNI = (GHI - DHI) / cos(zenith)` e `DHI = kd * GHI`

### Il problema

`solar_irradiance_poa` nel nome suggerisce **Plane of Array** (irradianza su piano inclinato del pannello). Se fosse vera POA, passarla a Erbs come GHI sarebbe fisicamente scorretto perche:
- POA include componente diretta gia proiettata sul piano inclinato
- POA include ground-reflected (albedo)
- Erbs assume piano orizzontale

### Diagnosi

**Evidenza che `solar_irradiance_poa` e in realta GHI (o un proxy molto vicino):**

1. `sentinel_hourly_loader.py:268-269`: fallback pvlib usa `cs["ghi"]` (GHI clear-sky orizzontale) e lo assegna a `solar_irradiance_poa`
2. `README.md:163`: "PVGIS reanalysis ERA5-derived" — PVGIS puo fornire sia GHI che GTI (Global Tilted Irradiance). Se il NetCDF e stato generato senza specificare tilt/azimuth, PVGIS restituisce **GHI orizzontale**
3. Il codice non contiene mai parametri `tilt`, `azimuth`, `surface_tilt`, `surface_azimuth` → nessuna trasformazione GHI->POA
4. `MODEL_REFERENCE.md:23`: documenta `solar_irradiance_poa` come proveniente da "NetCDF PVGIS"

**Conclusione**: `solar_irradiance_poa` e **molto probabilmente GHI orizzontale** rinominata "POA" — oppure e una POA PVGIS calcolata con tilt/azimuth di default. Senza ispezionare il NetCDF originale, non si puo stabilire al 100%.

**Rischio**: Se e vera POA, Erbs produce DNI/DHI approssimati. Se e GHI, Erbs e corretto. In entrambi i casi, il modello ha imparato su questi dati e funziona, quindi cambiare sorgente senza retraining e rischioso.

**Raccomandazione**: Ispezionare `piedmont_pvgis_2019.nc` sul server per verificare metadata/attributi e capire come e stato generato (API PVGIS con quali parametri).

---

## 5. Training pipeline

### Flusso dati

```
Sentinel CSV (ENERGIA)                PVGIS NetCDF (temp, solar_poa, wind)
         |                                        |
    load_sentinel_hourly()               merge_with_weather()
         |                                        |
         +---> xr.Dataset (plant, time) <---------+
                      |
              compute_quality_score()  -> m1..m5
                      |
               PVDataset.__init__()
                      |
          feature_arrays stacked -> self.feats (T, N, 16)
                      |
               __getitem__(idx)
          -> x (N, seq_len, 16), y_ghi, y_pv, eta, ghi_cs
                      |
                  DataLoader
                      |
                 STGNN.forward(x, edge_index, edge_weight, ghi_cs)
                      |
               pred_ghi, pred_pv
                      |
        physics_loss_full(pred_ghi, pred_pv, true_ghi, true_pv, eta)
```

### N_FEATURES = 16

- Definito a `dataset.py:20`: `N_FEATURES = 16`
- Importato in: `train.py:10`, `train_replay_continual.py:57`, `run_cl_experiment.py:43`
- Passato a `STGNN(n_features=N_FEATURES)` in tutti i file di training
- Il modello assume rigidamente 16: `BiLSTMEncoder(n_features=16, ...)` poi `LSTM(input_size=16, ...)`
- L'ordine e hardcoded nella lista `feature_arrays` a `dataset.py:202-215`
- **Non esiste** un registry o enum delle feature — solo commento header e lista nell'init

### Normalizzazione

- `_norm()` (z-score globale): applicata a temp, solar, wind, dghi (4 feature)
- sin/cos elev: gia in [0,1], non normalizzate
- m1..m5: gia in [0,1], non normalizzate
- pv_lag: normalizzato per plant via pv_scale (kwp o p99)
- kt: ratio, clip [0,1.5]
- kt_std_3h: raw rolling std
- dni_kwm2, dhi_kwm2: clip a [0,1.5] e [0,1.0]

---

## 6. Continual learning pipeline

### Usa stesso PVDataset?

**Si.** `train_replay_continual.py:57` importa `PVDataset` da `dataset.py`. Stesse 16 feature, stesso ordine, stessa normalizzazione.

### Ricalcola feature per ogni finestra?

**Si.** `_build_dataset_and_loader()` (riga 234-244) crea un nuovo `PVDataset` per ogni finestra temporale. QS, kt, Erbs etc vengono ricalcolati sulla slice di `ds_window`.

### Riusa pv_scale?

**Si** (dopo fix). `fixed_pv_scale` estratto dalla finestra iniziale, passato a tutte le finestre successive via parametro `pv_scale`. Con `--pv-norm-mode kwp`, usa kwp reale.

### Pesi nella loss?

**Si.** Peak-aware focal loss implementata in `train.py:72-95` con parametri `peak_alpha`, `peak_gamma`, `peak_loss_weight`, `under_penalty`. Usata anche nel CL (`_continual_update`).

### Replay buffer?

**Si.** `SimpleReplayBuffer` (`simple_replay_buffer.py`): FIFO ring buffer, uniform random sampling. Opzionale: `sample_peak_aware()` con stratified sampling (25% peak, 25% over_100).

### Metadati su feature/weather source nel buffer?

**No.** Il buffer salva solo `(x, y_ghi, y_pv, eta, ghi_cs)` — tensori numerici senza metadata su sorgente, finestra, o condizioni meteo.

### Drift score o regime labels?

**No drift score nel CL pipeline.** Esiste `quality_score.py` (m1..m5) come feature di input, ma non viene usato per drift detection o sample weighting nel buffer.

Il modulo `physiq_pv/agent/` contiene `qs_forensics.py` e `action_policy.py` con logica di drift detection basata su QS, ma e **logged-only** — nessun effetto sul training o sul buffer.

---

## 7. Open-Meteo readiness

### Cosa esiste gia

1. **`scripts/load_meteorological_data.py`**: script standalone che carica dati Open-Meteo da `/data/SentinelPV/open_meteo/old_run_history/*.nc`
   - Variabili lette: `temperature_2m`, `wind_speed_10m`, `relative_humidity_2m`, `cloud_cover`, `shortwave_radiation`
   - **Non integrato** nella pipeline di training — output e un CSV `energy_with_meteorology.csv`
   - Usa matching spaziale nearest-neighbor (come PVGIS merge)

2. **Variabili Open-Meteo disponibili** (dal NetCDF esistente):
   - `temperature_2m` ✓ (match diretto)
   - `wind_speed_10m` ✓ (match diretto)
   - `shortwave_radiation` ← candidato per sostituire `solar_irradiance_poa`
   - `relative_humidity_2m` ← extra, non usato nel modello
   - `cloud_cover` ← extra, non usato nel modello

3. **Variabili Open-Meteo NON presenti** nel NetCDF esistente ma disponibili via API:
   - `direct_normal_irradiance` (DNI diretto, non Erbs)
   - `diffuse_radiation` (DHI diretto, non Erbs)

### Cosa manca

- `shortwave_radiation` Open-Meteo e **GHI orizzontale** — sostituto diretto di `solar_irradiance_poa` se questa e effettivamente GHI
- Non esiste funzione `merge_with_openmeteo()` equivalente a `merge_with_weather()`
- Non esiste mapping variabili Open-Meteo → feature pipeline
- DNI/DHI Open-Meteo non sono stati scaricati (solo `shortwave_radiation` presente)

---

## 8. Train-serving mismatch risk

Se il modello e trainato su PVGIS e usato operativamente con Open-Meteo:

| Feature | Rischio | Severita |
|---------|---------|----------|
| `temperature_2m` | Basso. Stessa variabile ERA5 in entrambi | Basso |
| `wind_speed_10m` | Basso. Stessa variabile ERA5 | Basso |
| `solar_irradiance_poa` | **ALTO**. PVGIS potrebbe essere POA (tilt-corrected), Open-Meteo `shortwave_radiation` e GHI orizzontale. Differenza sistematica ~10-30% a latitudini medie | **Alto** |
| `kt` | **MEDIO**. Se solar cambia, kt cambia. Ma come ratio con ghi_cs, potrebbe auto-calibrarsi | Medio |
| `dghi_dt` | Medio. Dipende da solar, ma come differenza temporale il bias sistematico si cancella | Basso |
| `dni_norm` | **ALTO**. Erbs su PVGIS-POA vs Erbs su Open-Meteo-GHI produce valori diversi. Oppure Open-Meteo fornisce DNI diretto → diverso da Erbs | **Alto** |
| `dhi_norm` | **ALTO**. Come sopra | **Alto** |
| `m1..m5` | Medio. QS usa solar_irradiance_poa come reference → cambio sorgente cambia m1,m2,m4,m5 | Medio |
| `sin/cos_elev` | Nessuno. Calcolati da pvlib, indipendenti da meteo | Nessuno |
| `pv_lag` | Nessuno. Derivato da ENERGIA reale | Nessuno |

**Rischio principale**: shift sistematico in `solar_irradiance_poa` che si propaga a 6 feature dirette + 4 feature QS = **10/16 feature potenzialmente affette**.

---

## 9. Mapping proposto Open-Meteo

| Feature attuale | Sorgente attuale | Sorgente Open-Meteo | Note |
|-----------------|-----------------|---------------------|------|
| `temperature_2m` | PVGIS ERA5 | `temperature_2m` | Drop-in replacement |
| `solar_irradiance_poa` | PVGIS NetCDF | `shortwave_radiation` (GHI) | Rinominare a `solar_resource` o `ghi_proxy`. OK se PVGIS e GHI; shift se PVGIS e vera POA |
| `wind_speed_10m` | PVGIS ERA5 | `wind_speed_10m` | Drop-in replacement |
| `sin_solar_elev` | pvlib | pvlib (invariato) | Nessun cambiamento |
| `cos_solar_elev` | pvlib | pvlib (invariato) | Nessun cambiamento |
| `m1..m5` | quality_score.py | quality_score.py (usa nuova solar source) | Ricalcolati automaticamente |
| `pv_lag` | ENERGIA | ENERGIA (invariato) | Nessun cambiamento |
| `kt` | solar_poa / ghi_cs | shortwave_radiation / ghi_cs | Ricalcolato automaticamente |
| `kt_std_3h` | rolling(kt) | rolling(kt) (invariato) | Ricalcolato automaticamente |
| `dghi_dt` | diff(solar_poa) | diff(shortwave_radiation) | Ricalcolato automaticamente |
| `dni_norm` | Erbs(solar_poa) | **`direct_normal_irradiance`** Open-Meteo diretto | Upgrade: DNI reale vs stimato. Erbs solo fallback |
| `dhi_norm` | Erbs(solar_poa) | **`diffuse_radiation`** Open-Meteo diretto | Upgrade: DHI reale vs stimato. Erbs solo fallback |

---

## 10. Raccomandazioni minime

### Piano in 4 fasi

**Fase 0 — Verifica (prima di tutto)**
- Ispezionare `piedmont_pvgis_2019.nc` sul server: `ncdump -h` per vedere attributi, unita, e se contiene metadata su tilt/azimuth
- Se `solar_irradiance_poa` e GHI: tutto il codice e corretto, rinominare sarebbe cosmetico
- Se e vera POA: documentare la discrepanza, Erbs era approssimativo ma funzionante

**Fase 1 — Mantenere legacy PVGIS**
- Non toccare nulla nel branch principale
- Tutti i risultati di tesi usano PVGIS come sorgente

**Fase 2 — Branch `feat/openmeteo-data`**
- Creare `merge_with_openmeteo()` in `sentinel_hourly_loader.py`
- Scaricare NetCDF Open-Meteo con `shortwave_radiation`, `direct_normal_irradiance`, `diffuse_radiation`, `temperature_2m`, `wind_speed_10m`
- Mappare `shortwave_radiation` → variabile che oggi si chiama `solar_irradiance_poa`
- Se DNI/DHI Open-Meteo disponibili: usarli direttamente, Erbs come fallback
- Mantenere N_FEATURES=16 (stesse 16 feature, sorgente diversa)
- Parametro CLI: `--weather-source pvgis|openmeteo`

**Fase 3 — Ablation (opzionale, per tesi)**
- A. PVGIS legacy (baseline)
- B. Open-Meteo GHI proxy (shortwave_radiation come solar_irradiance_poa)
- C. Open-Meteo con DNI/DHI diretti (non Erbs)
- D. Se possibile, Open-Meteo GTI (tilted irradiance) — richiede tilt/azimuth per impianto

---

## File ispezionati

- `physiq_pv/data/dataset.py` — feature construction, PVDataset
- `physiq_pv/data/sentinel_hourly_loader.py` — data loading, merge_with_weather
- `physiq_pv/data/quality_score.py` — m1..m5 computation
- `physiq_pv/data/synthetic_generator.py` — synthetic data
- `physiq_pv/model/st_gnn.py` — STGNN architecture
- `physiq_pv/model/physics_loss.py` — loss function
- `physiq_pv/model/bilstm_encoder.py` — BiLSTM encoder
- `physiq_pv/continual/train_replay_continual.py` — CL pipeline
- `physiq_pv/continual/simple_replay_buffer.py` — replay buffer
- `train.py` — offline training
- `main.py` — entry point
- `scripts/load_meteorological_data.py` — Open-Meteo loader (standalone)
- `scripts/analyze_domain_shift_trend.py` — drift analysis
- `README.md`, `docs/MODEL_REFERENCE.md`, `docs/CONTINUAL_LEARNING.md`

## Punti certi

1. **N_FEATURES=16**, ordine hardcoded, nessun registry
2. **6/16 feature dipendono direttamente da `solar_irradiance_poa`**
3. **DNI/DHI sono stimati via Erbs**, non grezzi
4. **`solar_irradiance_poa` viene trattata come GHI proxy** ovunque nel codice
5. **Fallback pvlib usa GHI clear-sky** e la assegna a `solar_irradiance_poa`
6. **Open-Meteo NetCDF esistono** sul server ma non sono integrati nella pipeline
7. **CL usa stesso PVDataset** con stesse 16 feature
8. **Nessun drift detector** nel CL pipeline attuale

## Punti ambigui

1. **`solar_irradiance_poa` e GHI o POA?** — il nome dice POA, il codice la tratta come GHI, il README dice "ERA5-derived". Serve ispezione del NetCDF originale
2. **PVGIS API call originale** — non si sa con quali parametri e stato generato il NetCDF (tilt=0? tilt=ottimale? automatico?)
3. **Open-Meteo NetCDF completi?** — `shortwave_radiation` presente, DNI/DHI non verificati. Servono file aggiornati con tutte le variabili

## Modifiche consigliate per branch successivo

1. Ispezionare NetCDF PVGIS (`ncdump -h data/piedmont_pvgis_2019.nc`)
2. Creare `merge_with_openmeteo()` speculare a `merge_with_weather()`
3. Scaricare Open-Meteo con DNI/DHI diretti se non presenti
4. Aggiungere `--weather-source pvgis|openmeteo` a CLI
5. Se DNI/DHI Open-Meteo disponibili, condizionare Erbs: `if source == "openmeteo" and has_dni: use direct; else: erbs`
6. Mantenere N_FEATURES=16 in prima fase
7. Run comparativa: PVGIS vs Open-Meteo sugli stessi dati 2019
