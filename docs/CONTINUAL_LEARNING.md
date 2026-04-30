# PhysiQ-PV - Continual Learning Pipeline (v2, senza dipendenza da pvgis_ref)

---

## 1. Obiettivo

Aggiornare il modello con dati Sentinel nuovi mantenendo il training/online loop indipendente da `pvgis_ref`.

La pipeline corrente usa:
- `ENERGIA`
- `temperature_2m`
- `solar_irradiance_poa`
- `wind_speed_10m`
- geometria solare (`sin_elev`, `cos_elev`)

---

## 2. Stato attuale del codice

### 2.1 Input modello

Feature per timestep e impianto:
1. `temperature_2m` (normalizzata)
2. `solar_irradiance_poa` (normalizzata)
3. `wind_speed_10m` (normalizzata)
4. `sin_solar_elev`
5. `cos_solar_elev`
6. `QS`

Quindi `x` ha shape `(N, seq_len, 6)`.

### 2.2 Quality Score

`compute_qs()` usa come riferimento fisico `solar_irradiance_poa/1000` (kW/m^2), non `pvgis_ref`.

Metriche QS restano le stesse (m1..m5), ma il riferimento meteorologico e' ora irradiance-based.

### 2.3 Preprocessing dataset

`PVDataset`:
- day-mask da geometria + irradianza
- `pv_scale` da p99 produzione diurna
- `eta_adjusted` da rapporto tra PV normalizzato e irradianza normalizzata

Nessuna dipendenza da `pvgis_ref` nel training path.

---

## 3. Pipeline online consigliata (continual learning)

Per ogni batch/finestra nuova servono in tempo quasi reale:
1. `ENERGIA` per impianto
2. `temperature_2m`
3. `solar_irradiance_poa`
4. `wind_speed_10m`
5. timestamp + lat/lon impianto

Con questi dati:
1. si calcola QS
2. si costruiscono feature
3. `QualityGatedUpdater` decide se aggiornare pesi / replay

---

## 4. Sorgenti meteo pratiche

### 4.1 Online (consigliato)

Una sorgente meteo operativa (es. Open-Meteo + conversione POA con pvlib, oppure altra API/fornitore) deve fornire almeno:
- irradianza coerente (diretta o ricostruibile in POA)
- temperatura
- vento

### 4.2 Batch retrain periodico (opzionale)

Per retrain offline piu accurato si puo usare una sorgente reanalysis/storica piu stabile.

Nota: non e' necessario ricostruire `pvgis_ref` se il training path resta quello attuale.

---

## 5. Replay Buffer (schema aggiornato)

```python
@dataclass
class ReplayEntry:
    x: torch.Tensor          # (N, 24, 6)
    y_pv: torch.Tensor       # (N,)
    y_ghi: torch.Tensor      # (N,)
    qs: torch.Tensor         # (N,)
    eta: torch.Tensor        # (N,)
    timestamp: datetime
    met_source: str          # es. "open-meteo", "era5", "internal"
```

---

## 6. Cosa non serve piu

1. Usare `pvgis_ref` come feature modello.
2. Usare `pvgis_ref` per calcolare QS.
3. Dipendere da download PVGIS annuale solo per far funzionare il training.

---

## 7. Rischi operativi e mitigazioni

1. `solar_irradiance_poa` mancante o incoerente
- Impatto: QS e training degradano.
- Mitigazione: validazione range e completezza prima dell'update.

2. Lat/lon mancanti su alcuni impianti
- Impatto: geometria solare meno precisa.
- Mitigazione: fallback controllato + audit periodico mapping impianti.

3. Copertura temporale incompleta (mesi assenti)
- Impatto: stagionalita' parziale, possibili bias.
- Mitigazione: monitor copertura e retrain quando si accumula una finestra completa.

---

## 8. Conclusione

La soluzione convincente, allineata al codice, e':
1. training/continual learning basati su meteo reale + geometria solare
2. QS e vincoli fisici senza `pvgis_ref`
3. sorgente meteo online affidabile come prerequisito operativo

Questo rende la pipeline piu portabile su dati nuovi rispetto alla versione dipendente da PVGIS reference power.
