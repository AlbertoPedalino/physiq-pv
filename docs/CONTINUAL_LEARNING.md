# PhysiQ-PV — Continual Learning Pipeline

---

## 1. Problema

Il modello è addestrato su dati storici PVGIS 2019. In produzione arrivano dati Sentinel in tempo reale. Per aggiornare il modello servono:

1. **QS** per filtrare campioni corrotti e pesare la loss
2. **pvgis_ref** e **temperature_2m** per calcolare QS e il vincolo fisico `L_physics`
3. **target_ghi** per supervisionare la testa GHI

PVGIS classico è un servizio batch (download per anno/periodo) — non adatto a pipeline real-time.

---

## 2. Soluzione: pipeline ibrida

### 2.1 Online — per ogni nuovo batch Sentinel

**Sorgenti**: Open-Meteo API (free, latenza ~1h) + pvlib

```
Open-Meteo → temperature_2m          (diretto)
Open-Meteo → shortwave_radiation GHI (reale, nuvole incluse)
pvlib       → GHI → POA              (geometria solare + orientamento medio fleet)
pvlib       → pvgis_ref_proxy = POA / 1000 × eta_mean
```

Output: `QS_approx` calcolato con `compute_qs()` invariato.

**Uso**: `QualityGatedUpdater` decide se campione entra in `ReplayBuffer`.

### 2.2 Batch — al momento del retrain (settimanale/mensile)

**Sorgente**: ERA5 / Copernicus CDS (latenza 5+ giorni, qualità PVGIS-like)

```
ERA5 → temperature_2m, GHI/DNI/DHI
pvlib → POA → pvgis_ref_proxy_era5
```

**Uso**: ricalcolo `eta_adjusted` più preciso → `L_physics` più fedele alla fisica reale.

---

## 3. Perché QS non cambia drasticamente tra le due sorgenti

QS misura **ratio e correlazione** tra produzione reale e riferimento meteorologico:

```
m1 = Pearson(ENERGIA, pvgis_ref)          → shape del profilo giornaliero
m2 = 1 - |mean(real-ref)| / mean(ref)    → offset sistematico
m4 = clip(std_real / std_ref, 0, 1)      → sensore bloccato
m5 = 1 - mean(max(0, 1 - PR/eta_T))      → consistenza fisica
```

Se impianto sano:
- Open-Meteo reference e ERA5 reference differiscono in valore assoluto (5–8% RMSE) ma seguono lo stesso pattern temporale (nuvole, ciclo giornaliero)
- `real/ref ≈ 1` in entrambi i casi → QS alto in entrambi

Il flip alto→basso richiederebbe Open-Meteo sbagliato sul **pattern temporale** (non solo scala assoluta). Possibile solo in vallate alpine con orografia complessa — raro e limitato geograficamente.

**Conclusione**: Open-Meteo + pvlib fornisce QS affidabile per filtering. ERA5 non è necessario per QS, ma migliora la qualità del vincolo fisico al retrain.

---

## 4. Dove le due sorgenti divergono davvero

| Quantità | Impatto Open-Meteo vs ERA5 |
|----------|---------------------------|
| QS direction (alto/basso) | Uguale — pattern temporale concorda |
| QS valore assoluto | Differenza ~5% — trascurabile per weight `QS^0.2` |
| `eta_adjusted` | Open-Meteo: accuratezza ~90% vs PVGIS. ERA5: ~95% |
| `L_physics` vincolo | Leggermente meno preciso con Open-Meteo |
| `target_ghi` | Open-Meteo sufficiente per supervisione online |

---

## 5. Schema ReplayBuffer

```python
@dataclass
class ReplayEntry:
    x: torch.Tensor          # (N, 24, 5)
    y_pv: torch.Tensor       # (N,)
    y_ghi: torch.Tensor      # (N,)
    qs: torch.Tensor         # (N,) — QS_approx da Open-Meteo
    eta: torch.Tensor        # (N,) — eta_adjusted da Open-Meteo
    timestamp: datetime
    met_source: str          # "open-meteo" | "era5"
    # eta_era5: opzionale, aggiornato al batch retrain se ERA5 disponibile
```

Al retrain: se ERA5 disponibile per quel periodo → aggiorna `eta` con versione ERA5 prima di calcolare `L_physics`. QS rimane quello Open-Meteo (differenza trascurabile).

---

## 6. Dipendenze

```
pip install openmeteo-requests pvlib
```

### 6.1 Open-Meteo fetch

```python
import openmeteo_requests
import requests_cache
from retry_requests import retry

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
om = openmeteo_requests.Client(session=retry_session)

params = {
    "latitude": lat,
    "longitude": lon,
    "hourly": ["temperature_2m", "shortwave_radiation",
               "direct_normal_irradiance", "diffuse_radiation"],
    "timezone": "Europe/Rome",
    "start_date": date_start,
    "end_date": date_end,
}
response = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
```

### 6.2 pvlib GHI → POA

```python
import pvlib

location = pvlib.location.Location(latitude=lat, longitude=lon, tz="Europe/Rome")
solar_pos = location.get_solarposition(times)

# Orientamento medio fleet Piemonte (stima)
SURFACE_TILT = 30      # gradi
SURFACE_AZIMUTH = 180  # Sud

poa = pvlib.irradiance.get_total_irradiance(
    surface_tilt=SURFACE_TILT,
    surface_azimuth=SURFACE_AZIMUTH,
    solar_zenith=solar_pos["apparent_zenith"],
    solar_azimuth=solar_pos["azimuth"],
    dni=dni, ghi=ghi, dhi=dhi,
)
pvgis_ref_proxy = poa["poa_global"] / 1000.0  # W/m² → kW/kWp proxy
```

---

## 7. Limitazioni note

| Limitazione | Impatto |
|------------|---------|
| Orientamento pannelli sconosciuto | Errore POA ~10% per impianti con tilt/azimuth non standard |
| Open-Meteo risoluzione 11km (ERA5) / 2km (ICON) | Imprecisione in aree con orografia complessa (Alpi) |
| ERA5 latenza 5+ giorni | Vincolo fisico retrain leggermente ritardato rispetto ai dati |
| pvgis_ref_proxy ≠ simulazione impianto 1kWp | Manca modello termico pannello (perdite temperatura) — errore sistematico ~2-3% |
