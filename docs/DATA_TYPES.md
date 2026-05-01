# PhysiQ-PV - Data Reference

## Dataset

| Campo | Valore |
|---|---|
| Area | Piemonte, Italia |
| Risoluzione | oraria |
| Impianti | 1,116 UPN |
| Produzione | Sentinel/SCADA |
| Meteo | sorgente oraria esterna o fallback clear-sky |

Il codice attuale richiede meteo orario coerente con gli impianti. PVGIS puo' fornirlo per lo storico, ma `pvgis_ref` non e' richiesto.

## Variabili raw

### `ENERGIA`

Produzione Sentinel/SCADA per impianto.

- dimensioni: `(plant, time)`
- unita': kW
- aggregazione: mediana se ci sono piu' letture nella stessa ora
- uso: target PV normalizzato e stima di `pv_scale`

### `solar_irradiance_poa`

Irradianza sul piano o proxy coerente.

- dimensioni: `(plant, time)`
- unita': W/m2
- uso: feature meteo, target `y_ghi`, riferimento QS e stima `eta_adjusted`

### `temperature_2m`

Temperatura ambiente.

- dimensioni: `(plant, time)`
- unita': gradi Celsius
- uso: feature meteo e correzione termica nel QS

### `wind_speed_10m`

Velocita' del vento.

- dimensioni: `(plant, time)`
- unita': m/s
- uso: feature meteo

### Coordinate impianto

`lat`/`lon` o `latitude`/`longitude` sono usate per:
- grafo spaziale
- geometria solare
- matching alla sorgente meteo storica se necessario

## Variabili derivate

### `sin_solar_elev`, `cos_solar_elev`

Calcolate in `PVDataset` con pvlib.

- sostituiscono il vecchio uso di reference power come indicatore di ciclo solare
- sono deterministiche
- non dipendono dalla sorgente meteo

### `pv_scale`

```text
pv_scale[p] = p99(ENERGIA[p] nelle ore diurne)
```

Serve per normalizzare la produzione:

```text
target_pv = clip(ENERGIA / pv_scale, 0.0, 1.5)
```

### `solar_p99`

```text
solar_p99[p] = p99(solar_irradiance_poa[p] / 1000 nelle ore diurne)
```

Serve per normalizzare il riferimento irradiance-based nella stima di `eta_adjusted`.

Nota: `dataset.pvgis_p99` resta come alias legacy verso `solar_p99` per compatibilita' con notebook esistenti. Non indica piu' una dipendenza da PVGIS reference power.

### `eta_adjusted`

Proxy di Performance Ratio per impianto:

```text
solar_norm = solar_irradiance_poa_kwm2 / solar_p99
pv_norm    = ENERGIA / pv_scale
eta_adjusted = median(pv_norm / solar_norm)
```

Clip operativo: `[0.1, eta_max]`, con `eta_max=0.98` in `main.py`.

Il cap e' configurabile. Serve a evitare che troppi impianti saturino a PR=1.0 e spingano il vincolo fisico verso sovrastima.

Uso: target fisico nel termine `L_physics`.

### `QS`

Quality Score in `[0, 1]` per `(plant, time)`.

Formula:

```text
QS = (m1 * m2 * m3 * m4 * m5) ** 0.2
```

Metriche:
- `m1`: correlazione produzione-riferimento irradiance-scaled
- `m2`: bias
- `m3`: completezza
- `m4`: varianza relativa
- `m5`: coerenza fisica con eta termica

### `m1_past`

Feature causale derivata da `m1`.

```text
m1_past(t) = corr(pv_norm[t-window:t-1], solar_norm[t-window:t-1])
```

Usa solo dati precedenti al target, quindi non introduce leakage nel forecast.

## Output dataset

`PVDataset.__getitem__()` restituisce:

| Campo | Shape | Significato |
|---|---|---|
| `x` | `(N, seq_len, 7)` | feature meteo, geometria, QS, `m1_past` |
| `y_ghi` | `(N,)` | irradianza in kW/m2 |
| `y_pv` | `(N,)` | produzione normalizzata |
| `qs` | `(N,)` | QS al timestep target |
| `eta` | `(N,)` | `eta_adjusted` per impianto |

## Loading

```python
ds = load_sentinel_hourly(
    sentinel_dir="/data/SentinelPV/energy_data/piemonte_energy_data/single_ups",
    year=2019,
    plant_mapping_path="data/plant_mapping.csv",
    energy_coords_path="data/energy_with_coordinates.csv",
)

ds = merge_with_weather(ds, pvgis_path="data/piedmont_pvgis_2019.nc")
ds = _normalize_dataset(ds)
qs = compute_qs(ds)
dataset = PVDataset(ds, qs, kwp=kwp)
```

Se il file meteo PVGIS manca, `merge_with_weather()` crea variabili meteo di fallback con pvlib clear-sky, temperatura stagionale e vento costante.

## Limitazioni note

| Limite | Impatto |
|---|---|
| meteo fallback clear-sky | non modella nuvole, quindi non va usato come training source primaria |
| mesi mancanti nello storico | possibile bias stagionale |
| coordinate mancanti | geometria e grafo meno precisi |
| kWp reale assente per alcuni impianti | si usa stima data-driven/fallback fleet |
