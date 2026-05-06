# PhysiQ-PV - Data Flow

## Sorgenti

| Sorgente | File | Uso corrente |
|---|---|---|
| Sentinel/SCADA | `2019_UPN_*.csv` | produzione `ENERGIA` per impianto |
| Meteo orario | PVGIS storico, Open-Meteo, ERA5 o provider interno | `solar_irradiance_poa`, `temperature_2m`, `wind_speed_10m` |
| Registry impianti | `plant_mapping.csv`, `energy_with_coordinates.csv` | lat/lon, mapping UPN, kWp reale quando disponibile |

PVGIS e' una possibile sorgente meteo storica. Il training path non usa `pvgis_ref`.

## Loader

`load_sentinel_hourly()` legge i CSV Sentinel, aggrega le letture multiple della stessa ora con mediana e restituisce un `xr.Dataset` con dimensioni `(plant, time)`.

`merge_with_weather()` aggiunge:
- `temperature_2m`
- `solar_irradiance_poa`
- `wind_speed_10m`

Se il NetCDF meteo non esiste, viene usato un fallback pvlib clear-sky. Il fallback e' utile per test e inferenza degradata, non per training accurato.

## Dataset PyTorch

`PVDataset` costruisce:

```text
x[plant, t-23:t, :] =
[
  temperature_2m_z,
  solar_irradiance_poa_z,
  wind_speed_10m_z,
  sin_solar_elev,
  cos_solar_elev,
  QS,
  m1_past
]
```

Shape per sample: `(N, 24, 7)`.

Canali:
- temperatura normalizzata
- irradianza normalizzata
- vento normalizzato
- `sin_solar_elev`
- `cos_solar_elev`
- `QS`
- `m1_past`

La produzione passata non entra direttamente come feature. `QS` e `m1_past` sono feature storiche/osservate nella finestra input; il QS del target viene usato solo dopo osservazione per loss, diagnostica e continual learning.

## Target

```text
target_ghi = solar_irradiance_poa / 1000.0
target_pv  = clip(ENERGIA / pv_scale, 0.0, 1.5)
```

`pv_scale[p]` e' il p99 della produzione diurna osservata per impianto.

Le ore diurne sono identificate con:
- elevazione solare positiva (`sin_elev > 0.05`)
- irradianza minima (`solar_irradiance_poa / 1000 > 0.03`)

## Eta Adjusted

`eta_adjusted[p]` e' una proxy di Performance Ratio usata nel vincolo fisico, stimata via regressione lineare pesata attraverso l'origine.

```text
solar_norm = solar_irradiance_poa_kwm2 / solar_p99[p]
pv_norm    = ENERGIA / pv_scale[p]
w          = solar_norm                                       # peso lineare in irradianza
eta_adjusted[p] = sum(w * solar_norm * pv_norm) / sum(w * solar_norm^2)
```

Stimatore robusto: punti ad alta GHI (SNR alto) dominano, punti a basso GHI (rapporto rumoroso) contribuiscono poco. Calcolato sulle ore diurne.

Clip finale: `[0.1, eta_max]`, con `eta_max=0.98` nella configurazione operativa.

Fallback: mediana fleet per impianti con pochi campioni validi, salvo presenza di kWp reale.

## Quality Score

`compute_qs()` usa `solar_irradiance_poa / 1000.0` come riferimento fisico, scalato per impianto con p99.

Metriche:
- `m1`: correlazione rolling produzione-riferimento
- `m2`: bias rolling
- `m3`: completezza dati
- `m4`: rapporto di varianza
- `m5`: coerenza con eta termica

`QS` entra come feature storica nella finestra input e come peso soft della loss per il target osservato:

```text
weight = qs_weight_floor + (1 - qs_weight_floor) * QS^qs_weight_exponent
```

Configurazione corrente: `weight = 0.2 + 0.8 * QS^0.2`.

Quindi QS non filtra i campioni: abbassa il contributo dei dati di bassa qualita', ma non li elimina.

## Training

```text
Sentinel ENERGIA + meteo orario + lat/lon
        |
        v
load_sentinel_hourly + merge_with_weather
        |
        v
compute_qs
        |
        v
PVDataset
        |
        v
ST-GNN
        |
        v
pred_ghi, pred_pv
        |
        v
physics_loss_full + asymmetric peak loss + quality-aware overprediction loss
```

Il checkpoint salvato e' lo stato con minima validation loss.

## Inferenza

In inferenza servono gli stessi campi meteo usati in training. Dopo il forward:

```text
pred_pv = optional_linear_calibration(pred_pv)
pred_pv = clip(pred_pv, 0.0, None)
```

Il floor a zero e' necessario per mantenere la non-negativita' anche quando la calibrazione lineare ha intercetta negativa.
