# PhysiQ-PV - Continual Learning Pipeline

## Obiettivo

Aggiornare il modello su nuovi dati mantenendo il percorso di training e inferenza indipendente da `pvgis_ref`.

La pipeline corrente richiede:
- `ENERGIA`
- `temperature_2m`
- `solar_irradiance_poa`
- `wind_speed_10m`
- timestamp, latitudine e longitudine degli impianti

`pvgis_ref` non e' una feature, non entra nel Quality Score e non serve per costruire `eta_adjusted`.

## Feature e target

`PVDataset` produce finestre `(N, seq_len, 7)` con:
- meteo normalizzato
- geometria solare
- QS aggregato
- `m1_past` causale

| Canale | Variabile | Trasformazione |
|---|---|---|
| 0 | `temperature_2m` | z-score globale |
| 1 | `solar_irradiance_poa` | z-score globale |
| 2 | `wind_speed_10m` | z-score globale |
| 3 | `sin_solar_elev` | da pvlib, in `[0, 1]` |
| 4 | `cos_solar_elev` | da pvlib, in `[0, 1]` |
| 5 | `QS` | gia' in `[0, 1]` |

Target:
- `y_pv = clip(ENERGIA / pv_scale, 0.0, 1.5)`
- `y_ghi = solar_irradiance_poa / 1000.0`

`pv_scale` e `eta_adjusted` sono stimati da ore diurne definite con geometria solare e soglia di irradianza, non con PVGIS reference power.

## Quality Score

`compute_qs()` usa `solar_irradiance_poa / 1000.0` come riferimento fisico.

Metriche:
- correlazione rolling tra produzione e riferimento irradiance-scaled
- bias rolling
- completezza dati
- varianza relativa
- coerenza fisica con `eta_base` corretta per temperatura

Le ore senza sole reale vengono gestite separatamente: produzione nulla e sensore valido danno QS alto, produzione in buio fisico da' QS basso.

## Aggiornamento online

Per ogni finestra nuova:
1. caricare produzione e meteo orario coerenti
2. calcolare QS
3. costruire `PVDataset`
4. valutare drift/anomalie
5. aggiornare con replay DER++ usando loss gia' pesata dal QS

Schema replay:

```python
@dataclass
class ReplayEntry:
    x: torch.Tensor
    y_pv: torch.Tensor
    y_ghi: torch.Tensor
    qs: torch.Tensor
    eta: torch.Tensor
    timestamp: datetime
    met_source: str
```

## Sorgenti meteo

Per inferenza su dati nuovi la sorgente consigliata e' un provider meteo operativo, ad esempio Open-Meteo o un servizio interno, con conversione POA tramite pvlib quando serve.

Per retraining offline si puo' usare una reanalysis o un dataset storico piu' stabile.

PVGIS puo' ancora essere usato come sorgente meteo storica se disponibile, ma non e' piu' una dipendenza strutturale della pipeline.

Fallback implementato:
- se `data/piedmont_pvgis_2019.nc` manca, `merge_with_weather()` calcola clear-sky GHI con pvlib
- usa GHI come proxy POA
- usa temperatura stagionale euristica
- usa vento costante a 3 m/s

Questo fallback serve per demo, test e inferenza degradata. Non e' una sostituzione di una sorgente meteo reale per training di qualita'.

## Checkpoint e calibrazione

`train.py` conserva in memoria il `best_state` con minima validation loss e lo ricarica prima che `main.py` salvi `checkpoints/model.pt`.

La calibrazione lineare PV e' configurata da `calibration_kpi`:

| Valore | Abilita la calibrazione se |
|---|---|
| `rmse` | RMSE migliora |
| `mae` | MAE migliora |
| `both` | MAE e RMSE migliorano |
| `none` | mai |

La configurazione operativa corrente usa `calibration_kpi="none"` per valutare il modello senza compressione lineare dei picchi. La calibrazione resta disponibile come esperimento controllato.

## Uso operativo del QS

QS non e' un gate sui dati di training.

La loss usa una pesatura soft:

```text
weight = qs_weight_floor + (1 - qs_weight_floor) * QS^qs_weight_exponent
```

Configurazione corrente:
- `qs_weight_exponent = 0.2`
- `qs_weight_floor = 0.2`

Effetto:
- QS alto pesa vicino a 1
- QS basso pesa meno
- QS nullo pesa comunque 0.2

Questo mantiene informazione anche dai campioni degradati, ma limita il loro impatto.

I KPI sono calcolati dopo il floor fisico a zero, usando lo stesso post-processing dell'inferenza:

```python
pred = slope * pred + intercept
pred = clip(pred, 0.0, None)
```

Il file `checkpoints/pv_calibration.json` riporta:
- `enabled`
- `selection_reason`
- `calibration_kpi`
- `prediction_floor`
- MAE/RMSE prima e dopo
- numero di valori che sarebbero negativi prima del floor
- `best_val_epoch`

## Cosa non serve piu'

Non serve:
- usare `pvgis_ref` come feature
- calcolare QS da `pvgis_ref`
- stimare `eta_adjusted` da `pvgis_ref`
- scaricare PVGIS annuale solo per far partire training o continual learning

Serve ancora:
- una sorgente oraria credibile per irradiance, temperatura e vento
- mapping lat/lon per geometria solare e grafo spaziale
- monitoraggio esplicito della qualita' dei dati nuovi
