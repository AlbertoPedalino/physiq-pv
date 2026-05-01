# PhysiQ-PV - Optimization Log

## Stato Corrente

Configurazione modello corrente:

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
| `max_dist_km` | 10 |
| `batch_size` | 8 |

Feature input: 6 canali, senza `pvgis_ref`.

## Decisioni Mantenute

### Flash SDP

`dropout=0.0` consente l'uso di flash/memory efficient SDP senza il limite pratico incontrato con dropout attivo.

### Grafo Sparso

Il GAT usa aggregazione edge-sparse con `scatter_reduce_` e `scatter_add_`, evitando una matrice densa `(N, N)`.

### Dimensione Modello

`d_model=64`, `gat_dim=96` e un solo layer GAT mantengono il costo computazionale gestibile senza eliminare il ragionamento spazio-temporale.

### Data Augmentation Meteo

Durante training viene applicato rumore moltiplicativo ai canali meteo:

```text
temperature_2m, solar_irradiance_poa, wind_speed_10m
```

Geometria solare e QS non vengono perturbati.

## Correzioni Pipeline

- `pvgis_ref` rimosso da feature, QS e preprocessing corrente.
- `pv_scale` stimato da produzione diurna osservata.
- `solar_p99` stimato da `solar_irradiance_poa / 1000.0`.
- `eta_adjusted` stimato da rapporto tra PV normalizzato e irradiance normalizzata.
- `eta_adjusted` ora usa cap configurabile `eta_max=0.98` per ridurre la sovrastima indotta da saturazione a 1.0.
- `dataset.pvgis_p99` rimane solo come alias legacy verso `solar_p99`.

## Stabilita' Training

- Split train/validation stratificato per mese.
- Early stopping opzionale.
- Checkpoint finale salvato dal best validation epoch.
- Loss asimmetrica sui picchi PV per ridurre sottostima sistematica.

## Calibrazione PV

La calibrazione lineare e' opzionale e guidata da KPI:

| KPI | Regola |
|---|---|
| `rmse` | abilita se RMSE migliora |
| `mae` | abilita se MAE migliora |
| `both` | abilita se migliorano entrambi |
| `none` | disabilita sempre |

Configurazione operativa corrente: `calibration_kpi="none"`.

Configurazione QS corrente:
- `qs_weight_exponent=0.2`
- `qs_weight_floor=0.2`

QS resta un peso soft, non un gate.

Le metriche post-calibrazione sono calcolate dopo il floor fisico a zero.

## Prossime Ottimizzazioni Sensate

1. Rerun training completo con la loss asimmetrica e calibrazione floor-aware.
2. Valutare `calibration_kpi="mae"` solo se il modello raw non peggiora i picchi.
3. Misurare amp_ratio su un set stabile di impianti campione.
4. Implementare una sorgente meteo operativa per inferenza senza PVGIS storico.
