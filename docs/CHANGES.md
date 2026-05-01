# PhysiQ-PV - Changes

Questo file riassume lo stato corrente delle modifiche rilevanti rispetto alla baseline iniziale.

## Data Pipeline

- `pvgis_ref` e' stato rimosso dal percorso corrente di training.
- Le feature modello sono ora meteo orario, geometria solare e QS.
- `PVDataset` usa `solar_irradiance_poa / 1000.0` come riferimento irradiance-based.
- `pv_scale`, `solar_p99` ed `eta_adjusted` sono calcolati senza PVGIS reference power.
- `dataset.pvgis_p99` resta solo come alias legacy verso `solar_p99` per notebook esistenti.

## Feature Layout

| Canale | Feature |
|---|---|
| 0 | `temperature_2m` |
| 1 | `solar_irradiance_poa` |
| 2 | `wind_speed_10m` |
| 3 | `sin_solar_elev` |
| 4 | `cos_solar_elev` |
| 5 | `QS` |

`N_FEATURES = 7` dopo l'aggiunta di `m1_past` causale come feature diagnostica.

## Training

- Split train/validation stratificato per mese.
- Checkpoint finale salvato dal best validation epoch, non dall'ultimo epoch.
- Aggiunta loss asimmetrica sui picchi PV per penalizzare maggiormente la sottostima.
- Aggiunta penalita' quality-aware sulla sovrastima PV quando QS e `m1_past` sono bassi.
- Data augmentation meteo: rumore moltiplicativo sui canali 0-2 durante training.

## Calibrazione e Post-Processing

- Aggiunta calibrazione lineare PV configurabile con `calibration_kpi`.
- KPI supportati: `rmse`, `mae`, `both`, `none`.
- I KPI post-calibrazione sono calcolati dopo il floor fisico a zero.
- Il post-processing operativo applica:

```text
pred_pv = slope * pred_pv + intercept
pred_pv = clip(pred_pv, 0.0, None)
```

## Meteo e Inferenza

- PVGIS puo' ancora fornire meteo storico, ma non e' una dipendenza strutturale.
- Se il NetCDF PVGIS non e' disponibile, `merge_with_weather()` usa fallback pvlib clear-sky.
- Per inferenza reale su nuovi dati serve una sorgente meteo oraria affidabile, ad esempio Open-Meteo, ERA5 o provider interno.

## Documentazione

Documenti aggiornati allo stato corrente:
- `docs/DATA_TYPES.md`
- `docs/DATA_FLOW.md`
- `docs/TRAINING_ARCHITECTURE.md`
- `docs/CONTINUAL_LEARNING.md`
