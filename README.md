# PhysiQ-PV

Forecasting fotovoltaico distribuito su flotta reale con ST-GNN, vincoli fisici e Quality Score multi-componente. Tesi magistrale Politecnico di Torino.

Pipeline data-centric + physics-informed per flotta eterogenea (1116 impianti Piemonte 2019, dati orari Sentinel/SCADA + meteo PVGIS).

## Architettura

ST-GNN dual-head (~160k–200k parametri):
- Encoder BiLSTM per nodo
- GAT spaziale su grafo geografico (edge ≤ 20km, weight = 1/dist_km)
- Head GHI parametrizzata via clear-sky index: `pred_ghi = sigmoid(head_ghi)*1.2 * ghi_cs` (forza pred_ghi=0 di notte, hard physical bound)
- Head PV: `pred_pv = softplus(head_pv)`

Vincolo fisico moltiplicativo: `L_physics = (pred_pv - eta_T * pred_ghi)^2`. Evita divisione per zero quando `ghi_cs → 0`.

## Feature input (16 canali)

| Ch | Feature | Trasformazione |
|---|---|---|
| 0 | `temperature_2m` | z-score |
| 1 | `solar_irradiance_poa` | z-score |
| 2 | `wind_speed_10m` | z-score |
| 3 | `sin_solar_elev` | pvlib `[0,1]` |
| 4 | `cos_solar_elev` | pvlib `[0,1]` |
| 5 | `m1` corr_score | rolling 720h |
| 6 | `m2` bias_score | rolling 720h |
| 7 | `m3` nan_score | completeness |
| 8 | `m4` var_score | std ratio |
| 9 | `m5` eta_score | coerenza eta_T |
| 10 | `pv_lag` | target_pv_norm passato (causale, slice `t-seq_len:t`) |
| 11 | `kt` | clearness index `solar_poa/ghi_cs`, threshold ghi_cs > 0.1 |
| 12 | `kt_std_3h` | std rolling 3h di `kt` (variabilità nuvole) |
| 13 | `dghi_dt` | first-difference `solar_poa`, z-score (ramp rate) |
| 14 | `dni_norm` | DNI via Erbs decomposition, kW/m² (componente diretta) |
| 15 | `dhi_norm` | DHI via Erbs decomposition, kW/m² (componente diffusa) |

`pv_lag` è il segnale autoregressivo dominante sull'accuratezza. m1..m5 sono i componenti separati del Quality Score. QS aggregato `(m1·m2·m3·m4·m5)^0.2` **non entra nel modello**. Canali 11-13 catturano dinamica nuvole istantanea, canali 14-15 separano radiazione diretta da diffusa via Erbs (disambiguano regime nuvoloso vs sereno).

## Target

```text
y_ghi = solar_irradiance_poa / 1000.0          [kW/m²]
y_pv  = clip(ENERGIA / pv_scale, 0, 1.5)      [normalizzato]
```

## Loss

```text
L = MSE(pred_ghi, y_ghi) + MSE(pred_pv, y_pv) + lam * L_physics
  + peak_loss_weight * L_peak_asymmetric
```

`L_peak_asymmetric` penalizza sottostima dei picchi PV con under_penalty=2.0, peak_alpha=2.0, peak_gamma=2.0. Configurazione operativa: `peak_loss_weight=0.25`, `lam=0.1`.

## Risultati run corrente

Branch `feat/improvements-fleet-2025`, 10 epoche, full fleet, outlier filter **disabilitato**, n=3,061,186 daytime samples:

| KPI | Valore |
|---|---|
| MAE PV | 0.0498 |
| RMSE PV | 0.0842 |
| r PV | 0.967 |
| bias PV | +0.0048 |
| MAE GHI | 0.0564 |
| RMSE GHI | 0.0830 |
| r GHI | 0.952 |
| bias GHI | −0.0111 |
| MAE bin mid-low QS | 0.0729 |
| Best val epoch | 9 (val=0.0253) |

Train loss drop totale -68.2%. Per-plant time series r≈0.98 (plant 0/500/1115).

## Pipeline

```text
Sentinel CSV + PVGIS NetCDF + plant_mapping
        │
        ▼
load_sentinel_hourly + merge_with_weather → xr.Dataset
        │
        ▼
compute_qs → m1..m5 components + QS aggregato
        │
        ▼
PVDataset (11 feature, finestra 24h, ghi_cs via Ineichen)
        │
        ▼
STGNN (BiLSTM + GAT + dual-head con kt parametrization)
        │
        ▼
pred_ghi, pred_pv
```

## Repository

```text
main.py
train.py

physiq_pv/
  data/
    sentinel_hourly_loader.py
    dataset.py
    quality_score.py
    load_kwp.py
    synthetic_generator.py
  model/
    bilstm_encoder.py
    st_gnn.py
    graph_builder.py
    physics_loss.py
    postprocessing.py
  eval/
    benchmark.py

docs/
  MODEL_REFERENCE.md         architettura, feature, loss, QS, glossario parametri
  SOTA_REFERENCES.md         survey letteratura
  EXPERIMENTS.md             baseline persistence, ablation L=1, inference puntuale

scripts/
  experiments/
    persistence_baseline.py        baseline naive y_pred(t) = y_true(t-1)
    single_hour_inference.py       inference puntuale su 1 timestamp val
```

## Training

```powershell
uv run python main.py
```

Output:

```text
checkpoints/
  model.pt              best validation epoch
  loss_history.json     train/val per epoch
  model_config.json     iperparametri architettura
  training_config.json  eta_max, calibration_kpi
  pv_calibration.json   slope/intercept opzionale + KPI
```

`uv` path Windows: `C:\Users\alber\.local\bin\uv.exe` (prepend `$env:PATH`).

## Meteo

Sorgente primaria: `data/piedmont_pvgis_2019.nc` (PVGIS reanalysis ERA5-derived). Variabili: `solar_irradiance_poa`, `temperature_2m`, `wind_speed_10m`. Fallback pvlib clear-sky disponibile per demo, non per training accurato.

## Esperimenti e baseline

Vedi `docs/EXPERIMENTS.md` per dettagli completi. Riepilogo veloce:

| Esperimento | Scopo | Comando |
|---|---|---|
| Persistence baseline | naive `y_pred(t) = y(t-1)` come pavimento assoluto | `python scripts/experiments/persistence_baseline.py --wandb` |
| Single-hour inference | predizione modello già trainato su 1 timestamp | `python scripts/experiments/single_hour_inference.py --wandb` |
| Ablation ST-GNN L=1 | training completo con finestra 1h vs 24h | `python main.py` (branch `feat/persistence-baseline`) |

## Contributo tesi

Sistema **data-centric + physics-informed** per fleet reale eterogenea. Non architettura più complessa.

QS è segnale soft + diagnostico tramite m1..m5 come feature input del modello batch. Impatto marginale sul MAE quando lagged power presente.
