# PhysiQ-PV

Forecasting fotovoltaico distribuito su flotta reale con ST-GNN, vincoli fisici e Quality Score multi-componente. Tesi magistrale Politecnico di Torino.

Pipeline data-centric + physics-informed + continual-learning-safe per flotta eterogenea (1116 impianti Piemonte 2019, dati orari Sentinel/SCADA + meteo PVGIS).

## Architettura

ST-GNN dual-head (~160k–200k parametri):
- Encoder PatchTST channel-independent per nodo
- GAT spaziale su grafo geografico (edge ≤ 20km, weight = 1/dist_km)
- Head GHI parametrizzata via clear-sky index: `pred_ghi = sigmoid(head_ghi)*1.2 * ghi_cs` (forza pred_ghi=0 di notte, hard physical bound)
- Head PV: `pred_pv = softplus(head_pv)`

Vincolo fisico moltiplicativo: `L_physics = (pred_pv - eta_T * pred_ghi)^2`. Evita divisione per zero quando `ghi_cs → 0`.

## Feature input (11 canali)

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

`pv_lag` è il segnale autoregressivo dominante sull'accuratezza. m1..m5 sono i componenti separati del Quality Score (geometric mean).

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

Branch `feat/improvements-fleet-2025`, 10 epoche, full fleet, outlier filter attivo:

| KPI | Valore |
|---|---|
| MAE PV | 0.0505 |
| RMSE PV | — |
| r PV | 0.967 |
| bias PV | +0.0056 |
| MAE GHI | 0.0574 |
| MAE bin mid-low QS | 0.0805 |

Best run, -39% vs precedente clearsky senza lagged power.

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
STGNN (PatchTST + GAT + dual-head con kt parametrization)
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
    patchtst_encoder.py
    st_gnn.py
    graph_builder.py
    physics_loss.py
    postprocessing.py
  continual/
    replay_buffer.py
    quality_gated_update.py
  agent/
    cycle.py
    drift_monitor.py
    qs_clustering.py
    causal_classifier.py
  eval/
    benchmark.py
  uncertainty/
    mondrian_cp.py

docs/
  MODEL_REFERENCE.md         architettura, feature, loss, QS, glossario parametri
  CONTINUAL_LEARNING.md      pipeline online + framing data-centric
  LITERATURE_POSITIONING.md  contributo vs SOTA, claim difendibile
  SOTA_REFERENCES.md         survey letteratura
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

## Contributo tesi

Sistema **data-centric + physics-informed + continual-learning-safe** per fleet reale eterogenea. Non architettura più complessa.

QS gioca due ruoli distinti:
- **Modello batch:** segnale soft + diagnostico tramite m1..m5 come feature input. Impatto marginale sul MAE quando lagged power presente.
- **Framework CL (deploy):** load-bearing. Gating update via `QualityGatedUpdater`, drift detection ADWIN, replay buffer DER++, diagnostica per-plant.

Vedi `docs/LITERATURE_POSITIONING.md` per claim difendibile vs SOTA.
