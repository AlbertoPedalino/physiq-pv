# PhysiQ-PV

Forecasting fotovoltaico distribuito con ST-GNN, vincoli fisici e Quality Score.

La pipeline corrente usa produzione Sentinel/SCADA, meteo orario, geometria solare e QS. `pvgis_ref` non e' una feature del modello e non serve per QS, `eta_adjusted` o continual learning.

QS e' usato come peso soft, non come gate: la configurazione corrente usa `weight = 0.2 + 0.8 * QS^0.2`. Il modello riceve anche `m1_past`, una correlazione rolling causale PV-irradianza, per distinguere i casi in cui il QS basso dipende da bassa coerenza temporale con il sole.

Nota operativa: nelle feature entrano solo QS e `m1_past` storici/osservati nella finestra input. Il QS del target viene usato dopo osservazione per pesare la loss, diagnostica e continual learning.

## Pipeline

1. `load_sentinel_hourly()` carica i CSV orari Sentinel.
2. `merge_with_weather()` aggiunge `temperature_2m`, `solar_irradiance_poa`, `wind_speed_10m`.
3. `compute_qs()` calcola il Quality Score irradiance-based.
4. `PVDataset` costruisce finestre `(N, 24, 7)`.
5. `train.py` addestra ST-GNN con loss fisica, loss asimmetrica sui picchi, penalita' quality-aware sulla sovrastima PV e checkpoint best-val.
6. `main.py` salva modello, storico loss, configurazione e calibrazione PV.

## Feature Modello

| Canale | Feature |
|---|---|
| 0 | `temperature_2m` |
| 1 | `solar_irradiance_poa` |
| 2 | `wind_speed_10m` |
| 3 | `sin_solar_elev` |
| 4 | `cos_solar_elev` |
| 5 | `QS` |
| 6 | `m1_past` |

Target:
- `pred_ghi`: irradiance in kW/m2
- `pred_pv`: produzione normalizzata per impianto

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
  DATA_TYPES.md
  DATA_FLOW.md
  TRAINING_ARCHITECTURE.md
  MODEL_REFERENCE.md
  CONTINUAL_LEARNING.md
  QS_DATACENTRIC_CL.md
  LITERATURE_POSITIONING.md
```

## Training

```powershell
uv run python main.py
```

Output:

```text
checkpoints/
  model.pt
  loss_history.json
  model_config.json
  training_config.json
  pv_calibration.json
```

`model.pt` contiene il best validation epoch. `model_config.json` contiene solo gli argomenti architetturali di `STGNN`. `training_config.json` contiene pesatura QS, `eta_max` e `calibration_kpi`. `pv_calibration.json` contiene KPI prima/dopo, criterio di selezione e floor fisico a zero.

La configurazione operativa corrente usa `calibration_kpi="none"` per non comprimere i picchi con una calibrazione lineare. Il vincolo fisico usa `eta_max=0.98` per evitare saturazione eccessiva del PR proxy.

## Meteo

PVGIS puo' essere usato come sorgente meteo storica tramite `data/piedmont_pvgis_2019.nc`. Per dati nuovi servono variabili meteo equivalenti da un provider operativo o reanalysis.

Se il file PVGIS manca, `merge_with_weather()` usa un fallback clear-sky via pvlib. Il fallback e' adatto a test e demo, non a training accurato.

## Documentazione

- `docs/DATA_TYPES.md`: variabili e quantita' derivate
- `docs/DATA_FLOW.md`: flusso end-to-end
- `docs/TRAINING_ARCHITECTURE.md`: modello, loss, checkpoint e calibrazione
- `docs/MODEL_REFERENCE.md`: riferimento dettagliato architettura, QS, loss, KPI
- `docs/CONTINUAL_LEARNING.md`: strategia online
- `docs/QS_DATACENTRIC_CL.md`: framing QS come framework data-centric per CL
- `docs/LITERATURE_POSITIONING.md`: posizionamento accademico e ablation principale
