# PhysiQ-PV: presentation evidence pack

Data di verifica: **2026-06-10**

## Executive summary

- La pipeline principale e' **PVGIS-only**: usa esclusivamente variabili PVGIS,
  geometria solare e lag del target PVGIS. Non usa produzione reale osservata,
  `ENERGIA`, Sentinel/SCADA, kWp/UPN o quality score.
- Il protocollo paper-style usa **train 2016-2018, test 2019**, ST-GNN, MC
  Dropout con 20 passaggi e intervalli empirici q0.025/q0.975, senza
  calibrazione post-hoc.
- MC Dropout e' fortemente under-dispersed: **PICP 0.028**, contro target 0.95.
- Il Deep Ensemble di 5 seed aumenta PICP a **0.173** e migliora MAE globale a
  **19.19**, ma resta molto sotto il target di copertura.
- La variante post-hoc raggiunge **PICP 0.949**, ma solo espandendo gli
  intervalli con fattori calibrati molto grandi (`k_global` medio circa 74.8).
  Non e' evidenza che l'incertezza nativa del modello sia calibrata.

## 1. Setup pipeline PVGIS-only

### Split e protocollo

| Protocollo | Train | Calibration | Test | Uso principale |
|---|---:|---:|---:|---|
| MC Dropout paper-style, sweep `7c8llckm` | 2016, 2017, 2018 | nessuna | 2019 | baseline principale |
| Deep Ensemble, sweep `vfry1hgx` | 2016, 2017, 2018 | nessuna | 2019 | confronto principale |
| Post-hoc calibrated, sweep `ozii0s5s` | 2016, 2017 | 2018, group | 2019 | confronto secondario |

La normalizzazione e' fittata esclusivamente sui dati di training e poi
applicata a calibration/test (`physiq_pv/data/pvgis_stgnn_dataset.py:354-443`).
Le finestre non attraversano gli anni. Configurazione comune dei protocolli
principali:

- `seq_len=24`, `horizon=1`
- `max_train_samples=50000`
- `epochs=5`, `batch_size=16`, `lr=0.001`
- `dropout=0.2`
- seed 1, 2, 3, 4, 5
- `coverage_target=0.95`, `clc_eta=10`

Fonti:

- `configs/sweeps/pvgis_stgnn_seed_only_paper_style.yaml`
- `configs/sweeps/pvgis_stgnn_deep_ensemble_members.yaml`
- `configs/sweeps/pvgis_stgnn_seed_only.yaml`

### Feature set e target

`feature_set=full` contiene 11 feature:

1. `temperature_2m`
2. `solar_irradiance_poa`
3. `wind_speed_10m`
4. `sin_elev`
5. `cos_elev`
6. `kt`
7. `kt_std_3h`
8. `dghi_dt`
9. `dni_norm`
10. `dhi_norm`
11. `pv_lag_pvgis`

Il target e' **`pv_power_output` PVGIS**, un'ora avanti. Il task e' quindi
PVGIS passato -> PVGIS futuro, non previsione di produzione reale misurata.

Fonti:

- `physiq_pv/data/pvgis_stgnn_dataset.py:2-21`
- `physiq_pv/data/pvgis_stgnn_dataset.py:40-65`
- `physiq_pv/data/pvgis_stgnn_dataset.py:354-443`

### Modello ST-GNN

Configurazione effettiva PVGIS:

- encoder temporale BiLSTM bidirezionale, 2 layer, attention pooling,
  `d_model=128`;
- proiezione a `gat_dim=96`;
- un layer GAT geografico, 4 attention heads;
- grafo non diretto fra localita' entro 20 km, peso arco `1 / distanza_km`
  (default CLI `--max-dist-km 20`; gli sweep non lo sovrascrivono — il default
  50 km in `graph_builder.py` non e' quello effettivo);
- testa PV con output non negativo (`softplus`);
- training sul target PV normalizzato con MSE e AdamW.

Fonti:

- `physiq_pv/data/pvgis_stgnn_dataset.py:450-485`
- `physiq_pv/model/st_gnn.py:76-187`
- `physiq_pv/model/graph_builder.py:14-59`

### MC Dropout

- `mc_dropout=true`
- `mc_samples=20`
- inferenza con modello in `eval()` e riattivazione delle sole `nn.Dropout`;
- previsione puntuale = media dei 20 passaggi;
- incertezza = deviazione standard dei 20 passaggi;
- intervallo primario = quantili empirici q0.025/q0.975;
- banda gaussiana `mean +/- 1.96 * std` solo diagnostica;
- nessuna calibrazione post-hoc nel protocollo paper-style.

Fonti:

- `configs/sweeps/pvgis_stgnn_seed_only_paper_style.yaml`
- `physiq_pv/data/pvgis_stgnn_dataset.py:586-699`
- `physiq_pv/experiments/pvgis_stgnn_runner.py:709-750`

### Deep Ensemble

- 5 ST-GNN indipendenti, seed 1-5;
- ogni membro usa la stessa configurazione del paper-style;
- per ogni membro viene salvata la media MC per campione (`y_pred_mean`);
- il vero ensemble combina **per campione** le 5 medie dei membri;
- previsione puntuale = media fra i 5 membri;
- incertezza = deviazione standard fra i 5 membri;
- PI primario = quantili empirici q0.025/q0.975 fra i 5 membri;
- min/max riportato solo come diagnostica, perche' 5 membri rendono i quantili
  molto grossolani;
- nessuna calibrazione post-hoc.

Quindi non e' una semplice media delle metriche dei seed: le predizioni sono
allineate per `sample_id` e aggregate prima di calcolare le metriche.

Fonti:

- `configs/sweeps/pvgis_stgnn_deep_ensemble_members.yaml`
- `physiq_pv/experiments/pvgis_stgnn_runner.py:274-294`
- `scripts/analyze_pvgis_deep_ensemble.py:1-15`
- `scripts/analyze_pvgis_deep_ensemble.py:242-313`

### Assenza di calibrazione nel paper-style

Nel protocollo paper-style:

- non c'e' calibration year;
- `enable_posthoc_calibration=False` nei CSV del sweep MC Dropout;
- `posthoc_calibration=false` nel run di aggregazione Deep Ensemble;
- PICP e' misurato, non forzato;
- gli intervalli derivano direttamente dai campioni MC o dalle predizioni dei
  membri.

La variante `ozii0s5s`, invece, stima su 2018 fattori separati per
`normal` e `rare_or_extreme`, poi applica bande calibrate sul test 2019.

## 2. Sweep e run ID

### `7c8llckm`: paper-style MC Dropout

- W&B sweep: `7c8llckm`
- Stato verificato: `FINISHED`
- URL: <https://wandb.ai/albertopedalino-politecnico-di-torino/PhysiQ-PV/sweeps/7c8llckm>

| Seed | Run name | Run ID |
|---:|---|---|
| 1 | `balmy-sweep-1` | `zq36yt8a` |
| 2 | `feasible-sweep-2` | `gxzmgdgf` |
| 3 | `brisk-sweep-3` | `nigld4wx` |
| 4 | `astral-sweep-4` | `k0wg2dew` |
| 5 | `golden-sweep-5` | `m9hmrjyt` |

### `vfry1hgx`: Deep Ensemble

- W&B sweep membri: `vfry1hgx`
- Nome W&B: `Deep Ensamble`
- Stato verificato: `FINISHED`
- URL: <https://wandb.ai/albertopedalino-politecnico-di-torino/PhysiQ-PV/sweeps/vfry1hgx>

| Seed | Run name | Run ID |
|---:|---|---|
| 1 | `lemon-sweep-1` | `tz5lfk5v` |
| 2 | `ethereal-sweep-2` | `1ct8hiug` |
| 3 | `vocal-sweep-3` | `g9rpzz8q` |
| 4 | `swept-sweep-4` | `0wy9x013` |
| 5 | `pleasant-sweep-5` | `v09urucu` |

Run di aggregazione per-sample:

- run ID: `29yahjvt`
- run name: `pvgis_deep_ensemble_vfry1hgx`
- URL: <https://wandb.ai/albertopedalino-politecnico-di-torino/PhysiQ-PV/runs/29yahjvt>
- `n_models=5`, `n_samples=10,037,664`
- commit registrato: `a38c3f7646d7e979982f3a2409b88680dbc6e470`

### `ozii0s5s`: post-hoc calibrated variant

- sweep ID registrato nel report/CSV: `ozii0s5s`
- 5 run finiti:

| Seed | Run name | Run ID |
|---:|---|---|
| 1 | `blooming-sweep-1` | `xh4jf44r` |
| 2 | `faithful-sweep-2` | `h4v9pb7g` |
| 3 | `scarlet-sweep-3` | `7db07oen` |
| 4 | `pleasant-sweep-4` | `v48c6sdo` |
| 5 | `polar-sweep-5` | `5a9mas9s` |

Nota di provenienza: al 2026-06-10 l'API W&B non risolve piu' l'oggetto sweep
`ozii0s5s`; i cinque run sono ancora accessibili nel progetto `PhysiQ-PV`, ma
risultano senza associazione live allo sweep. Per questo confronto il record
congelato e riproducibile e' il CSV/report locale elencato nella sezione
"Evidenze".

## 3. Risultati chiave

### Tabella completa: metriche per strato

Valori MC Dropout e calibrated = media sui 5 seed. Valori Deep Ensemble = unica
aggregazione per-sample dei 5 membri.

| Protocollo | Strato | MAE | PICP | MPIW | NMPIL | CLC |
|---|---|---:|---:|---:|---:|---:|
| MC Dropout `7c8llckm` | global | 20.3446 | 0.0282 | 3.1613 | 0.0035 | 35.6295 |
| MC Dropout `7c8llckm` | normal | 19.2154 | 0.0282 | 3.1087 | 0.0035 | 35.0376 |
| MC Dropout `7c8llckm` | rare/extreme | 30.7416 | 0.0283 | 3.6461 | 0.0041 | 41.0863 |
| Deep Ensemble `vfry1hgx` | global | 19.1925 | 0.1726 | 15.1574 | 0.0170 | 40.3814 |
| Deep Ensemble `vfry1hgx` | normal | 18.0754 | 0.1705 | 14.6325 | 0.0164 | 39.8026 |
| Deep Ensemble `vfry1hgx` | rare/extreme | 29.4780 | 0.1917 | 19.9903 | 0.0224 | 43.9733 |
| Calibrated `ozii0s5s` | global | 20.9045 | 0.9487 | 129.0545 | 0.1445 | 0.2909 |
| Calibrated `ozii0s5s` | normal | 19.7577 | 0.9494 | 124.7034 | 0.1396 | 0.2802 |
| Calibrated `ozii0s5s` | rare/extreme | 31.4633 | 0.9426 | 169.1161 | 0.1893 | 0.3933 |

Per `ozii0s5s` la tabella usa gli intervalli **calibrati**. Gli intervalli raw
dello stesso sweep restano under-dispersed:

- PICP raw global: 0.0283
- MPIW raw global: 3.3697
- NMPIL raw global: 0.0038
- CLC raw global: 37.9417

### Ratio rare/normal

| Protocollo | MAE rare/normal | Uncertainty rare/normal |
|---|---:|---:|
| MC Dropout `7c8llckm` | 1.6016 | 1.1731 |
| Deep Ensemble `vfry1hgx` | 1.6308 | 1.3676 |
| Calibrated `ozii0s5s` | 1.5931 | 1.1822 |

L'errore rare/extreme cresce di circa 59-63% in tutti i protocolli. MC Dropout
aumenta lo spread solo del 17%; il Deep Ensemble segue meglio il cambio di
regime, con spread +37%, ma ancora meno dell'aumento dell'errore.

### Definizione delle metriche

- `PICP = mean(y_true in [lower, upper])`
- `MPIW = mean(upper - lower)`
- `NMPIL = MPIW / target_range`
- `CLC = NMPIL * (1 + exp(-eta * (PICP - gamma)))`
- qui `gamma=0.95`, `eta=10`

Fonte: `physiq_pv/experiments/pvgis_stgnn_runner.py:124-159`.

## 4. Confronto finale

| Aspetto | MC Dropout paper-style | Deep Ensemble | Post-hoc calibrated |
|---|---|---|---|
| Oggetto statistico | media risultati di 5 modelli seed | un ensemble per-sample di 5 modelli | media risultati di 5 modelli seed |
| Calibration | nessuna | nessuna | group calibration su 2018 |
| MAE global | 20.34 | **19.19** | 20.90 |
| MAE rare | 30.74 | **29.48** | 31.46 |
| PICP global | 0.028 | **0.173** fra i non calibrati | **0.949** |
| MPIW global | **3.16** | 15.16 | 129.05 |
| CLC global | 35.63 | 40.38 | **0.291** |
| Uncertainty ratio rare/normal | 1.17 | **1.37** | 1.18 raw |
| Lettura | molto sharp, non affidabile | piu' dispersione e migliore point forecast, ma ancora under-covered | coverage quasi target, ottenuta tramite forte riscaling post-hoc |

### Messaggi da portare in presentazione

1. **Deep Ensemble migliora il point forecast.** Rispetto a MC Dropout riduce
   MAE globale del 5.7% e MAE rare/extreme del 4.1%.
2. **Deep Ensemble produce incertezza piu' informativa.** PICP sale da 2.8% a
   17.3% e il ratio uncertainty rare/normal da 1.17 a 1.37.
3. **Deep Ensemble non risolve la calibrazione.** Il PICP 0.173 resta lontano
   dal target 0.95; MPIW e' 4.8 volte MC Dropout e CLC e' comunque peggiore
   (40.38 contro 35.63).
4. **La coverage 0.95 appartiene solo alla variante post-hoc.** `ozii0s5s`
   richiede `k_global` medio 74.8 e intervalli globali circa 40.8 volte piu'
   larghi del paper-style MC Dropout.
5. **Il confronto calibrated e' secondario.** `ozii0s5s` usa 2018 per
   calibration anziche' training, mentre i due protocolli principali allenano
   anche su 2018. Le differenze di MAE non isolano quindi il solo effetto della
   calibrazione.
6. **CLC va letto insieme a PICP e MPIW.** La variante calibrata ottiene CLC
   molto basso perche' porta PICP vicino a `gamma`; non significa che lo spread
   nativo del modello fosse corretto.

## 5. Evidenze e provenienza

### File locali usati

| Evidenza | Path | Uso |
|---|---|---|
| CSV sweep MC Dropout | `outputs/sweep_analysis/7c8llckm.csv` | metriche/run per seed |
| Report MC Dropout | `outputs/sweep_analysis/7c8llckm_report.md` | setup, aggregati e interpretazione |
| CSV calibrated | `outputs/sweep_analysis/ozii0s5s.csv` | metriche/run per seed |
| Report calibrated | `outputs/sweep_analysis/ozii0s5s_report.md` | aggregati raw/calibrated e fattori `k` |
| Confronto preesistente | `outputs/sweep_analysis/paper_style_vs_calibrated_comparison.csv` | controllo MC vs calibrated |
| Config MC Dropout | `configs/sweeps/pvgis_stgnn_seed_only_paper_style.yaml` | protocollo principale |
| Config Deep Ensemble | `configs/sweeps/pvgis_stgnn_deep_ensemble_members.yaml` | membri e salvataggio NPZ |
| Config calibrated | `configs/sweeps/pvgis_stgnn_seed_only.yaml` | split e impostazioni calibrated |
| Aggregatore Deep Ensemble | `scripts/analyze_pvgis_deep_ensemble.py` | definizione aggregazione e metriche |

### Artefatti Deep Ensemble sul sistema di esecuzione

Il metadata del run W&B `29yahjvt` registra:

- predictions input:
  `/home/apedalino/physiq_pv/outputs/pvgis_deep_ensemble/predictions/vfry1hgx`
- analysis output:
  `/home/apedalino/physiq_pv/outputs/pvgis_deep_ensemble/analysis/vfry1hgx`

File prodotti dall'aggregatore in tale directory:

- `deep_ensemble_metrics.json`
- `deep_ensemble_report.md`
- `deep_ensemble_predictions_summary.csv`

Questi file non sono presenti nel checkout locale analizzato e non risultano
caricati come file del run W&B. Le metriche qui riportate sono state verificate
direttamente dal `wandb-summary.json` del run di aggregazione `29yahjvt`; path e
nomi dei file sono verificati dal metadata del run e dal codice
`scripts/analyze_pvgis_deep_ensemble.py:288-313`.

### Anomaly labels

- Nel paper-style MC Dropout le label 2019 vengono attaccate **dopo** la
  predizione e servono solo a stratificare `normal` e `rare_extreme`.
- Nel Deep Ensemble `anomaly_group` serve solo alle metriche stratificate.
- Le label non sono mai feature, target o segnale di training dello ST-GNN.
- Eccezione da dichiarare: nella variante secondaria `ozii0s5s`, le label 2018
  vengono usate dalla calibrazione post-hoc `group` per scegliere fattori
  diversi fra normal e rare/extreme. Questo non modifica il modello, ma modifica
  gli intervalli calibrati.

Fonti:

- `physiq_pv/experiments/pvgis_stgnn_runner.py:709-730`
- `physiq_pv/experiments/pvgis_stgnn_runner.py:172-174`
- `scripts/analyze_pvgis_deep_ensemble.py:35-39`

### PVGIS-only, nessun dato reale

La directory di input contiene il segmento di path `/data/SentinelPV/`, ma il
dataset utilizzato e' `pvgis_summed_irradiance`: il nome della directory padre
non implica uso di feature Sentinel.

Vincoli implementati:

- no `ENERGIA`;
- no produzione reale osservata;
- no Sentinel/SCADA;
- no kWp/UPN/`load_kwp`;
- no `compute_qs` o quality score reale;
- `pv_lag_pvgis` e' il lag del target PVGIS, non della produzione reale.

Fonti:

- `physiq_pv/data/pvgis_stgnn_dataset.py:2-21`
- `physiq_pv/experiments/pvgis_stgnn_runner.py:23-25`
- `scripts/analyze_pvgis_deep_ensemble.py:8-15`

## 6. Claim finale difendibile

> Sul test PVGIS 2019, lo ST-GNN mostra un errore rare/extreme circa 1.6 volte
> quello normal. Il Deep Ensemble di cinque modelli migliora MAE e aumenta la
> sensibilita' dell'incertezza alle condizioni rare, ma la copertura nativa
> resta insufficiente: PICP 0.173 contro target 0.95. La copertura circa 0.95 si
> ottiene solo nella variante post-hoc calibrata su 2018, al costo di intervalli
> molto piu' larghi; non e' quindi evidenza di incertezza nativamente calibrata.
