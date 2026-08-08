# MTGFlow — Flusso di esecuzione (branch `feat/pvgis-climate-anomaly-detectors`)

Mappa **runtime** (chi chiama chi), non gli import. Ordine reale di esecuzione
partendo dai tre comandi. Riferimenti `file:riga`.

Teoria, equazioni e architettura: **[MTGFLOW_MODEL.md](MTGFLOW_MODEL.md)**.
Qui c'è solo il flusso.

## I tre comandi

```
1) PREPARAZIONE   scripts/prepare_pvgis_mtgflow.py     NetCDF → CSV + manifest
2) DETECTION      scripts/run_pvgis_mtgflow.py         CSV → score + flag
3) VALUTAZIONE    scripts/evaluate_predictions_on_climate_anomalies.py
                                                       score + predizioni SDE-Net → metriche
```

`notebooks/mtgflow_pvgis_workflow.ipynb` non è un entrypoint di codice: lancia
1) e 2) via `subprocess.run` (celle `nb:127`, `nb:304`, `nb:348`, `nb:1539`).

## 1. Albero delle chiamate — preparazione

```
python scripts/prepare_pvgis_mtgflow.py --pvgis-dir ... --train-start 2005 --train-end 2018 --test-year 2019
│
└─ main()                                        scripts/prepare_pvgis_mtgflow.py:42
   │
   └─ prepare_pvgis_climate_data()               anomaly_detection/pvgis_climate.py:135
      │
      ├─ load_pvgis_years()                      data/pvgis_dataset.py
      │     └─ xr.open_dataset per anno
      │
      ├─ _assert_same_locations()                pvgis_climate.py:125
      │     └─ ordine località identico fra anni, altrimenti ValueError
      │
      ├─ for batch di località (default 8):
      │    ├─ build_year_raw(ds)                 data/pvgis_dataset.py
      │    └─ raw_to_climate_frame()             pvgis_climate.py:38
      │          → timestamp, solar_irradiance_poa, temperature_2m,
      │            wind_speed_10m, is_daytime
      │
      ├─ [--seasonal-normalization] SeasonalRobustScaler.fit/transform   :57
      │     (opt-in, FUORI dal protocollo di riferimento)
      │
      └─ per località: to_csv train / validation / test + metadata.json   :213-232
         └─ manifest_shard_XXXX.csv                                       :250
```

Output: `prepared/<site_key>/{train,test}.csv` + `manifest_shard_0000.csv`.
`--test-year 2019` agisce **solo qui**: decide quale anno finisce in `test.csv`.

## 2. Albero delle chiamate — detection

```
python scripts/run_pvgis_mtgflow.py --manifest ... --out-dir ... [--score-stride 1]
│
└─ main()                                        scripts/run_pvgis_mtgflow.py:258
   │
   ├─ pd.read_csv(manifest) + colonne richieste  :261-264
   ├─ out_dir deve essere vuota                  :271   (anti-mescolamento run)
   │
   └─ for seed in (15,16,17,18,19):              config.py:40
        │
        └─ for row in manifest:                  :287
           │
           ├─ _verify_preparation_metadata()     :144   (rifiuta dati seasonal)
           │
           ├─ fit_and_score_mtgflow(train, test) mtgflow/pipeline.py:67
           │   │
           │   ├─ seed_everything(seed)          common.py:53   (RNG + kernel deterministici)
           │   ├─ chronological_frame()          common.py:138  (no duplicati, monotono)
           │   ├─ detector_features(train)       common.py:15   ← GUARDIA ANTI-LEAKAGE
           │   │     └─ ValueError se compaiono pv_power_output / label /
           │   │        is_anomaly / anomaly_label / attack
           │   ├─ validate_numeric_features()    common.py:38   (no NaN, no inf)
           │   │
           │   ├─ mean/std SOLO su train         pipeline.py:113   Eq. 5
           │   │
           │   ├─ Windows(train, train_stride)   pipeline.py:121,145   finestre M=60
           │   ├─ Windows(train, score_stride)   pipeline.py:146
           │   ├─ Windows(test,  score_stride)   pipeline.py:147
           │   │     └─ _regular_window_starts() pipeline.py:43
           │   │           └─ cadenza modale + prefix-sum → scarta finestre con salti
           │   │
           │   ├─ MTGFlow(**model_config)        mtgflow/model.py:139   (pesi random)
           │   │     ├─ DynamicGraphAttention    model.py:21
           │   │     ├─ nn.LSTM                  model.py:158
           │   │     ├─ SpatioTemporalConditioner model.py:41
           │   │     └─ EntityAwareMAF           model.py:93
           │   │
           │   ├─ Adam(lr=2e-3, weight_decay=5e-4)   pipeline.py:165
           │   │
           │   ├─ for epoca in 1..40:            pipeline.py:169   (no early stop, no val)
           │   │     └─ [hot path, vedi §3]
           │   │
           │   ├─ torch.save(checkpoint.pt)      pipeline.py:182
           │   │     └─ pesi + model_config + mean/std + features + environment
           │   │
           │   ├─ score(train_score_ds)          pipeline.py:212,236
           │   ├─ score(test_score_ds)           pipeline.py:237
           │   │     └─ [scoring path, vedi §4]
           │   │
           │   └─ MTGFlowResult(...)             mtgflow/result.py:11
           │
           ├─ fit_threshold(train_scores)        thresholds.py:35    T = Q3 + 1.5·IQR
           ├─ fit_entity_iqr_thresholds(...)     thresholds.py:65    T_k = 0.8·(Q3_k + 1.5·IQR_k)
           │     (entrambe calibrate su TRAIN, applicate invariate al test)
           │
           ├─ _global_output()                   :204 → apply_threshold()         thresholds.py:60
           ├─ _entity_output()                   :176 → apply_entity_thresholds() thresholds.py:89
           │
           └─ to_csv per sito + metadata.json    :372-393
        │
        └─ export aggregato del seed             :413-427
   │
   └─ summary_by_seed / summary_aggregate / run_metadata / environment   :434-493
        └─ reference_protocol_deviations()       config.py:43
              └─ registra ogni scostamento dai valori del paper
```

## 3. Catena critica per-batch (hot path, ripetuta ogni batch × 40 epoche)

```
DataLoader → Windows.__getitem__                 pipeline.py:132
   └─ slice data[start : start+60].T → (K, 60, 1)
   │
   → MTGFlow.forward(x)                          model.py:199
      └─ likelihood_components(x)                model.py:169
         ├─ DynamicGraphAttention.forward()      model.py:32   → A (B,K,K)
         │     └─ Q·Kᵀ / sqrt(M·input) → softmax per riga → dropout 0.2
         ├─ nn.LSTM (B·K, 60, 1)                 model.py:178  → H (B,K,60,32)
         ├─ SpatioTemporalConditioner.forward()  model.py:50   → C (B,K,60,32)
         │     └─ einsum(A,H) + shift temporale → ReLU → proiezione
         └─ EntityAwareMAF.point_log_prob()      model.py:123
               └─ ConditionalMAFBlock.forward × 2  model.py:85
                     └─ z = (x - shift(C))·exp(-log_scale(C)),  logdet = -log_scale
   │
   → loss = -mean(entity_log_prob)               pipeline.py:173
   → loss.backward()
   → clip_grad_value_(params, 1.0)               pipeline.py:175
   → optimizer.step()
```

Un solo optimizer su `model.parameters()`: attenzione, LSTM, conditioner e flow
si addestrano **insieme**.

## 4. Catena di scoring (eval, senza gradienti)

```
score(dataset)                                   pipeline.py:212
   ├─ model.eval()                               (dropout attenzione spento)
   ├─ torch.no_grad()
   └─ per batch:
        ├─ likelihood_components() → entity_log_prob (B, K)
        ├─ S_ck = -entity_log_prob / K           pipeline.py:226   Eq. 14
        └─ S_c  = S_ck.sum(dim=1)                pipeline.py:228   Eq. 12
   → (startpoints, endpoints, S_c, S_ck)         pipeline.py:229-234
```

Lo score della finestra è registrato sul suo **istante finale**
(`endpoints`, `pipeline.py:139`).

## 5. Artefatti scritti

```
<out_dir>/
├── seed_<S>/
│   ├── anomaly_scores.csv          S_c, anno di test               :413
│   ├── train_anomaly_scores.csv    S_c, anni di training           :419
│   ├── entity_anomaly_scores.csv   S_ck, una riga per entità       :422
│   ├── summary.csv                                                 :425
│   └── <site_key>/
│       ├── checkpoint.pt                                pipeline.py:182
│       ├── test_scores.csv / train_scores.csv                      :372-373
│       ├── test_entity_scores.csv / train_entity_scores.csv        :374-375
│       └── metadata.json    soglie, backend, verifiche             :391
├── summary_by_seed.csv                                             :434
├── summary_aggregate.csv                                           :446
├── run_metadata.json        deviazioni dal protocollo              :489
└── environment.json                                                :492
```

**Contratto canonico** (`CANONICAL_SCORE_COLUMNS`, `:39-46`), ciò che i
consumatori a valle si aspettano:

```
location, timestamp, method, anomaly_score, threshold, is_anomaly
```

## 6. Albero delle chiamate — valutazione

```
python scripts/evaluate_predictions_on_climate_anomalies.py \
    --predictions <sde_net>/predictions.csv \
    --detector-scores <out>/seed_15/anomaly_scores.csv --method mtgflow
│
└─ main()                                  scripts/evaluate_predictions_on_climate_anomalies.py:30
   │
   ├─ attach_detector_scores()             anomaly_detection/evaluation.py:9
   │     ├─ richiede {location, timestamp, anomaly_score, is_anomaly}    :18
   │     ├─ filtra per --method                                          :22
   │     ├─ rifiuta duplicati (location, timestamp)                      :32
   │     └─ left-join sulle predizioni, validate="many_to_one"           :38
   │
   └─ forecast_metrics_by_detection()      anomaly_detection/evaluation.py:41
         └─ per stratum {anomaly, normal}: MAE, RMSE, bias,
            mean_predictive_std, mean_pi_width, PICP
   →  predictions_with_detector_scores.csv + forecast_metrics_by_detection.csv
```

## 7. Chi consuma cosa

```
anomaly_scores.csv         (S_c)   → evaluation.py:41              confronto con SDE-Net
entity_anomaly_scores.csv  (S_ck)  → mtgflow_run_audit.ipynb:522   audit del detector
```

`S_ck` **non entra** nella pipeline SDE-Net: ha K righe per timestamp e
violerebbe il vincolo di unicità a `evaluation.py:32`. Serve all'attribuzione —
quale canale (POA / temperatura / vento) ha reso improbabile la finestra.

**Cosa MTGFlow non alimenta.** Il flag `--anomaly-scores` di
`pvgis_stgnn_runner.py` passa da `data/pvgis_labels.py:25`, che pretende una
colonna `label` semantica prodotta solo dalla climatologia
(`data/pvgis_anomaly_scores.py:376`). Due etichettatori separati per costruzione:

```
climatologia → label semantica → stratificazione DENTRO il run SDE-Net
MTGFlow      → is_anomaly      → confronto POST-HOC sulle predizioni
```

Entrypoint dell'altro etichettatore: `scripts/run_pvgis_climatology_anomaly.py:186`
(un anno) e `scripts/run_pvgis_climatology_anomaly_years.py` (più anni).

## 8. Ruolo di ogni file

```
pvgis_climate  → prepara input              (prima)
common         → valida e protegge          (durante: guardia)
mtgflow/       → modella e assegna score    (durante: cuore)
thresholds     → score → flag booleano      (dopo)
evaluation     → flag → metriche forecast   (molto dopo: altro track)
```

| File | Ruolo |
|---|---|
| `anomaly_detection/pvgis_climate.py` | NetCDF → CSV per località; `CLIMATE_FEATURES:22` fissa le 3 entità |
| `anomaly_detection/common.py` | guardia anti-leakage (`:15`), validazione, seeding, ambiente |
| `anomaly_detection/mtgflow/model.py` | architettura: attenzione, conditioner, MAF entity-aware |
| `anomaly_detection/mtgflow/pipeline.py` | finestre, normalizzazione, training, checkpoint, scoring |
| `anomaly_detection/mtgflow/config.py` | iperparametri di riferimento, seed, deviazioni |
| `anomaly_detection/mtgflow/result.py` | `MTGFlowResult` |
| `anomaly_detection/thresholds.py` | soglie IQR globali e per entità; accetta **solo score** |
| `anomaly_detection/evaluation.py` | join predizioni ↔ detector, metriche stratificate |
| `scripts/prepare_pvgis_mtgflow.py` | CLI step 1 |
| `scripts/run_pvgis_mtgflow.py` | CLI step 2 |
| `scripts/evaluate_predictions_on_climate_anomalies.py` | CLI step 3 |
| `notebooks/mtgflow_pvgis_workflow.ipynb` | orchestrazione via `subprocess` |
| `notebooks/mtgflow_run_audit.ipynb` | audit del run, unico consumatore di `S_ck` |
| `tests/test_pvgis_climate_anomaly.py` | test del detector e del contratto di export |

## 9. Foglie di calcolo (librerie esterne)

| Funzione | Libreria | Dove |
|---|---|---|
| `xr.open_dataset` (NetCDF PVGIS) | xarray | `pvgis_climate.py` via `data/pvgis_dataset.py` |
| `nn.LSTM` | torch | `model.py:158` |
| softmax attention scritta a mano | torch | `model.py:32` |
| `einsum` (aggregazione grafo) | torch | `model.py:54` |
| `Adam`, `clip_grad_value_` | torch | `pipeline.py:165,175` |
| `np.quantile` (IQR) | numpy | `thresholds.py:47,84` |

## Note

- **Semantica del flag**: lo score copre 60 ore ma è registrato sull'ultimo
  istante (`window_score_semantics: "whole_window_assigned_to_window_end"`).
  Timestamp anomalo = "la finestra che finisce qui è improbabile", non
  "questa ora è anomala".
- **Copertura oraria**: il default `score_stride = 10` (`config.py:24`) segue il
  paper e scora ~1 finestra ogni 10 ore. Per etichettare **ogni ora** serve
  `--score-stride 1`, tracciato come
  `scoring_profile: "dense_hourly_window_adaptation"` (`run_pvgis_mtgflow.py:470`).
- **Nessun leakage dal test**: il 2019 non tocca normalizzazione
  (`pipeline.py:113`), ottimizzazione (`pipeline.py:169`, solo `train_fit_ds`)
  né soglie (`run_pvgis_mtgflow.py:316`). Viene solo scorato.
- **Nessuna model selection**: 40 epoche fisse,
  `checkpoint_selection = "final_fixed_epoch"` (`pipeline.py:205`). Il
  `validation` eventuale è validato per schema ma escluso da training e soglie.
- **API non usata dalla pipeline**: `MTGFlow.test()` (`model.py:204`) e
  `.locate()` (`model.py:209`) sono esercitati solo dai test
  (`test_pvgis_climate_anomaly.py:147-148`); la pipeline chiama direttamente
  `likelihood_components`.
- **MTGFlow non entra mai nel training della SDE-Net**: il contatto è solo lo
  step 3, post-hoc, su predizioni già prodotte.
