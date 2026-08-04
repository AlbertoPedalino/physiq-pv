# Come funziona MTGFlow

Detector non supervisionato di anomalie su serie multivariate, da Zhou et al.,
*Label-Free Multivariate Time Series Anomaly Detection* (arXiv:2312.11549v2).
Qui è implementata la variante **base** (senza estensione `MTGFlow_cluster`).

Ogni sezione indica il file e la riga in cui vive il codice descritto.

---

## Indice — due letture possibili

Il documento è diviso in due tracce. Ogni titolo è marcato con la sua traccia.

### ▶ FLUSSO — dove vive il codice, chi chiama chi

Leggi **solo queste** se ti serve orientarti nel repository:

| § | Sezione | Risponde a |
|---|---|---|
| [0](#0-flusso--punto-di-ingresso-e-catena-completa) | Punto di ingresso e catena completa | Da dove parte tutto? |
| [2](#2-flusso--preparazione-dei-dati) | Preparazione dei dati | Da dove escono i CSV di input? |
| [10](#10-flusso--orchestrazione-ed-export) | Orchestrazione ed export | Cosa fa il runner, quali file scrive? |
| [11](#11-flusso--chi-consuma-cosa-a-valle) | Chi consuma cosa | Dove finiscono gli score? |
| [12](#12-flusso--ruolo-di-ogni-file) | **Ruolo di ogni file** | A cosa serve ciascun modulo? |
| [13](#13-flusso--schema-riassuntivo) | Schema riassuntivo | Tutto il flusso in un diagramma |

**Percorso minimo:** §0 → §12 → §13.

### ○ TEORIA — come funziona l'algoritmo

Sezioni [1](#1-teoria--idea-di-fondo), [3](#3-teoria--input-finestre),
[4](#4-teoria--grafo-dinamico-via-self-attention-eq-6--7),
[5](#5-teoria--condizione-spazio-temporale-44-eq-8),
[6](#6-teoria--normalizing-flow-entity-aware-eq-2--4-9-11),
[7](#7-teoria--ottimizzazione-congiunta-46),
[8](#8-teoria--score-di-anomalia-eq-12-14),
[9](#9-teoria--soglie-iqr-eq-13-15),
[14](#14-teoria--fedeltà-e-scelte-non-specificate-dal-paper):
equazioni del paper, architettura, scelte di fedeltà. Anche queste puntano al
codice, ma spiegano il *perché* invece del *dove*.

---

## 0. FLUSSO — Punto di ingresso e catena completa

L'entrypoint operativo è `notebooks/mtgflow_pvgis_workflow.ipynb`, ma il notebook
**non importa nulla**: lancia i due script via `subprocess.run` (celle a
`nb:127-138` e `nb:304-316 / 348-357 / 1539-1551`). Il punto di ingresso reale
del codice resta quindi il CLI, ed è riproducibile da terminale senza notebook.

```text
1) PREPARAZIONE   scripts/prepare_pvgis_mtgflow.py            main() :42
   NetCDF PVGIS → train.csv (2005-2018) / test.csv (2019) per località
                + manifest_shard_XXXX.csv

2) DETECTION      scripts/run_pvgis_mtgflow.py                main() :258
   manifest → training + scoring + soglie → anomaly_scores.csv (S_c)
                                          → entity_anomaly_scores.csv (S_ck)

3) VALUTAZIONE    scripts/evaluate_predictions_on_climate_anomalies.py
   predizioni SDE-Net + anomaly_scores.csv → metriche stratificate anomaly/normal
```

Lo step 3 è l'unico punto di contatto con la pipeline di forecasting, ed è
**post-hoc**: MTGFlow non entra mai nel training della SDE-Net.

**Dove entra l'anno di test.** `--test-year 2019` agisce solo nello step 1:
decide quale anno finisce in `test.csv`. Il 2019 non tocca né la normalizzazione
(`pipeline.py:113`, statistiche su train), né l'ottimizzazione
(`pipeline.py:169`, solo `train_fit_ds`), né le soglie
(`run_pvgis_mtgflow.py:316`, IQR su score di train). Viene solo *scorato*
(`pipeline.py:237`).

**Quale run etichetta davvero il 2019.** Il notebook contiene tre invocazioni
dello step 2, non intercambiabili:

| cella | flag | scopo | copertura del 2019 |
|---|---|---|---|
| `nb:304` | `--epochs 1 --max-locations 1` | smoke test | irrilevante |
| `nb:348` | default | run di riferimento, 5 seed | ~1 finestra ogni 10 ore |
| `nb:1539` | `--score-stride 1 --seed 15` | **etichettatura** | ogni ora |

Il default `score_stride = 10` (`config.py:24`) fa parte del protocollo del
paper e produce ~876 timestamp scorati sull'anno. Per etichettare ogni ora serve
`--score-stride 1`, che il runner registra come
`scoring_profile: "dense_hourly_window_adaptation"` (`run_pvgis_mtgflow.py:470`)
— cioè è tracciato come adattamento consapevole, non come deviazione silenziosa.

**Semantica dell'etichetta.** Lo score copre 60 ore ma è registrato
sull'**ultimo** istante della finestra (`pipeline.py:139`; metadata
`window_score_semantics: "whole_window_assigned_to_window_end"`). Un timestamp
marcato anomalo significa "la finestra che finisce qui è improbabile", non
"questa ora è anomala".

## 1. TEORIA — Idea di fondo

Non si impara "cosa è normale" da un training set pulito (approccio one-class):
si **stima la densità di probabilità di tutti i dati di training**, contaminati
inclusi. L'ipotesi è che le anomalie stiano in regioni a bassa densità, e siano
comunque una minoranza, quindi la densità stimata resta dominata dal
comportamento normale.

Conseguenze pratiche:

- il training set può contenere anomalie: non serve etichettarle né rimuoverle;
- lo score non è un errore di ricostruzione ma una **log-verosimiglianza
  negativa**: più bassa la densità, più alto lo score;
- serve un modello di densità flessibile e con verosimiglianza *esatta*: da qui
  il normalizing flow.

Il problema è che la densità di una finestra multivariata dipende sia dal tempo
sia dalle relazioni tra entità. MTGFlow risolve stimando una densità
**condizionata**: il flow non modella `P(x)` in astratto, ma `P(x | C)` dove `C`
è una condizione spazio-temporale prodotta da un grafo dinamico + RNN.

## 2. FLUSSO — Preparazione dei dati

> `physiq_pv/anomaly_detection/pvgis_climate.py` — CLI: `scripts/prepare_pvgis_mtgflow.py`

`prepare_pvgis_climate_data` (`pvgis_climate.py:135`) legge i NetCDF PVGIS anno
per anno, verifica che l'ordine delle località non cambi
(`_assert_same_locations`, `:125`), e per ogni località scrive
`train.csv` / `validation.csv` / `test.csv` + `metadata.json`
(`:213-232`), più un manifest per shard (`:250`).

Le `K = 3` entità sono fissate in `CLIMATE_FEATURES` (`pvgis_climate.py:22`):

```python
("solar_irradiance_poa", "temperature_2m", "wind_speed_10m")
```

Il target di regressione `pv_power_output` è **escluso di proposito**: il
detector modella condizioni meteo-solari, non produzione.
`raw_to_climate_frame` (`:38`) costruisce il frame per singola località; oltre
alle feature scrive `is_daytime`, usato solo come metadato diagnostico.

`SeasonalRobustScaler` (`:57`, normalizzazione median/MAD per bucket mese-ora) è
**opt-in** (`--seasonal-normalization`) e *non* fa parte del protocollo di
riferimento; il runner rifiuta i dati preparati così
(`run_pvgis_mtgflow.py:144` `_verify_preparation_metadata`).

## 3. TEORIA — Input: finestre

> `physiq_pv/anomaly_detection/mtgflow/pipeline.py`

Normalizzazione per entità (Eq. 5), statistiche calcolate **solo sul training**
(`pipeline.py:113-119`); `std` degenere viene forzato a 1:

```python
x_k = (x_k - mean(x_k)) / std(x_k)
```

Quali colonne diventano entità lo decide `detector_features`
(`common.py:15`): scarta `timestamp` e `is_daytime`, e **solleva eccezione** se
compaiono `pv_power_output`, `label`, `is_anomaly`, `anomaly_label`, `attack`.
È la guardia anti-leakage: supervisione e target di forecast non possono entrare.

Finestra scorrevole di dimensione `M = 60` ore con stride `S`
(`Windows`, `pipeline.py:121`):

```text
train_stride = 10   finestre usate per ottimizzare
score_stride = 10   finestre usate per calibrare le soglie e per lo score
```

Una finestra è scartata se al suo interno c'è un salto temporale irregolare
(`_regular_window_starts`, `pipeline.py:43`: cadenza dedotta come differenza
modale, prefix-sum sugli scarti). Il tensore di batch è:

```text
x : (B, K, M, 1)     batch, entità, tempo, canale
```

Il canale è 1 perché ogni entità è una serie scalare: tensorizzazione pointwise
`input_size=1` presa dal repository ufficiale (il paper non la specifica).

Il **livello di decisione è la finestra**, non il singolo punto: una finestra è
anomala se contiene almeno un punto anomalo. Lo score della finestra viene poi
registrato sul suo istante finale (`endpoints`, `pipeline.py:138`); l'istante
iniziale resta disponibile come diagnostica (`startpoints`, `:142`).

## 4. TEORIA — Grafo dinamico via self-attention (Eq. 6--7)

> `DynamicGraphAttention` — `mtgflow/model.py:21`

Ogni entità è un nodo. La finestra di ciascun nodo (`M` valori appiattiti) viene
proiettata in query e key e confrontata a coppie:

```python
e_ij = (x_i W_Q)(x_j W_K)^T / sqrt(M * input_size)
a_ij = exp(e_ij) / sum_j exp(e_ij)        # softmax per riga
A    = [a_ij]                              # (B, K, K)
```

`A` è la matrice di adiacenza del grafo, **ricalcolata per ogni finestra**:
quindi le dipendenze tra entità evolvono nel tempo, invece di essere un DAG
statico come in GANF. Le proiezioni sono lineari senza bias (`model.py:27-28`),
la scala è `sqrt(window_size * input_size)` (`:30`), e dopo il softmax c'è un
dropout 0.2 (`:38`) — attivo solo in `train()`.

Nota: non c'è softmax sulle colonne né simmetrizzazione — `a_ij` quantifica il
flusso da `j` verso `i` e la riga `i` somma a 1.

## 5. TEORIA — Condizione spazio-temporale (§4.4, Eq. 8)

> LSTM in `MTGFlow.__init__` (`model.py:158`); grafo in `SpatioTemporalConditioner` (`model.py:41`)

**Parte temporale.** Un LSTM a 1 layer processa ogni entità indipendentemente
(`model.py:177-179`):

```text
(B, K, M, 1) → reshape (B*K, M, 1) → LSTM → (B*K, M, 32) → (B, K, M, 32)
```

Gli stati nascosti `H^t_k` sono il "time encoding": riassumono il passato della
finestra fino a `t` per quell'entità. I pesi dell'LSTM sono condivisi tra entità.

**Parte spaziale.** Una convoluzione su grafo mescola gli stati dei nodi secondo
`A`, sommando anche lo stato precedente dello stesso nodo (termine di storia,
ereditato da GANF) — `model.py:50-58`:

```python
neighbours = einsum("bij,bjth->bith", A, H)   # aggregazione dai vicini
C = ReLU(neighbours @ W1 + H_shift @ W2) @ W3
# H_shift[:, :, t] = H[:, :, t-1], azzerato a t = 0
```

`C` ha forma `(B, K, M, 32)`: per ogni entità e ogni istante, un vettore che
codifica insieme *dove* si trova l'entità nel grafo corrente e *cosa* è successo
prima. È l'unico ingresso informativo del flow.

## 6. TEORIA — Normalizing flow entity-aware (Eq. 2--4, 9, 11)

> `ConditionalMAFBlock` (`model.py:61`), `EntityAwareMAF` (`model.py:93`)

Un normalizing flow è una trasformazione invertibile `z = f(x)` che mappa il dato
su una distribuzione target nota. Il cambio di variabile dà la densità esatta:

```text
log P_X(x) = log P_Z(f(x)) + log |det df/dx|
```

Qui la trasformazione è **condizionata** da `C` e composta da `n_blocks = 2`
blocchi affini. Con variabile scalare, shift e scala non possono dipendere dalla
variabile stessa senza rompere la bigezione: sono prodotti da una MLP che riceve
**solo** la condizione (`model.py:79-90`).

```python
shift, log_scale = MLP(C)                 # Linear → Tanh → Linear → Tanh → Linear
z      = (x - shift) * exp(-log_scale)
logdet = -log_scale                        # jacobiano esatto, nessuna stima
```

La MLP ha `n_hidden = 1` layer interno: da qui i tre `Linear` alternati a `Tanh`.

**Entity-aware** significa che ogni entità ha una distribuzione target diversa
(`model.py:121`):

```text
Z_k = N(mu_k, I),   mu_k ~ N(0, 1)
```

`mu_k` è estratto una volta sola ed è registrato come *buffer*, non come
parametro: non viene addestrato (coerente con Eq. 9), ma finisce nello
`state_dict` e quindi nel checkpoint. I pesi del flow sono invece **condivisi tra
tutte le entità** (Eq. 11) — è questo che evita la crescita di memoria con `K`
pur mantenendo densità distinte per entità.

Perché serve: entità con meccanismi fisici diversi (irraggiamento, temperatura,
vento) hanno anomalie con caratteristiche di sparsità diverse. Mapparle tutte su
`N(0, I)` come fa GANF comprime queste differenze.

Log-densità di un punto (`point_log_prob`, `model.py:123-136`) e aggregazione
sulla finestra (`likelihood_components`, `model.py:196`):

```python
log P(x_k,t) = -0.5*(z - mu_k)^2 - 0.5*log(2*pi) + logdet
entity_log_prob[b, k] = sum_t log P(x_k,t)      # (B, K)
```

## 7. TEORIA — Ottimizzazione congiunta (§4.6)

> `pipeline.py:164-176`

Attenzione, LSTM, conditioner e flow sono addestrati **insieme** con un solo
optimizer (`pipeline.py:165`, su `model.parameters()`), massimizzando la
verosimiglianza media (`MTGFlow.forward`, `model.py:199-202`):

```text
loss = - mean_{b,k} entity_log_prob[b, k]

Adam(lr=2e-3, weight_decay=5e-4)
clip_grad_value_(params, 1.0)                    pipeline.py:175
epochs=40, batch_size=256, shuffle=True, window=60, n_blocks=2, hidden=32
```

I valori vivono in `MTGFlowReferenceConfig` (`config.py:17-33`); i seed di
riferimento in `REFERENCE_SEEDS = (15, 16, 17, 18, 19)` (`config.py:40`).

Nessuna etichetta, nessun early stopping, nessun model selection: si usano
epoche fisse (`checkpoint_selection = "final_fixed_epoch"`, `pipeline.py:205`).
Ottimizzare tutti i moduli insieme evita che grafo e flow convergano a ottimi
locali separati.

Il checkpoint (`pipeline.py:182-210`) salva pesi, `model_config`, media/std di
normalizzazione, nomi delle feature e ambiente di esecuzione; si ricarica con
`load_mtgflow_checkpoint` (`pipeline.py:22`), che rifiuta bundle con
`format_version != 2`.

## 8. TEORIA — Score di anomalia (Eq. 12, 14)

> `score()` interna a `fit_and_score_mtgflow` — `pipeline.py:212-234`

Lo score di finestra è la NLL media sulle entità; la sua decomposizione dà il
contributo di ciascuna entità (`pipeline.py:226-228`):

```python
entity_score = -entity_log_prob / K        # S_ck   (B, K)
global_score = entity_score.sum(dim=1)     # S_c = sum_k S_ck   (B,)
```

`sum_k (-llk_k / K)` è esattamente la media delle NLL dell'Eq. 12: `global_score`
è lo score di finestra e `entity_score` la sua attribuzione per entità
(anomaly interpretation, Eq. 14): dice *quale* variabile ha reso la finestra
improbabile.

Score alto = densità bassa = finestra più probabilmente anomala. Lo scoring gira
in `eval()` (dropout dell'attenzione spento) e sotto `torch.no_grad()`
(`pipeline.py:216-219`).

I metodi `MTGFlow.test()` (`model.py:204`) e `MTGFlow.locate()` (`model.py:209`)
espongono le stesse quantità come API pulita, ma la pipeline chiama direttamente
`likelihood_components`: `test`/`locate` sono esercitati solo dai test
(`tests/test_pvgis_climate_anomaly.py:147-148`).

Il risultato è impacchettato in `MTGFlowResult` (`result.py:11`), che porta
score globali, score per entità, nomi delle entità, inizio e fine finestra, e un
dizionario `metadata` con l'intero protocollo (`pipeline.py:253-324`).

## 9. TEORIA — Soglie IQR (Eq. 13, 15)

> `physiq_pv/anomaly_detection/thresholds.py`

Non esiste un validation set pulito su cui prendere il massimo: anche training e
validation contengono anomalie. Si usa quindi una soglia robusta calcolata sugli
score di **training**:

```text
globale     T   = Q3 + 1.5 * (Q3 - Q1)                    fit_threshold        :35
per entità  T_k = lambda * (Q3_k + 1.5 * (Q3_k - Q1_k))   fit_entity_iqr_...   :65
            lambda = 0.8   (entity_threshold_scale, config.py:33)
flag        score >= soglia                               apply_threshold      :60
                                                          apply_entity_...     :89
```

La soglia per entità è separata perché ogni `S_ck` ha scala propria (target
gaussiani diversi): una soglia unica assegnerebbe implicitamente pesi diversi
alle entità. `lambda` corregge il fatto che anche le osservazioni normali
fluttuano con ampiezze diverse da entità a entità.

`thresholds.py` accetta **solo score**: le etichette non compaiono nell'API,
quindi non possono contaminare la calibrazione. Le soglie stimate sul training
vengono poi applicate invariate al test
(`run_pvgis_mtgflow.py:316-324`, poi `:328-370`).

## 10. FLUSSO — Orchestrazione ed export

> `scripts/run_pvgis_mtgflow.py`

`main` (`:258`) legge il manifest, e per ogni seed × località:

| passo | riga | cosa fa |
|---|---|---|
| verifica preparazione | `:289` | rifiuta dati con normalizzazione stagionale |
| fit + score | `:297` | `fit_and_score_mtgflow`, checkpoint per sito |
| soglia globale | `:316` | `fit_threshold` su `train_scores` |
| soglie per entità | `:320` | `fit_entity_iqr_thresholds` su `train_entity_scores` |
| export globale | `:328`, `:342` | `_global_output` → contratto canonico |
| export per entità | `:353`, `:362` | `_entity_output` → una riga per (finestra, entità) |

Il **contratto canonico** (`CANONICAL_SCORE_COLUMNS`, `:39-46`) è ciò che i
consumatori a valle si aspettano:

```text
location, timestamp, method, anomaly_score, threshold, is_anomaly
```

Artefatti scritti:

```text
<out_dir>/
├── seed_<S>/
│   ├── anomaly_scores.csv          S_c, anno di test          :413
│   ├── train_anomaly_scores.csv    S_c, anni 2016-2018        :419
│   ├── entity_anomaly_scores.csv   S_ck, tutte le entità      :422
│   ├── summary.csv                                            :425
│   └── <site_key>/
│       ├── checkpoint.pt                                      pipeline.py:182
│       ├── test_scores.csv / train_scores.csv                 :372-373
│       ├── test_entity_scores.csv / train_entity_scores.csv   :374-375
│       └── metadata.json  (soglie, backend, verifiche)        :391
├── summary_by_seed.csv                                        :434
├── summary_aggregate.csv                                      :446
├── run_metadata.json  (deviazioni dal protocollo)             :489
└── environment.json                                           :492
```

Guardie di riproducibilità: la directory di output deve essere vuota (`:271`),
il manifest deve avere le colonne richieste (`:262`), e
`reference_protocol_deviations` (`config.py:43`) registra ogni scostamento dai
valori di riferimento in `run_metadata.json`.

## 11. FLUSSO — Chi consuma cosa (a valle)

Due file di score, due destini diversi.

**`anomaly_scores.csv` (S_c) → valutazione del forecast.**

```text
scripts/evaluate_predictions_on_climate_anomalies.py:34
  → physiq_pv/anomaly_detection/evaluation.py:9   attach_detector_scores
      richiede {location, timestamp, anomaly_score, is_anomaly}   :18
      filtra per --method, rifiuta duplicati (location, timestamp) :32
  → physiq_pv/anomaly_detection/evaluation.py:41  forecast_metrics_by_detection
      MAE / RMSE / bias / mean_pi_width / PICP per stratum anomaly vs normal
```

**`entity_anomaly_scores.csv` (S_ck) → audit del detector.**

Unico consumatore: `notebooks/mtgflow_run_audit.ipynb:522`
(`excess_over_threshold`, riepilogo per canale al timestamp di picco,
`entity_anomaly_rate` nel tempo). Non è collegabile a `evaluation.py`: contiene
`K` righe per timestamp e violerebbe il vincolo di unicità a `evaluation.py:32`.

**Cosa NON consuma MTGFlow.** Il flag `--anomaly-scores` di
`pvgis_stgnn_runner.py` passa da `physiq_pv/data/pvgis_labels.py:25`, che
pretende una colonna `label` semantica
(`unusually_low_solar_potential`, ...). Quella colonna la produce solo la
climatologia (`physiq_pv/data/pvgis_anomaly_scores.py:376`). Le due sorgenti di
anomalia restano quindi **separate per costruzione**:

```text
climatologia → label semantica → stratificazione DENTRO il run SDE-Net
MTGFlow      → is_anomaly       → confronto POST-HOC sulle predizioni
```

### 11.1 FLUSSO — I due etichettatori a confronto

Nel progetto convivono due definizioni indipendenti di "anomalia", con
entrypoint distinti:

```text
DISTRIBUZIONE   scripts/run_pvgis_climatology_anomaly.py:186        un anno
                scripts/run_pvgis_climatology_anomaly_years.py      più anni + aggregazione
                  └─ physiq_pv/data/pvgis_anomaly_scores.py:204,288,538

DENSITÀ         scripts/prepare_pvgis_mtgflow.py:42
                scripts/run_pvgis_mtgflow.py:258
```

|  | climatologia | MTGFlow |
|---|---|---|
| criterio | valore fuori dalla banda quantilica **marginale** | finestra a **bassa densità congiunta** |
| unità | punto singolo (ora) | finestra di 60 ore |
| multivariato | no, una variabile per volta | sì, grafo dinamico tra entità |
| riferimento | quantili storici per bin (località, giorno±D, ora) | densità appresa dai dati |
| output | `label` semantica | `is_anomaly` + `anomaly_score` |
| training | nessuno | 40 epoch |

La climatologia cattura il *valore estremo*; MTGFlow la *combinazione
improbabile*. Variabili tutte dentro banda ma reciprocamente incoerenti vengono
segnalate da MTGFlow e ignorate dalla climatologia. Non sono intercambiabili
anche a livello di contratto: i consumatori a valle pretendono colonne diverse
(`pvgis_labels.py:25` vuole `label`, `evaluation.py:18` vuole `is_anomaly`).

## 12. FLUSSO — Ruolo di ogni file

Ordine di esecuzione, non alfabetico.

```text
pvgis_climate  → prepara input              (prima)
common         → valida e protegge          (durante: guardia)
mtgflow/       → modella e assegna score    (durante: cuore)
thresholds     → score → flag booleano      (dopo)
evaluation     → flag → metriche forecast   (molto dopo: altro track)
```

**`physiq_pv/anomaly_detection/pvgis_climate.py` — a monte.**
Trasforma i NetCDF PVGIS in un CSV per località con le 3 entità
(`CLIMATE_FEATURES:22`), splitta per anno e scrive il manifest. Non sa nulla di
MTGFlow: produce un formato tabellare generico. Contiene anche
`SeasonalRobustScaler:57`, opt-in e fuori protocollo.

**`physiq_pv/anomaly_detection/common.py` — dentro, come guardia.**
Chiamato da `pipeline.py:100-110`. È il modulo che impedisce il leakage:

- `detector_features:15` — sceglie le colonne-entità e **solleva eccezione** se
  compaiono `pv_power_output`, `label`, `is_anomaly`, `anomaly_label`, `attack`;
- `validate_numeric_features:38` — rifiuta NaN e infiniti;
- `chronological_frame:138` — rifiuta timestamp duplicati o non monotoni;
- `seed_everything:53` — RNG e kernel Torch deterministici;
- `runtime_environment:73` — versioni, GPU, commit git per i metadata.

**`physiq_pv/anomaly_detection/mtgflow/` — il cuore.**

| file | ruolo |
|---|---|
| `model.py` | architettura: attenzione dinamica, conditioner spazio-temporale, MAF entity-aware |
| `pipeline.py` | finestre, normalizzazione, training loop, checkpoint, scoring |
| `config.py` | iperparametri di riferimento, seed, rilevazione deviazioni dal protocollo |
| `result.py` | `MTGFlowResult`: score globali, per entità, confini finestra, metadata |
| `__init__.py` | superficie pubblica (`fit_and_score_mtgflow`, `load_mtgflow_checkpoint`, ...) |

**`physiq_pv/anomaly_detection/thresholds.py` — a valle dello score.**
Trasforma score continui in flag booleani. Chiamato dal runner
(`run_pvgis_mtgflow.py:316,320`), **non** dalla pipeline. Accetta solo score:
le etichette non compaiono nell'API, quindi non possono contaminare la
calibrazione. È l'unico modulo ri-esportato da `anomaly_detection/__init__.py`.

**`physiq_pv/anomaly_detection/evaluation.py` — fuori dal detector.**
Non partecipa né al training né allo scoring. Unisce le predizioni della SDE-Net
al CSV di MTGFlow (`attach_detector_scores:9`) e calcola le metriche per stratum
(`forecast_metrics_by_detection:41`). È il ponte post-hoc tra i due track.

**Entrypoint e materiale di supporto**

| File | Ruolo |
|---|---|
| `scripts/prepare_pvgis_mtgflow.py` | CLI step 1: preparazione |
| `scripts/run_pvgis_mtgflow.py` | CLI step 2: training, scoring, soglie, export |
| `scripts/evaluate_predictions_on_climate_anomalies.py` | CLI step 3: valutazione post-hoc |
| `notebooks/mtgflow_pvgis_workflow.ipynb` | orchestrazione end-to-end via `subprocess` |
| `notebooks/mtgflow_run_audit.ipynb` | audit del run, unico consumatore di `S_ck` |
| `tests/test_pvgis_climate_anomaly.py` | test del detector e del contratto di export |

## 13. FLUSSO — Schema riassuntivo

```text
NetCDF PVGIS
   │  pvgis_climate.py:135 → train.csv / test.csv + manifest
   ▼
serie grezze (K=3 entità, oraria)
   │  z-score per entità, statistiche training-only   Eq. 5   pipeline.py:113
   ▼
finestre M=60, stride S, senza salti temporali → (B,K,M,1)    pipeline.py:43,121
   │
   ├─ self-attention su finestra → A (B,K,K)         Eq. 6-7  model.py:21
   ├─ LSTM per entità            → H (B,K,M,32)      §4.4     model.py:158
   ▼
C = ReLU(A H W1 + H^{t-1} W2) W3   → (B,K,M,32)      Eq. 8    model.py:41
   │
   ▼
MAF condizionale (2 blocchi, pesi condivisi)         Eq. 2-4,11  model.py:61,93
   z = (x - shift(C)) * exp(-log_scale(C)),  logdet = -log_scale
   base: N(mu_k, I), mu_k ~ N(0,1) fisso per entità  Eq. 9    model.py:121
   │
   ▼
entity_log_prob (B,K) = somma su M delle log-densità          model.py:196
   │
   ├─ training: loss = -mean(entity_log_prob)        §4.6     pipeline.py:169
   └─ scoring : S_ck = -llk/K ; S_c = sum_k S_ck     Eq.12,14 pipeline.py:226
                soglie IQR su score di training      Eq.13,15 thresholds.py:35,65
                   │
                   ├─ S_c  → anomaly_scores.csv        → evaluation.py:41
                   └─ S_ck → entity_anomaly_scores.csv → mtgflow_run_audit.ipynb
```

## 14. TEORIA — Fedeltà e scelte non specificate dal paper

Policy: `paper_when_explicit_official_repo_when_underspecified`
(`config.py:14`; dettagli in `docs/MTGFLOW_PAPER_ALIGNMENT.md`).

Dal paper: z-score per entità, attenzione normalizzata per riga, condizione
`ReLU(A H W1 + H^{t-1} W2) W3`, target gaussiani per entità con parametri di flow
condivisi, MLE congiunta, score come NLL media, decomposizione per entità, soglie
IQR globale e per entità con `lambda`.

Dal repository ufficiale (dettagli lasciati aperti dal testo): tensorizzazione
pointwise `input_size=1`, hidden size 32, 1 layer interno per blocco, 2 blocchi,
weight decay, gradient clipping, seed `15--19`.

Non implementato: `MTGFlow_cluster` (Eq. 10), cioè il clustering `KShape` delle
entità con target condiviso per cluster. Con `K = 3` entità la variante base
coincide di fatto con il caso in cui ogni cluster ha cardinalità 1.
