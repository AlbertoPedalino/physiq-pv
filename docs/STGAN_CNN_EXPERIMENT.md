# STGAN ConvGRU su griglia + LSTM del trend

Implementazione dedicata nel branch `experiment/stgan-cnn`, derivato da
`feat/anomaly-spatial-threshold-comparison`. La pipeline di partenza proviene
da `feat/stgan-paper`, snapshot `777df6bc6deddeccafbf806bd1c380f79ea146a1`.
Il modello GCN-GRU originale rimane nel suo branch. Qui non esiste un selettore
di architettura e nessuna matrice di adiacenza entra nel training o nello scoring.

## Modello e protocollo

- Input recente: sequenza `[batch, recent_steps, 3, 3, 3]` (tempo, canali,
  righe, colonne dopo il batch), con POA, temperatura e vento. Il default
  resta il timestamp precedente (`recent_steps=1`), come nell'adattamento orario
  del repository di riferimento; non si modifica la finestra per introdurre la GRU.
- Ramo recente: due strati ConvGRU con 32 canali di stato ciascuno. I gate reset
  e update usano sigmoid; il candidato usa tanh. Ogni gate usa Conv2d 3x3,
  stride 1, padding 1 al posto della convoluzione sul grafo del GCGRU originale.
  Il candidato riceve lo stato precedente moltiplicato per il reset gate;
  lo stato nuovo e' `update * previous + (1 - update) * candidate`.
  La maschera di validita' e' un canale aggiuntivo a ogni strato; non si usano
  partial convolutions. Lo stato e' azzerato nelle celle assenti dopo ogni
  strato/istante e riparte da zero per ogni finestra, senza persistenza tra batch.
- Ramo lungo: LSTM originale, due layer, hidden 64, 168 ore della localita'
  centrale. Calendario originale: 7+24 indicatori.
- Generatore: proiezione 1x1 senza grafo dopo la concatenazione dei tre rami;
  restituisce le tre variabili per tutte le celle della finestra.
- Discriminatore: ConvGRU sull'intera sequenza storica e proiezione 1x1 sul dato corrente;
  celle assenti mascherate anche prima di flatten e max pooling.
- Celle assenti: zero DOPO la normalizzazione; stessa maschera e trattamento
  per reali e generati. Non vengono mai interpretate come target osservati.
- Ricostruzione: media dell'errore quadratico su celle valide e tre variabili
  per campione, poi media dei campioni. La maschera non e' una variabile da predire.
- Loss originale: G = 500*MSE + BCE, D = media BCE reale/fake;
  etichette D reale=0, generato=1.
- Score originale: minmax globale test del residuo del generatore + minmax
  globale test di D(reale)-D(generato); top-K esatto globale, default 1%.
  E' un protocollo di ranking retrospettivo, non una soglia online calibrata sul train.
- Normalizzazione delle variabili stimata esclusivamente sul train; il test
  riceve le ultime 168 ore train come contesto, senza usarle come target test.
- `recent_steps` puo' essere maggiore di 1, con `trend_steps >= recent_steps`.
  Tutti gli istanti sono elaborati in ordine. Con il default 1 la ConvGRU esegue
  un singolo aggiornamento da stato zero; la storia lunga resta nel ramo LSTM.
  `cnn_channels` e `cnn_layers` indicano canali di stato e strati della ConvGRU;
  `hidden_size` e `n_layers` continuano a configurare la LSTM del trend.

Parametri default con tre variabili:

| | GCN-GRU originale | ConvGRU con maschera |
|---|---:|---:|
| Generatore | 63.171 | 140.931 |
| Discriminatore | 36.769 | 114.593 |

I conteggi sono ricalcolati e salvati per ogni configurazione. Sono inclusi
il quarto canale e la proiezione corrente del discriminatore con maschera.
Le convoluzioni 3x3 nei tre gate aumentano i parametri: non si assume parita'
con la baseline o con la precedente CNN senza ricorrenza. Metadata e history registrano
tempi, campioni/s e picco di memoria CUDA. Non sono benchmark dedicati di latenza.

## Griglia e bordi

Prima del caricamento dei CSV temporali, il runner verifica le coordinate.
Il default e' `--grid-crs EPSG:32632` (UTM 32N), verificato sulle coordinate
PVGIS disponibili nei report locali. L'opzione `--grid-crs auto` cerca una griglia metrica nelle proiezioni EPSG:32632,
EPSG:3035, EPSG:3857 (passo 5000 m, tolleranza 25 m), poi una griglia regolare
lat/lon. L'ultimo caso viene dichiarato in gradi, NON chiamato griglia metrica
di 5 km. E' preferibile specificare il CRS originale quando noto.
Un insieme irregolare o celle duplicate causano un errore; non viene riordinato
arbitrariamente o convertito in una nuova selezione kNN.

`grid_audit.json`, `grid_layout.npz` e `grid_locations.csv` riportano orientamento,
passo, residuo di allineamento, celle valide e conteggi interni/bordi. Per le
finestre complete viene verificata anche l'uguaglianza dell'insieme di localita'
con il kNN originale; questo audit non costruisce un grafo.
Audit locale sulle 1149 coordinate di `geographical_clusters.csv` nel bundle
del 3 settembre: **917 complete / 232 incomplete, 917/917 coincidenti con kNN**,
passo 5000 m in EPSG:32632, residuo massimo di allineamento circa 0,000761 m.
Il runner ripete comunque la verifica sul manifest effettivamente usato.

Ai bordi la CNN ha meno osservazioni del kNN originale, che trova comunque
otto vicini. Per il confronto con pari osservazioni usare le finestre complete
e verificare il conteggio di corrispondenza kNN. Riportare separatamente i bordi.

## Esecuzione sul server

Questa versione richiede nuovo training: i checkpoint della precedente CNN
senza ricorrenza (`STGAN_CNN`, formato 1) non sono compatibili e vengono rifiutati
con un messaggio esplicito. I nuovi checkpoint sono `STGAN_CONVGRU`, formato 2,
e salvano anche `recent_steps` e `trend_steps`. Il notebook usa di default
`outputs/pvgis_stgan_cnn/convgru_reference` e verifica l'architettura tramite
`alignment_policy` prima di riusare un run. Nomi CLI, variabili d'ambiente e
colonna CSV `method=stgan_cnn` restano compatibili con i lettori posthoc;
il backend nei metadata e' `stgan_convgru_lstm_pvgis`.

Installare le dipendenze aggiornate (`uv sync`, oppure installare il progetto
nell'ambiente esistente). `pyproj` e' necessario per l'audit delle coordinate.
Si puo' riusare lo stesso manifest STGAN gia' preparato, senza rigenerare i dati:

```bash
python scripts/run_pvgis_stgan.py \
  --manifest outputs/pvgis_stgan/prepared/manifest.csv \
  --out-dir outputs/pvgis_stgan_cnn/grid_audit --audit-only

python scripts/run_pvgis_stgan.py \
  --manifest outputs/pvgis_stgan/prepared/manifest.csv \
  --out-dir outputs/pvgis_stgan_cnn/convgru_reference \
  --paper-top-k-percent 1 --device cuda --seeds 20
```

Se il manifest si chiama `manifest_shard_0000.csv`, passare quel percorso.
La directory di output deve essere vuota: il runner protegge i risultati esistenti.
Per preparare dati nuovi usare `scripts/prepare_pvgis_stgan.py --pvgis-dir ...`;
scrive di default in `outputs/pvgis_stgan_cnn/prepared`, mai sopra la baseline.

Il notebook `notebooks/stgan_cnn_pvgis_workflow.ipynb` comprende configurazione,
audit, conteggio parametri, training, grafici di loss e riepilogo copiabile.
Se un run completo con identica configurazione esiste, lo legge senza riaddestrare.
Per una prova ridotta usare un output distinto e `--train-samples-per-epoch 128`:
questo riduce il training ma lo scoring continua a coprire l'intero test.

Gli output mantengono le colonne dei CSV originali, con `method=stgan_cnn`.
Il confronto opzionale nel notebook legge una localita' alla volta e confronta
decisioni sugli stessi timestamp. Misura accordo, non accuratezza del detector;
usa le decisioni salvate prima del filtro di qualita' applicato nei notebook posthoc.

## Riutilizzo delle tre analisi

`stgan_pointwise_posthoc_sdenet.ipynb` legge di default il run
`outputs/pvgis_stgan_cnn/convgru_reference/seed_20`. Basta eseguire tutte le celle:
ogni esecuzione della configurazione sceglie una nuova sottocartella sotto
`outputs/sde_stgan_cnn_quality_filtered`, con nome del run, seed, data UTC e ID.
Anche un `STGAN_POSTHOC_ROOT` personalizzato viene usato come contenitore:
le analisi precedenti non sono riutilizzate o sovrascritte. Il join rifiuta
directory gia popolate; per ripartire rieseguire dalla configurazione.
Le singole celle dei grafici aggiornano gli output dell'esecuzione corrente.
Il nome del detector nei metadata viene letto dai CSV (`method=stgan_cnn`).
Se il workflow usa `STGAN_CNN_OUT_DIR`, passare lo stesso valore al posthoc;
`STGAN_SEED_DIR` ha precedenza e permette di scegliere un altro seed/run.

Prima di avviare Jupyter impostare (stesso ambiente ereditato dal kernel):

```bash
export STGAN_SEED_DIR="$PWD/outputs/pvgis_stgan_cnn/convgru_reference/seed_20"
export STGAN_POSTHOC_ROOT="$PWD/outputs/sde_stgan_cnn_quality_filtered"
export STGAN_INPUT_TARGET_OUT_DIR="$PWD/outputs/stgan_cnn_input_target_cases_t1_t6"
export ANOMALY_SENSITIVITY_OUT_DIR="$PWD/outputs/anomaly_threshold_sensitivity_cnn_t1_t6"
```

Rieseguire `stgan_pointwise_posthoc_sdenet`, `stgan_input_target_cases_sdenet`
e `anomaly_threshold_sensitivity_mtgflow_stgan`. Il forecaster rimane fisso.
Usare lo stesso dataset/filtro di qualita' della baseline. Questi tre notebook
aggregano tutte le localita'; il confronto interno/bordo delle decisioni e'
nel nuovo notebook. I riepiloghi MAE/RMSE non provano da soli la qualita' del detector.

## Ricerca successiva

Il runner espone `--cnn-layers`, `--cnn-channels`, `--hidden-size`, `--n-layers`
e `--patch-size 1|3|5`. Il notebook genera una tabella di configurazioni e
conteggi, senza avviare una ricerca sul test. Definire prima criterio di
validazione, budget e seed comuni; il runner attuale esegue un singolo protocollo
train/test, non sceglie automaticamente un vincitore. K=1 elimina il contesto
spaziale, K=3 usa al massimo 9 celle, K=5 al massimo 25. La dimensione della
proiezione nel discriminatore varia con K; i confronti devono riportarlo.
Confrontare diverse K su un insieme comune di target e non soltanto sulle
diverse popolazioni di finestre complete.

## Verifica locale

`python tests/test_stgan_cnn.py` verifica orientamento, bordi, celle mancanti,
invarianza rispetto al riempimento, gradienti, training/scoring sintetici,
checkpoint, ranking, preparazione NetCDF e compatibilita' con il lettore posthoc.
Verifica inoltre le equazioni dei gate, la dipendenza dai primi istanti della
sequenza, l'ordine temporale, l'assenza di stato persistente tra finestre,
le sequenze recenti multiple e il rifiuto dei vecchi checkpoint/run CNN.
Quando disponibile esegue anche una prova CUDA con le dimensioni del modello
di riferimento e la LSTM sulle 168 ore.
