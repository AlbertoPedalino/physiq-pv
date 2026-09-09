# STGAN CNN spaziale + LSTM

Implementazione dedicata nel branch `experiment/stgan-cnn`, derivato da
`feat/anomaly-spatial-threshold-comparison`. La pipeline di partenza proviene
da `feat/stgan-paper`, snapshot `777df6bc6deddeccafbf806bd1c380f79ea146a1`.
Il modello GCN-GRU originale rimane nel suo branch. Qui non esiste un selettore
di architettura e nessuna matrice di adiacenza entra nel training o nello scoring.

## Modello e protocollo

- Input recente: il timestamp precedente, disposto su una finestra geografica
  3x3. I canali sono POA, temperatura, vento e maschera (1 presente, 0 assente).
- CNN spaziale: due Conv2d 3x3 con 32 canali, ReLU, stride 1, padding zero 1.
  La maschera e' un input della CNN standard; non si usano partial convolutions.
  Il padding degli strati e le celle geografiche assenti sono concetti distinti.
- Ramo lungo: LSTM originale, due layer, hidden 64, 168 ore della localita'
  centrale. Calendario originale: 7+24 indicatori.
- Generatore: proiezione 1x1 senza grafo dopo la concatenazione dei tre rami;
  restituisce le tre variabili per tutte le celle della finestra.
- Discriminatore: CNN sul dato storico e proiezione 1x1 sul dato corrente;
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
- `recent_steps` deve essere 1: questa variante non elimina tacitamente una
  sequenza temporale recente piu' lunga. La storia resta nel ramo LSTM.

Parametri default con tre variabili:

| | GCN-GRU originale | CNN con maschera | Differenza |
|---|---:|---:|---:|
| Generatore | 63.171 | 63.907 | +1,17% |
| Discriminatore | 36.769 | 37.569 | +2,18% |

I conteggi sono ricalcolati e salvati per ogni configurazione. Sono inclusi
il quarto canale e la proiezione corrente del discriminatore con maschera.
Parita' dei parametri non implica parita' di costo: metadata e history registrano
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

Installare le dipendenze aggiornate (`uv sync`, oppure installare il progetto
nell'ambiente esistente). `pyproj` e' necessario per l'audit delle coordinate.
Si puo' riusare lo stesso manifest STGAN gia' preparato, senza rigenerare i dati:

```bash
python scripts/run_pvgis_stgan.py \
  --manifest outputs/pvgis_stgan/prepared/manifest.csv \
  --out-dir outputs/pvgis_stgan_cnn/grid_audit --audit-only

python scripts/run_pvgis_stgan.py \
  --manifest outputs/pvgis_stgan/prepared/manifest.csv \
  --out-dir outputs/pvgis_stgan_cnn/reference \
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

Prima di avviare Jupyter impostare (stesso ambiente ereditato dal kernel):

```bash
export STGAN_SEED_DIR="$PWD/outputs/pvgis_stgan_cnn/reference/seed_20"
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
Quando disponibile esegue anche una prova CUDA con le dimensioni del modello
di riferimento e la LSTM sulle 168 ore.
