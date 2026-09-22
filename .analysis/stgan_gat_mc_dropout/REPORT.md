# STGAN ERA5 GAT + MC Dropout

## Separazione Git, prima di ripristinare il lavoro GAT

- Branch sorgente: `experiment/stgan-era5-mc-dropout`.
- Commit sorgente CNN: `1264aa5026fc38019c6a5278f1c7360df6bb4cde`.
- Messaggio: `feat: finalize STGAN ERA5 MC dropout pipeline`.
- Nuovo branch: `experiment/stgan-era5-gat-mc-dropout`, ricreato esattamente dal commit sorgente.
- `git status --porcelain` nel worktree sorgente `.analysis/stgan_gat_mc_dropout/cnn_source`: output vuoto, pulito.
- `git rev-parse HEAD experiment/stgan-era5-mc-dropout` dopo la creazione: entrambi `1264aa5026fc38019c6a5278f1c7360df6bb4cde`.
- Nel workspace principale rimaneva soltanto `?? .worktrees/`: worktree preesistente di un altro esperimento, non modificato.
- Stato iniziale CNN preservato in `preexisting.patch`; corrispondenza esatta verificata con `git diff --binary` nel worktree CNN (normalizzando CRLF).
- Backup dello stato di lavoro precedente alla separazione: stash `d6224e8d367fd750e505b0ed0ff17b2563b87bc6`. Non eliminato.

Il primo intervento aveva creato il branch GAT mentre la versione CNN era ancora non committata.
Dopo la precisazione dell'utente, il lavoro GAT è stato messo da parte, la sola versione CNN iniziale
è stata ricostruita, verificata e committata sul branch sorgente, e il branch GAT è stato eliminato
(non aveva commit propri) e ricreato dal commit CNN. Nessun merge. Da questo punto il branch CNN
non viene più modificato.

## Verifiche della versione CNN prima del commit

Eseguiti nel worktree sorgente con il Python del progetto:

- `python -m unittest discover -s tests -p test_stgan*.py`: 52 test OK.
- `python -m unittest discover -s tests -p test_era5*.py`: 11 test OK.
- `python -m unittest discover -s tests -p test_download_era5.py`: 30 test OK (download simulati).
- `git diff --check`: OK.

Controllati: tre Dropout G, p=0.2, M=20, due forward G nel training, shared score normalization
fitted sulla sola calibration e congelata sul test senza clipping, mean/std e cubi, threshold,
morfologia, clustering e temporal linking. Split: train 1980–2002, calibration 2003–2004,
test 2005–ultimo anno locale completo. Nessun training ERA5 reale.

La normalizzazione degli **score** usa esclusivamente la calibration. Il separato scaling delle
**feature di input** rimane train-only, esattamente come nella versione CNN iniziale.

## Architettura finale

L'entrypoint `scripts/run_era5_stgan.py train` seleziona GAT per default. L'API generica
`fit_and_score_stgan` mantiene il default ConvGRU per compatibilità PVGIS; per GAT usare
`spatial_encoder="gat"` o `STGANGATConfig`. Checkpoint identificati come `STGAN_GAT`, con
grafo e ordine dei nodi salvati; il loader supporta anche i checkpoint `STGAN_CONVGRU`.

```text
recent (B,R,N,15), tutte le celle valide
  -> GAT 1: 4 head x 16 dimensioni, concatenazione, ELU
  -> GAT 2: 4 head x 32 dimensioni, media delle head
  -> se R>1: celle ConvGRU esistenti con kernel 1x1, solo tempo per nodo
  -> se R=1: nessun modulo temporale recent
  -> spatial Dropout(0.2) -------------------------+
                                                 |
trend (B,N,56,15) -> LSTM(15,64,2)                 |
  -> ultimo stato -> temporal Dropout(0.2) -------+-> concat -> fusion Dropout(0.2)
calendar (B,31) -> Linear(31,64), ReLU ------------+    -> Conv2d(160,15,1), Tanh
                                                      -> prediction (B,N,15)
```

R=recent_steps; 56 timestamp a 3 ore mantengono le 168 ore del trend ERA5.
La GAT è applicata indipendentemente a ogni timestamp, vettorizzando B e R.
Con R>1 rimangono le equazioni e gli strati GRU esistenti, con convoluzioni 1x1:
la sola GAT combina nodi diversi. Nessun tensore di patch entra nel Generator GAT.
Il Trend LSTM, il calendario, la concatenazione, i tre dropout e la proiezione finale
sono quelli originali. Ogni nodo usa la propria sequenza trend.

Il grafo è fisso e diretto: ogni cella valida riceve messaggi dagli otto vicini
immediati validi e da se stessa. Coppie geografiche presenti in entrambe le direzioni;
nessun wrap ai bordi e nessun collegamento attraverso celle escluse. L'ordine dei nodi
è identico a quello delle location nei dati. Lo strato 1 raggiunge la prima corona;
lo strato 2 arriva alla seconda corona lungo i cammini presenti.

Attenzione additiva multi-head implementata con PyTorch nativo: softmax stabile per
destinazione tramite scatter_reduce/scatter_add. Memoria del grafo O(N+E), attenzione
O(B*R*E*heads), messaggi O(B*R*E*heads*channels). Nessuna adjacency N×N, nessun
fully-connected, nessuna dipendenza aggiunta.

### Discriminator, loss e scoring

`STGANDiscriminator` in `model.py` è byte-per-byte invariato rispetto al commit CNN.
Riceve le stesse patch mascherate, ricostruite **dopo** la previsione globale; queste
patch appartengono solo al Discriminator e alla riduzione della reconstruction.
Rimangono BCE real=0/fake=1, reconstruction weight=500, media per celle/feature valide
per patch e poi per campioni, Adam con lr=1e-3, D update seguito da G update.

Due forward G completi e indipendenti per ogni step: il primo senza gradienti per D,
il secondo per G. Le patch D vengono elaborate in chunk e le loss pesate per il numero
di centri; un solo optimizer step per rete. Per G si accumula il gradiente rispetto
alla previsione globale, quindi lo si propaga attraverso G una sola volta. Il confronto
con l'intero batch senza chunk verifica loss, gradienti e aggiornamenti Adam.

MC scoring riattiva solo i tre dropout di G, ripete tutto G per ciascuno dei 20 campioni,
mantiene D in eval e ripristina gli stati precedenti. La reconstruction dello score
rimane la media sulla patch; i feature residuals rimangono quelli della cella centrale.
Restano `save_raw_mc=False`, scratch memmap, aggregazione chunked, media e deviazione
standard di popolazione. I raw MC temporanei vengono eliminati dopo l'aggregazione.

### Shape e output esterni

| Tensore | Shape |
|---|---|
| recent | B,R,N,15 |
| trend | B,N,56,15 |
| mask nodi | B,N,1; tutti i nodi sono celle valide |
| edge_index | 2,E, int64 |
| GAT 1 | B*R,N,64 |
| GAT 2 | B*R,N,32 |
| fusion | B,N,160 |
| previsione G / target | B,N,15 |
| input history D per chunk | C,R,15,3,3 |
| componenti MC temporanei | M,T,N |
| anomaly_mean / anomaly_std | T,N |
| feature_scores | T,N,15 |
| anomaly_mean_cube / uncertainty_cube | T,H,W |

`STGANResult`, rimappatura geografica, threshold, opening/closing, clustering e temporal
linking non sono stati modificati. I valori delle previsioni e degli eventi possono
cambiare per il nuovo modello; il formato e gli algoritmi di post-processing rimangono
identici. I test ripercorrono scoring -> checkpoint -> cubi -> eventi sintetici.

## Nodi, archi e parametri

Non è presente un cache ERA5 reale preparato con maschera statica nel workspace:
il numero effettivo di celle valide sarà ricavato da quel cache ed esportato nei metadata.
Nessuna lettura/training dei dati ERA5 reali è stata eseguita.

| Griglia sintetica | Nodi | Archi diretti, self-loop inclusi |
|---|---:|---:|
| 3×4 completa, test | 12 | 70 |
| 12×16 completa, benchmark piccolo | 192 | 1.564 |
| 81×131 completa, estensione geografica ERA5 richiesta | 10.611 | 94.231 |

Per un rettangolo completo: E=(3H−2)(3W−2). Con maschera: N è il numero di celle valide,
E è contato sui soli vicini presenti ed E≤9N. I test coprono anche buchi, permutazioni,
nodi isolati, angoli con 4 archi entranti e bordi con 6.

Parametri con F=15, hidden_size=64, n_layers=2, cnn_channels=32, cnn_layers=2,
gat_hidden_dim=16 per head, gat_heads=4, gat_layers=2, recent_steps=1:

| Modello | Generator | Discriminator | Totale |
|---|---:|---:|---:|
| CNN/ConvGRU sorgente | 156.303 | 125.729 | 282.032 |
| GAT | 68.111 | 125.729 | 193.840 |

## Memoria, batch e benchmark sintetico

Default ERA5: batch_size=1 e score_batch_size=1 **timestamp globale**. Il vecchio batch
contava patch tempo-location ed è quindi un'unità diversa. Nessun sottocampionamento
dei nodi: ogni campione include l'intero grafo. `train_samples_per_epoch`, se positivo,
ora conta timestamp con rimpiazzo; con 0 vengono visitati tutti i timestamp target e
tutti i nodi a ogni epoca. Cambiano raggruppamento e ordine dei batch rispetto alla CNN.

Controlli esposti: `gat_hidden_dim`, `gat_heads`, `gat_layers` (vincolato a 2),
`discriminator_chunk_size=256`, `trend_chunk_size=256`, batch training e scoring,
recent_steps. Le dimensioni del trend e della proiezione finale restano invariate.

Il primo tentativo sulla griglia 81×131, senza chunking del trend, ha prodotto CUDA OOM
con limite dell'allocator all'80% della GPU da 4 GB. Il medesimo LSTM viene ora eseguito
per blocchi di nodi; in training si ricalcolano le attivazioni dei blocchi nel backward
(`checkpoint`, non-reentrant). Non cambia la rete né il dropout; si scambiano calcolo
aggiuntivo e memoria. Il test di equivalenza confronta anche questa modalità.

Ambiente: PyTorch 2.11.0+cu128, Windows, CPU con 2 thread, NVIDIA GeForce GTX 1650 with
Max-Q Design. Dati esclusivamente casuali, 15 feature, recent=1, trend=56, Adam completo,
due forward G per step. Dopo 1 warmup: 3 step nel benchmark piccolo, 2 nel completo.

| Device | N | B | Timestamp training/s | Secondi MC20 per batch | Picco CUDA training MiB | Picco CUDA MC MiB |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 192 | 1 | 11,142 | 0,443 | — | — |
| CPU | 192 | 2 | 9,291 | 0,811 | — | — |
| GPU | 192 | 1 | 33,739 | 0,142 | 134,03 | 96,50 |
| GPU | 192 | 2 | 37,996 | 0,230 | 169,60 | 117,67 |
| GPU | 10.611 | 1 | 0,952 | 5,361 | 430,96 | 296,58 |

Le misure CUDA riportano memoria allocata da PyTorch; memoria reserved e picco RSS
del processo sono nei JSON. Il picco RSS CPU comprende runtime e librerie ed è un
massimo cumulativo di processo, non un conteggio delle sole attivazioni. Tempi esclusi:
I/O dati, normalizzazione e post-processing; questi risultati non stimano la durata
di un training ERA5 reale e non ne misurano accuratezza o qualità dell'incertezza.

La griglia completa usa 1.507.696 byte per edge_index; una adjacency float32 N×N
richiederebbe 450.373.284 byte, **mai allocati** dall'implementazione.

Comandi riproducibili (solo sintetici):

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_stgan_gat.py --output .analysis/stgan_gat_mc_dropout/benchmark_small.json
.\.venv\Scripts\python.exe scripts/benchmark_stgan_gat.py --device cuda --height 81 --width 131 --batches 1 --steps 2 --warmup 1 --cuda-memory-fraction .8 --output .analysis/stgan_gat_mc_dropout/benchmark_full.json
```

Per una futura esecuzione autorizzata, la CLI ERA5 espone `--spatial-encoder gat`,
`--gat-hidden-dim 16 --gat-heads 4 --gat-layers 2`, `--batch-size 1 --score-batch-size 1`.
Non è stata invocata la CLI di training su un cache reale.

## Test finali GAT e regressione CNN

- `python -m unittest discover -s tests -p test_stgan*.py`: **64 test OK**,
  inclusi i 12 nuovi test GAT e i 52 test CNN/MC esistenti.
- `python -m unittest discover -s tests -p test_era5*.py`: **11 test OK**.
- `python tests/test_stgan_gat.py`: **12 test OK** dopo il chunking del trend.
- CLI `scripts/run_era5_stgan.py train --help`: OK.
- Benchmark CPU/GPU e griglia completa GPU: OK.
- `git diff --check`: OK.

Copertura nuova: 8-neighborhood/self-loop, angoli/bordi/buchi e assenza di wrap;
ordine nodi; shape per recent=1 e recent=3; due layer effettivi e raggio esattamente
due hop via gradienti; gradienti finiti non nulli su tutti i parametri G; controllo
delle allocazioni PyTorch durante forward/backward che fallisce su shape N×N;
adapter D equivalente al dataset patch originale; loss/gradienti/Adam equivalenti
con chunk D e trend; due forward G e nessun gradiente G nell'update D; MC20 stocastico,
stati eval ripristinati, MC disabilitato/dropout=0 deterministici; memory/memmap
equivalenti e cleanup raw; DataLoader Windows spawn con memmap; checkpoint; calibration
invariata alterando soltanto il test; score oltre range senza clipping; rimappatura
dei cubi mean/std ed esportazione eventi.

La suite ha mostrato avvisi preesistenti NumPy/netCDF e cuDNN flatten_parameters;
nessun test fallito. I test del downloader (30 OK) erano già stati eseguiti prima
del commit CNN; quel codice non è cambiato nel branch GAT.

## Differenze circoscritte rispetto al commit CNN

Nuovi moduli `graph.py`, `gat.py`, `graph_data.py`; selezione backend e configurazione;
loader checkpoint GAT; adapter del training/scoring per timestamp globali; script
benchmark; test GAT. Nessuna modifica a `model.py` (contiene la CNN e il Discriminator),
`physiq_pv/era5/*`, normalizzatore, sampler, caricamento memmap originale, download,
preparazione split o API dei risultati. Il ramo GAT riusa queste componenti.

Il report e i JSON di benchmark sono conservati esplicitamente anche se `.analysis/`
è normalmente ignorata da Git. Il worktree temporaneo CNN, verificato pulito e sul
commit sorgente anche a fine lavoro, è stato rimosso per liberare il branch al normale
checkout. Gli altri file di audit e lo stash di sicurezza restano locali.
