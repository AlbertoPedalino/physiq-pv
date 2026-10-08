# ERA5 STGAN: protocollo del punteggio

Riferimento: Deng et al., *Graph Convolutional Adversarial Networks for
Spatiotemporal Anomaly Detection*, IEEE TNNLS 2022, DOI
`10.1109/TNNLS.2021.3136171`. La formula e la scelta di `lambda=1` seguono
l'implementazione di riferimento gia presente nel branch `feat/stgan-paper`
(`docs/STGAN_PAPER_ALIGNMENT.md`, sezione sullo score). Il `tester.py` pubblico
degli autori esporta i componenti grezzi e non esegue la fusione finale.

- Split: training 1980-2003, validation 2004, test dal 2005 (`--validation-holdout`).
  Senza validation il training usa tutte le osservazioni fino al 2004 incluso, come
  nelle run precedenti. La validation serve solo al monitoraggio per epoca e
  all'objective dello sweep: non entra nello score del test, nei suoi min-max, in
  soglie o label. Nessuna label di test entra nel training.
- Le feature sono normalizzate con parametri stimati solo sul training.
- Per ogni coppia tempo-localita del test: `sG` e l'errore quadratico medio
  del generatore; `sD = D(reale) - D(generato)`.
- Il punteggio combinato e `minmax_test(sG) + minmax_test(sD)` con peso uno.
  Ogni min-max usa un solo minimo e massimo sull'intero prodotto tempo-localita
  del test. Una componente costante contribuisce zero.
- Con MC dropout, si stimano prima le mappe medie di `sG` e `sD`; i minimi e
  massimi si ricavano da queste mappe. Gli stessi intervalli trasformano ogni
  draw MC per calcolare la deviazione standard del punteggio combinato.
  Questa scelta e un adattamento MC: il paper non usa MC dropout.
- `events` usa il punteggio combinato per default. I componenti grezzi restano
  disponibili con `--score-component generator` o `discriminator` per diagnosi.

Il min-max globale usa tutti gli score del test senza usare le label: e una
valutazione retrospettiva. Aggiungere nuovi anni al test puo cambiare i minimi,
i massimi e il ranking combinato degli anni precedenti. La CNN/ConvGRU e un
adattamento architetturale del modello originale basato su GCGRU; il GAT su
griglia e un altro adattamento, non una replica dell'architettura originale.
La run gia avviata su un altro host non acquisisce queste modifiche da sola.

## Variante CNN full-grid (`--cnn-training-mode full_grid`)

Variante di training distinta dal baseline a patch, non una sua ottimizzazione.
Il default resta `patch`; codice in `physiq_pv/anomaly_detection/stgan/full_grid.py`.

- Unita del campione. `patch`: una cella target con la sua patch 3x3 a un
  timestamp. `full_grid`: l'intero campo `[H,W]` di un timestamp. In `full_grid`
  `--batch-size` e `--score-batch-size` contano timestamp (default 1 e 1).
- Optimizer step per epoca: `ceil(T*N/batch)` contro `ceil(T/batch)`. Con ERA5
  1980-2003 (T=70072, N=10611): 2.904.430 step con batch 256 contro 70.072 con
  batch 1, circa 41 volte meno aggiornamenti dei pesi per epoca. Le run delle due
  modalita non sono confrontabili a parita di epoche.
- Generatore: stessi moduli e stessi nomi dei parametri. Il ConvGRU scorre
  sull'intero campo, `[B,T,F,H,W] -> [B,F,H,W]`: il campo recettivo non e piu
  troncato al bordo della patch e ogni cella e ricostruita una volta sola. Trend
  LSTM e codifica temporale (`onehot` o `cyclic`) sono calcolati per cella.
- Discriminatore: resta locale 3x3. La patch di ogni cella viene estratta dal
  campo con un gather vettoriale e il discriminatore a patch le valuta tutte in
  una passata, con le stesse maschere e gli stessi zeri del dataset a patch.
  L'output di una cella e quello del discriminatore a patch sulla sua patch. Il
  ConvGRU di D ha padding dentro la patch: farlo scorrere sul campo non darebbe
  lo stesso valore, quindi il costo di D per cella resta quello del baseline.
- Loss: ricostruzione = media su celle con localita e feature del campo, poi
  sui timestamp; avversaria = media della BCE sugli output locali, uno per
  cella. Target, peso della ricostruzione, learning rate e rapporto D:G invariati.
- Score: stesse definizioni per coppia tempo-localita. `sG` e l'errore
  quadratico medio sulla patch 3x3 della cella, calcolato sul campo ricostruito;
  `sD` e la differenza degli output locali di D sulla stessa patch.
- Checkpoint: `cnn_training_mode` e salvato; il resume tra modalita diverse e
  rifiutato. Negli sweep W&B `cnn_training_mode` e una chiave di configurazione
  e richiede `batch_size` e `score_batch_size` in timestamp.
