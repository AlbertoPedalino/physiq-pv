# ERA5 STGAN: protocollo del punteggio

Riferimento: Deng et al., *Graph Convolutional Adversarial Networks for
Spatiotemporal Anomaly Detection*, IEEE TNNLS 2022, DOI
`10.1109/TNNLS.2021.3136171`. La formula e la scelta di `lambda=1` seguono
l'implementazione di riferimento gia presente nel branch `feat/stgan-paper`
(`docs/STGAN_PAPER_ALIGNMENT.md`, sezione sullo score). Il `tester.py` pubblico
degli autori esporta i componenti grezzi e non esegue la fusione finale.

- Training: tutte le osservazioni fino al 2004 incluso. Il cache preparato
  conserva 2003-2004 nel campo storico `calibration`, ma il runner lo aggiunge
  al training. Il test parte nel 2005; nessuna label di test entra nel training.
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
