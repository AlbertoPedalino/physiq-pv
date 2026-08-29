# STGAN: allineamento al paper e adattamenti PVGIS

Riferimento: Deng et al., *Graph Convolutional Adversarial Networks for
Spatiotemporal Anomaly Detection*, IEEE TNNLS 33(6), 2022,
DOI `10.1109/TNNLS.2021.3136171`; repository ufficiale `dleyan/STGAN`, commit
verificato `20d2f6b365ea003500a57737647a846fb41267aa`.

## Parti riprodotte

- sottografo di nove località, con il target in posizione zero;
- peso gaussiano `exp(-d^2/sigma^2)`, self-loop e normalizzazione del
  repository ufficiale;
- recent GCGRU, trend LSTM e weekday/hour one-hot;
- target GAN reale/normale `0` e generato/fake `1`;
- loss generatore `500*MSE + BCE` e loss discriminatore media reale/fake;
- score `normalize(sG) + normalize(D(real)-D(fake))`, con min-max globale
  separato dei due componenti sull'intero test e `lambda=1`;
- configurazione di riferimento: 6 epoche, batch 256, Adam `1e-3`, seed 20.

La normalizzazione delle feature viene stimata soltanto sul training. I due
componenti dello score sono invece normalizzati globalmente sul test, come
richiesto dal protocollo di ranking del paper; non vengono usate label.
`pv_power_output` non è un input del detector: STGAN rileva pattern
meteorologici spaziotemporali senza conoscere target o label SDE.

## Grafo PVGIS

Il paper dispone di una topologia fisica (rete stradale o regioni adiacenti).
PVGIS non fornisce un equivalente universale. La pipeline usa quindi un unico
surrogato: un KNN geografico diretto. Ogni nodo ha archi uscenti verso le otto
località più vicine e ogni sottografo contiene il target più questi otto nodi,
come nel paper. I pesi gaussiani, i self-loop e la normalizzazione seguono il
repository ufficiale. L'inferenza degli archi dalla distanza resta l'unico
adattamento strutturale sostanziale e non viene descritta come topologia fisica.

## Continuità train-test

Per il primo target del test servono le osservazioni immediatamente precedenti.
La pipeline premette quindi al test le ultime `trend_steps` ore del training
(168 nella configurazione di riferimento). Queste righe sono soltanto input di
contesto: timestamp, score e decisioni restituiti appartengono tutti al test.
La normalizzazione resta fit sul training e la pipeline fallisce se il confine
temporale non è regolare e contiguo.

## Decisione top-K

Il runner espone un solo comportamento. Ordina globalmente gli score test di
tutte le coppie tempo-località e marca esattamente il `K%` più alto, con
`--paper-top-k-percent K`. Il top-K non utilizza label, ma usa l'intera
distribuzione test e presuppone un budget di valutazione a priori. Poiché la
decisione è basata sul rango e non su una soglia trasferibile, `threshold`
rimane `NaN`; `global_rank`, `global_percentile` e `is_anomaly` descrivono la
decisione senza ambiguità nei pareggi.
