# Literature Positioning

Questo documento raccoglie i riferimenti utili per posizionare PhysiQ-PV rispetto alla letteratura su PV forecasting, data quality e approcci physics-informed.

## Idea Centrale

Il contributo principale non e' proporre una architettura sempre piu' complessa, ma spostare parte dell'attenzione sul dato:

```text
model-centric learning  ->  data-centric + physics-informed learning
```

La qualita' del dato viene modellata esplicitamente e usata nel training, senza diventare un gate rigido.

## Riferimenti Vicini

### Data Quality-Aware Forecasting

**Sundararajan et al. (2022)**  
*A Data Quality-Aware Framework to Reliably Forecast Photovoltaic Generation and Consumer Load for an Improved Resilience of Microgrids*  
IEEE PEDG 2022 / Oak Ridge National Laboratory.

Link:
- https://www.ornl.gov/publication/data-quality-aware-framework-reliably-forecast-photovoltaic-generation-and-consumer
- https://doi.org/10.1109/PEDG54999.2022.9923109

Punto rilevante:

```text
Il paper introduce un framework data-quality-aware per PV/load forecasting.
La qualita' del dato viene usata per monitorare missing values, drift,
incertezza e performance, e per attivare classi diverse di modelli/use case.
```

Differenza rispetto a PhysiQ-PV:

```text
Nel framework ORNL, la qualita' del dato guida la scelta della strategia
o classe di forecasting.

In PhysiQ-PV, la qualita' del dato entra direttamente nel processo di
apprendimento: come feature causale storica, come peso soft nella loss
e come segnale diagnostico per continual learning.
```

### Physics-Guided PV Forecasting

**Yu, Loskot, Gao (2026)**  
*PhysEmbedFormer: a physics-guided interpretable architecture for days-ahead forecasting of PV power*  
Scientific Reports, 16, Article 4705.

Link:
- https://www.nature.com/articles/s41598-025-34874-8

Punto rilevante:

```text
Il lavoro segue il filone physics-guided/physics-informed:
usa informazione fisica per migliorare interpretabilita' e robustezza
del forecasting PV.
```

Differenza rispetto a PhysiQ-PV:

```text
PhysEmbedFormer lavora soprattutto sulla architettura e sulla
scomposizione physics-guided del problema.

PhysiQ-PV usa invece un Quality Score fisicamente interpretabile come
segnale operativo di qualita' del dato, integrandolo nel training e nella
diagnostica.
```

## Formulazione Del Contributo

Formulazione breve:

```text
A differenza di approcci data-quality-aware che utilizzano la qualita'
del dato per selezionare o attivare strategie di forecasting differenti,
questo lavoro incorpora la qualita' del dato direttamente nel training,
sia come informazione causale in input sia come peso soft nella funzione
di perdita.
```

Formulazione estesa:

```text
Il lavoro proposto integra un Quality Score fisicamente interpretabile nel
processo di apprendimento di un modello PV distribuito. Il QS non viene
usato come gate per scartare campioni, ma come segnale soft: entra nella
finestra storica osservata, pesa la loss in modo continuo e resta
disponibile per diagnostica e continual learning.

In questo modo, il modello non cambia classe e non elimina i campioni
degradati, ma apprende in modo differenziato in funzione della loro
affidabilita' fisica.
```

## Claim Difendibile

Claim consigliato:

```text
In letteratura esistono approcci physics-informed e data-quality-aware
per il forecasting PV, ma PhysiQ-PV si distingue per l'integrazione di un
Quality Score fisicamente interpretabile direttamente nel training, non
come filtro rigido ma come segnale soft e diagnostico, validato su una
flotta reale eterogenea di impianti non selezionati.
```

Claim da evitare:

```text
Nessuno ha mai usato la qualita' del dato nel PV forecasting.
```

Motivo:

```text
La letteratura su data quality, anomaly detection, missing data,
physics-informed forecasting e drift esiste gia'. La parte originale e'
la combinazione operativa: QS fisico, feature causale, loss soft-weighted,
nessun gate e validazione su flotta reale.
```

## Risultato Sperimentale Da Collegare

A parita' di architettura e pipeline, la rimozione di `QS` e `m1_past`
peggiora il forecasting PV. Configurazione corrente (η stimata via WLS through origin, n=3.06 M campioni diurni):

```text
QS baseline:
MAE  = 0.0891
RMSE = 0.1440
r    = 0.906

No-QS base:
MAE  = 0.0960
RMSE = 0.1517
r    = 0.896
```

Riduzione errore con QS:

```text
MAE  -7.2%
RMSE -5.1%
```

Nel caso critico `mid-low QS` con produzione reale alta (`actual PV >= 0.8`), il modello QS-aware mantiene predizioni vicine al target reale, mentre il modello no-QS sotto-stima sistematicamente.

Questo supporta la tesi che il QS non sia solo una regolarizzazione, ma un segnale informativo utile per distinguere casi fisicamente diversi.
