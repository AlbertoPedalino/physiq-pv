# Anticipo: regole fissate prima di vedere i risultati

Analisi chiesta da Simone (8 novembre 2026): il detector, ricostruendo i dati ogni 3 ore come in un flusso, comincia a segnalare un'anomalia meteorologica documentata prima che inizi, nel tempo o nello spazio? Eventi: i 39 di `lista_esatta.csv`. Metodi: $G_{\mathrm{n}}$ di optuna-27 e le 8 baseline della Sezione 6.6 della tesi. Ogni regola è fissata qui, con la data, prima di calcolare qualsiasi risultato dell'anticipo.

## Regola 1. Inizio dell'anomalia misurato in ERA5 (fissata l'8 novembre 2026)

Base comune: ERA5 sullo stesso reticolo del detector (0.5°, ogni 3 ore, 00–21 UTC), percentili per cella stimati sugli anni 1980–1999. L'inizio si cerca da 3 giorni prima della data di inizio della fonte fino alla data di fine. Le aree sono pesate con il coseno della latitudine.

| Tipo | Variabile | Soglia | Area | Durata | Inizio | Fonte |
|---|---|---|---|---|---|---|
| Caldo | massima giornaliera di t2m sugli 8 istanti del giorno UTC | oltre il 90° percentile, finestra di calendario ±15 giorni | un quadrato di 7×7 celle (3.5°, la misura dispari più vicina ai 3.75° della fonte) centrato in una cella del rettangolo, con almeno il 60% dell'area oltre la soglia | almeno 3 giorni consecutivi con un quadrato così | 00 UTC del primo di quei giorni | Russo et al. 2015 (90°, 31 giorni); Lavaysse et al. 2018 (3 giorni); Stefanon et al. 2012 (quadrato, 60%) |
| Freddo | minima giornaliera di t2m | sotto il 10° percentile, ±15 giorni | come il caldo | come il caldo | come il caldo | Lavaysse et al. 2018; ETCCDI (CSDI) |
| Vento | velocità a 10 m, $\sqrt{u10^2+v10^2}$, a ogni istante | oltre il 98° percentile locale della massima giornaliera, su tutti i giorni della base | cluster di celle adiacenti (lato in comune) oltre la soglia, con area di almeno 155 000 km² (due quadrati di 2.5° all'equatore), che tocca il rettangolo | un cluster così tocca il rettangolo in almeno 8 istanti consecutivi (24 h) | il primo di quegli istanti | Klawa & Ulbrich 2003 (98°); Leckebusch et al. 2008 (cluster, area minima, almeno 24 h) |
| Pioggia | totale giornaliero stimato = somma degli 8 accumuli orari di tp × 3 (i dati hanno l'accumulo dell'ultima ora a ogni istante, non il totale di 3 ore) | oltre il 95° percentile dei giorni piovosi (≥1 mm) della base, senza finestra di calendario | come il caldo | un giorno | 00 UTC del primo giorno con un quadrato così | ETCCDI (R95p) |
| Polvere | nessuna variabile fra le 15 | — | — | — | 00 UTC della data di inizio della fonte | nessuna regola pubblicata con sole variabili meteorologiche |

Nostre semplificazioni, da scrivere come tali:
- Vento: il tracking di Leckebusch (centro di massa, sovrapposizione del 30%, 720 km) è sostituito dalla presenza nel rettangolo. L'adiacenza per lato in comune è una nostra scelta.
- Pioggia: il totale giornaliero è stimato da 8 ore su 24 (× 3); un temporale breve può cadere fuori dalle ore campionate.
- Caldo e freddo: la massima e la minima giornaliere vengono da 8 istanti, non dai valori orari.
- Sensibilità (Gilleland et al. 2009): il vento si ripete con metà area, 77 500 km².
- Se nella finestra nessuna area soddisfa la regola, l'inizio è la data della fonte, come per la polvere, e l'evento è segnato.

## Regola 2. Anticipo e controllo (fissata l'8 novembre 2026)

| Pezzo | Regola | Fonte |
|---|---|---|
| Misura per istante | % di celle del rettangolo segnalate dal metodo, ciascuno con la propria soglia (95° percentile sugli anni di validazione, come nella tesi) | Ebert 2008 (copertura minima di una regione) |
| Finestra | da 72 h prima dell'inizio misurato (regola 1) alla data di fine della fonte, ogni 3 ore | Lavin & Ahmad 2015 (NAB) |
| Controllo | per ogni evento, 300 date casuali nello stesso rettangolo, negli stessi mesi dell'evento, negli anni di test 2005–2025, con finestre che non si sovrappongono a nessuno dei 39 eventi; ogni data casuale è trattata esattamente come l'evento | Grazzini et al. 2021; Laken & Čalogović 2013; Rao et al. 2019 |
| Segnale | la % dell'evento supera la soglia della finestra: il 95° percentile, sulle 300 date casuali, del loro massimo sugli stessi ritardi. Una soglia per la finestra prima dell'inizio e una per quella dopo. (Corretta l'8 novembre 2026, vedi sotto.) | Groppe, Urbach & Kutas 2011, sez. 2.2 |
| Rilevato | il massimo fra l'inizio misurato e la data di fine della fonte supera la soglia della finestra dopo | Tatbul et al. 2018 |
| Anticipo | ore fra il primo segnale prima dell'inizio e l'inizio; conta solo se l'evento è anche rilevato, altrimenti è un falso allarme | Ghorbani et al. 2024 (PATE); Lavin & Ahmad 2015 |
| Falsi allarmi | quota degli eventi con segnale prima dell'inizio ma non rilevati; per le date casuali, la quota che supera la soglia (circa 5% per costruzione: è il controllo che la regola tiene) | JWGFVR (verifica categorica) |
| Risultati | per tipo e metodo: quota di eventi rilevati, anticipo mediano e scarto interquartile, quota di eventi con anticipo > 0, falsi allarmi | Matsuoka et al. 2018; JWGFVR |
| Sensibilità | finestra di 24, 48 e 72 h; segnale al 95° e al 99° percentile | Paparrizos et al. 2022 (VUS); Gilleland et al. 2009 |

Figure: curve composite per ritardo, per tipo, con la fascia 2.5–97.5% delle date casuali (Laken & Čalogović 2013). La fascia per ritardo serve solo a leggere la figura; i numeri usano la soglia della finestra.

### Correzione dell'8 novembre 2026 (prima di qualsiasi risultato reale)

La prima versione confrontava ogni ritardo, da solo, col 95° percentile delle date casuali a quel ritardo. Provata sui dati sintetici del test, con uno score di puro rumore, dava l'evento "rilevato" e un anticipo di 72 h, e il 40–65% delle date casuali con un segnale. Il motivo: ogni ritardo supera la soglia per caso il 5% delle volte, ma una finestra ha da 8 a oltre 40 ritardi, e "almeno uno sopra" capita quasi sempre. È il problema dei confronti multipli.

La correzione usa il massimo sulla finestra, come nei test di permutazione con controllo dell'errore di famiglia (Groppe, Urbach & Kutas 2011, Psychophysiology 48:1711, sez. 2.2: la distribuzione del valore più estremo sull'intera famiglia di test, che "guarantees strong control of FWER"). Con una sola soglia per finestra, una data casuale la supera ovunque nella finestra solo il 5% delle volte. Nel test il rumore torna a 0 eventi rilevati e a 0% di date casuali con segnale, mentre lo score costruito con anticipo conserva le sue 24 h. Nessun numero reale era stato calcolato prima della correzione.

## Regola 3. Anticipo nello spazio (fissata l'8 novembre 2026)

| Pezzo | Regola | Fonte |
|---|---|---|
| Anello | celle fuori dal rettangolo entro la distanza d dal suo bordo (distanza sulla sfera dal centro della cella); le celle fuori dal dominio si scartano | Zschenderlein et al. 2019 |
| Larghezze | d = 100, 250 e 500 km, riportate tutte; 100 km (2 celle) è sotto il minimo di 5 celle di Roberts & Lean 2008, quindi è solo una tolleranza stretta e non entra nelle conclusioni | Lagerquist et al. 2019; Zschenderlein et al. 2019; Ebert 2008; Gilleland et al. 2009 |
| Direzione | tutto l'anello, senza scegliere a mano una direzione di provenienza | — |
| Misura, controllo, segnale | come nella regola 2, sulle celle dell'anello | Laken & Čalogović 2013; Grazzini et al. 2021 |
| Anticipo spaziale | segnale nell'anello prima che nel rettangolo e prima dell'inizio; conta solo se l'evento è poi rilevato nel rettangolo | Ghorbani et al. 2024 (PATE) |

Cautela da scrivere: un segnale nell'anello prima dell'inizio può essere l'anomalia che arriva da fuori o un'altra anomalia vicina.

## Figura principale (decisa l'8 novembre 2026)

Una sola figura, da leggere a colpo d'occhio: 5 riquadri, uno per tipo. Asse x: ore dall'inizio misurato, da −72 a +48. Asse y: % di celle segnalate.
- Modello ($G_{\mathrm{n}}$): linea piena nel rettangolo, tratteggiata nell'anello da 250 km.
- Metodi semplici: distanza dal normale e variazione in 3 h, due linee separate (misurano stato e cambiamento).
- Metodi statistici: la mediana dei 6 shallow, con una fascia chiara dal minimo al massimo.
- Grigio: la fascia 2.5–97.5% delle date casuali (il livello normale).
- Verticale a 0 (inizio misurato); per la polvere 0 è la data della fonte.
Sotto, una tabella breve per tipo: quota di eventi rilevati e anticipo mediano per modello, semplici e statistici. Il resto (100 e 500 km, sensibilità, i 6 shallow uno per uno) va in appendice.
