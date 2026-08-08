# MTGFlow — assunzioni, limiti e uso in tesi

Questo file raccoglie ciò che va saputo *spiegare* di MTGFlow, non ciò che va
letto riga per riga. MTGFlow è strumentazione di misura: il contributo della
tesi è la backbone STGNN+SDE e la quantificazione dell'incertezza, mentre il
detector serve a produrre una definizione operativa di "condizione anomala"
con cui stratificare la valutazione.

Documenti complementari: [`MTGFLOW_MODEL.md`](MTGFLOW_MODEL.md) (teoria ed
equazioni), [`CODE_FLOW_MTGFLOW.md`](CODE_FLOW_MTGFLOW.md) (flusso runtime e
file), [`MTGFLOW_PAPER_ALIGNMENT.md`](MTGFLOW_PAPER_ALIGNMENT.md) (fedeltà al
paper e comandi).

## 1. Cosa fa, in tre frasi

MTGFlow stima la **densità congiunta** di finestre multivariate tramite un
normalizing flow condizionato; non predice una variabile target. Il training
è **label-free** su dati potenzialmente contaminati: non assume che lo storico
sia normale, assume che l'anomalo sia raro e finisca nelle code della densità
stimata. Lo score di anomalia è la **negative log-likelihood**: bassa densità
sotto il modello ⇒ score alto.

## 2. La catena, a livello di scatole

```text
finestra (60h, 3 variabili)
  → z-score train-only                       (Eq. 5)
  → self-attention → grafo dinamico A (3x3)  (Eq. 6-7)
  → RNN sullo stato per entità
  → conditioner spazio-temporale C           (Eq. 8)
  → MAF entity-aware condizionato da C       (Eq. 2-4, 9, 11)
  → log-likelihood → score S_c               (Eq. 12, 14)
  → soglia IQR stimata sul training          (Eq. 13, 15)
```

Il grafo è **appreso dai dati**, non imposto da una topologia fisica, e cambia
finestra per finestra: è la parte "multivariata dinamica" del metodo.

## 3. Unità di analisi: un modello per località

Ogni località ha il proprio modello, addestrato sui propri 14 anni. Il
confronto è **temporale** (questa finestra contro il clima storico *di quella
località*), non spaziale. Non esiste un grafo fra località: le 1149 località
non si osservano fra loro.

## 4. Perché il grafo è 3×3 e non 1149×1149

Il grafo di MTGFlow collega le **variabili** (`solar_irradiance_poa`,
`temperature_2m`, `wind_speed_10m`), non i nodi geografici. Estenderlo alle
località sarebbe insostenibile: il costo dell'einsum di attenzione scala con
`B·K²·M·h`, cioè `256·1149²·60·32 ≈ 6.5·10¹⁴` MAC per batch contro
`≈ 4.4·10⁹` con `K=3` — circa cinque ordini di grandezza. La dipendenza
spaziale è modellata a valle, dalla GAT della backbone STGNN, non qui.

## 5. Assunzioni di adattamento

| # | Assunzione | Motivo |
|---|------------|--------|
| 1 | Dominio PVGIS (3 canali climatici) invece dei benchmark SWaT/WADI/PSM | Non esiste ground truth di anomalia per il clima PV: si trasferisce il *metodo*, non si replicano i numeri del paper |
| 2 | `pv_power_output` escluso dalle feature del detector | È il target del forecaster: includerlo sarebbe leakage di supervisione. Il guard è in `anomaly_detection/common.py:detector_features` |
| 3 | Split 2005–2018 (train) / 2019 (test) | Normalizzazione, pesi e soglie derivano **solo** dal training. "Anno pulito" significa non toccato da statistiche del 2019, non privo di anomalie |
| 4 | `--score-stride 1` invece di 10 | Serve uno score per ogni ora del 2019 per il join con le predizioni STGNN. Registrato nei metadati come `dense_hourly_window_adaptation` |
| 5 | Solo MTGFlow base, senza `MTGFlow_cluster` (Eq. 10) | Il clustering KShape ha senso con molte entità eterogenee; con `K=3` non aggiunge nulla |
| 6 | `input_size=1` (tensorizzazione pointwise) dal repo ufficiale | Dettaglio non specificato dal paper; policy dichiarata: paper quando esplicito, repo quando sottospecificato |

## 6. Limiti che si propagano ai risultati

- **Detector non validato sul dominio.** Nessuna ground truth PV consente di
  misurare AUROC come nel paper. Le etichette sono una definizione operativa,
  non una verità.
- **Granularità della finestra.** Un flag riguarda le 60 ore precedenti, non
  l'ora singola; `timestamp` è alias di `window_end`. Con stride 1 le finestre
  si sovrappongono per 59/60, quindi i flag sono fortemente autocorrelati.
- **Soglia da training contaminato.** `Q3 + 1.5·IQR` è calibrata su score che
  contengono già anomalie: la soglia è conservativa per costruzione.
- **Soglie per località.** Ogni località usa la propria soglia, quindi un flag
  significa "raro per questa località", non "raro in assoluto".

## 7. Aggregazione regionale

Per timestamp si contano le località segnalate:

```text
anomaly_fraction = n_anomaly / n_scored     → estensione dell'evento
mean_excess      = media(score - soglia)    → intensità dell'evento
```

Le due misure vanno riportate insieme: molte località appena sopra soglia e
poche località molto sopra soglia sono eventi diversi. Le località sono
**spazialmente correlate**, quindi `anomaly_fraction` è una misura descrittiva
di estensione — nessun test binomiale di significatività è valido, perché
l'ipotesi di indipendenza non regge.

## 8. Mitigazione: doppia definizione indipendente

Esiste un secondo etichettatore, climatologico, basato su bande quantiliche
marginali (`physiq_pv/data/pvgis_labels.py`, colonna `label`). MTGFlow lavora
invece sulla densità congiunta e produce `is_anomaly`. Le due definizioni sono
metodologicamente indipendenti e hanno contratti di colonna distinti per
costruzione. La conclusione della tesi non poggia su un singolo etichettatore:
se il degrado dell'accuratezza in condizioni anomale si osserva sotto entrambe
le definizioni, il risultato non è un artefatto della scelta del detector.

## 9. Cosa non serve saper spiegare

Derivazione del cambio di variabile e del log-Jacobiano, motivo formale per cui
`input_size=1` preserva la biiezione, dettagli di `_regular_window_starts`,
forma dell'`einsum`, nomi delle classi in `model.py`. Sono dettagli di
implementazione di un metodo di terzi: si citano, non si difendono.
