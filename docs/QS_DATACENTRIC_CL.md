# PhysiQ-PV — Quality Score come framework data-centric per Continual Learning

Posizionamento concettuale: il Quality Score è un **segnale di controllo data-centric** che governa la pipeline di Continual Learning, indipendentemente dal modello base.

---

## 1. Tesi centrale

Il contributo non è un'architettura più complessa per il forecasting PV. Il contributo è il QS come segnale che guida il sistema di Continual Learning su flotte reali eterogenee.

Il QS non sostituisce il modello: lo affianca. Determina quando aggiornare, quali campioni privilegiare nel replay, quali impianti monitorare per degrado, quali eventi flaggare come anomali.

Il framework è agnostico al modello base: funziona con qualsiasi forecaster (ST-GNN, LSTM, Random Forest, baseline naive). Il modello fa predizioni, il framework decide se/come/quando aggiornarlo.

---

## 2. Ruoli del QS

| Ruolo | Dove agisce | Richiede QS in input al modello? | Modifica architettura? |
|---|---|---|---|
| Feature | forward pass | sì | sì (un canale in più) |
| Soft loss weight | training/fine-tuning step | no | no |
| Signal diagnostic / control | esterno al modello | no | no |

Ruoli 2 e 3 sono completamente separati dall'architettura. Il QS guida il processo di apprendimento e monitoraggio, non l'inferenza punto a punto.

---

## 3. Componenti del framework CL

### A. Replay buffer QS-weighted

```text
sample_weight ∝ QS_observed
loss_step = mean(sample_weight × MSE(pred, target))
```

Campioni con QS basso (rumorosi, sensori guasti, ombreggiamenti anomali) contribuiscono meno all'aggiornamento. Il dato non viene scartato, il suo peso è regolato dalla qualità.

### B. Quality-gated update rule

```text
if mean(QS_window) > 0.5:
    apply gradient step
else:
    skip update
```

Si evita catastrofic forgetting da batch di bassa qualità. Se un mese di dati è dominato da letture SCADA disconnesse, il modello non viene degradato.

### C. Drift detection via QS time series

```text
ADWIN(QS_plant[t]) → flag se cambio significativo
slope_OLS(QS_plant) over months → degrado sistemico
```

Identifica impianti che stanno degradando prima che si veda nei KPI di previsione. QS è early-warning indicator.

### D. Clustering soft-DTW su trajectorie QS

```text
QS_matrix (N_plants, T) → soft-DTW TimeSeriesKMeans → labels
```

Cluster tipici:
- impianti sani stazionari
- impianti con degrado lineare
- impianti con ciclo soiling stagionale
- impianti con failure improvviso

### E. Anomaly detection via spatial QS z-score

```text
z[plant, t] = (QS[plant, t] - mean_fleet[t]) / std_fleet[t]
```

- `z < -2` su singolo impianto → guasto isolato
- `mean_fleet[t]` drop con `std_fleet[t]` basso → evento regionale (nuvola, eclissi)

Distinzione automatica tra anomalie locali e globali, senza bisogno che il modello le abbia mai viste.

### F. Monitoraggio impianti più in salute

```text
top_k_healthy = argsort(mean(QS_plant), descending)[:k]
```

Impianti con QS storicamente alto e stabile diventano riferimento di flotta: i loro pattern sono benchmark per identificare deviazioni negli altri.

---

## 4. Tabella decisione CL guidata da diagnosi

QS multi-componente classifica la causa, framework agisce di conseguenza:

| Pattern QS | Causa | Update modello | Replay | Alert |
|---|---|---|---|---|
| QS singolo plant ↓↓ improvviso, m4↓ | sensor_failure | NO | escludi plant | urgente |
| QS singolo plant ↓ lineare, slope< 0 | panel_degradation | NO se transitorio, SÌ se permanente | mantieni storia | manutenzione |
| QS molti plant ↓ sincrono, m4 alto | regional_event | NO | mantieni | nessuno |
| QS oscillante stagionale | soiling | NO durante anomalia | mantieni storia | pulizia |
| QS stabile, errori ↑ | model_drift | SÌ (DER++ + QS-weighted) | replay attivo | nessuno |
| QS area geografica ↓ | local_perturbation | NO (transitorio fisico) | mantieni | nessuno |

Azione CL dipende dalla diagnosi, non solo dal drift flag generico. Senza classifier causale, CL ingenuo retraina ovunque QS scende → poison da sporcizia / sensori.

---

## 5. Indipendenza framework-modello

I componenti CL non chiedono al modello di sapere cosa è il QS. Funzionano con qualsiasi forecaster:

| Modello base | Replay QS-weighted | Update gating | Drift detection | Clustering | Anomaly detection |
|---|:---:|:---:|:---:|:---:|:---:|
| ST-GNN (attuale) | ✓ | ✓ | ✓ | ✓ | ✓ |
| LSTM | ✓ | ✓ | ✓ | ✓ | ✓ |
| Random Forest | ✓ | ✓ | ✓ | ✓ | ✓ |
| XGBoost | ✓ | ✓ | ✓ | ✓ | ✓ |
| Persistence baseline | ✓ | ✓ | ✓ | ✓ | ✓ |

Il QS framework è un layer sopra il modello, non un suo componente interno. In deploy industriale con modello legacy, il framework si applica senza ridisegnare l'architettura.

---

## 6. Pipeline retrain con QS pesato

```text
nuovi dati arrivano (mese M+1)
   ▼
calcola QS(t) per ogni sample (5 metriche m1-m5)
   ▼
drift detector ADWIN/KS su QS(t) → flag drift?
   ▼ se drift
classifier causale (cluster soft-DTW + z-score + slope) → diagnosi
   ▼
tabella decisione: update | skip | mask | flag
   ▼ se update
soft weighting:    loss = mean(qs_i^0.2 * MSE_i)
hard gate:         if mean(qs_batch) < 0.5 → skip
DER++ replay:      + α MSE(curr, old_pred) + β MSE(curr, ground_truth)
                   replay buffer pesato anch'esso da QS storico
   ▼
loss.backward(), optimizer.step()
```

Il modello non vede mai QS in input nel setting CL puro. QS controlla solo quando/come modello viene aggiornato.

---

## 7. Architettura logica

```
                 ┌─────────────────────────────────────┐
                 │   Framework Continual Learning      │
                 │           (contributo)              │
                 │                                     │
   QS signal ───▶│  ┌───────────────────────────────┐ │
                 │  │ replay buffer QS-weighted     │ │
                 │  │ quality-gated update rule     │ │
                 │  │ drift detection on QS series  │ │
                 │  │ clustering soft-DTW           │ │
                 │  │ spatial anomaly detection     │ │
                 │  │ healthy plant monitoring      │ │
                 │  └───────────────────────────────┘ │
                 │                  │                  │
                 │                  ▼                  │
                 │         decisioni di update         │
                 │                                     │
                 └──────────────────┬──────────────────┘
                                    │
                                    ▼
                 ┌─────────────────────────────────────┐
                 │   Modello base (intercambiabile)    │
                 │                                     │
                 │   ST-GNN + QS  (implementazione)    │
                 │   LSTM, RF, ...  (alternative)      │
                 └─────────────────────────────────────┘
```

Il QS vive nel layer di controllo. Il modello fa predizioni; il framework decide se, come e quando aggiornarlo.
