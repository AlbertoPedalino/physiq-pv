# PhysiQ-PV — Continual Learning Framework

Pipeline di aggiornamento online + framing del Quality Score come segnale data-centric per CL safe su flotta reale.

---

## 1. Tesi centrale

Il contributo non è un'architettura più complessa per il forecasting PV. È il **QS come segnale di controllo data-centric** che governa la pipeline di Continual Learning, indipendentemente dal modello base.

Il QS non sostituisce il modello: lo affianca. Determina quando aggiornare, quali campioni privilegiare nel replay, quali impianti monitorare per degrado, quali eventi flaggare come anomali.

Il framework è **agnostico al modello base**: funziona con qualsiasi forecaster (ST-GNN, LSTM, Random Forest, baseline naive). Il modello fa predizioni, il framework decide se/come/quando aggiornarlo.

---

## 2. Ruoli del QS

| Ruolo | Dove agisce | Richiede QS aggregato in input al modello? | Modifica architettura? |
|---|---|---|---|
| Feature m1..m5 separati | forward pass | no (solo componenti m1..m5, non QS aggregato) | sì (5 canali) |
| Soft loss weight | training step | no | no |
| Signal diagnostic / control | esterno al modello + notebook | no | no |

**Stato run corrente:**
- Ruolo 1: m1..m5 individuali come feature canali 5..9. QS aggregato NON è feature.
- Ruolo 2: disattivato (loss non QS-weighted, rimosso con lagged power on).
- Ruolo 3: QS aggregato calcolato solo per binning diagnostico post-hoc nel notebook + infrastruttura CL (vedi sezioni 4–5) non esercitata nel training batch.

I ruoli 2 e 3 sono completamente separati dall'architettura. Il QS guida il processo di apprendimento e monitoraggio, non l'inferenza punto a punto.

---

## 3. Reframe narrativa post lagged power

**Lagged power domina l'accuratezza.** m1..m5 contribuiscono marginalmente al MAE puro nel batch training. La narrativa NON è "QS migliora il forecast" ma:

> PhysiQ-PV è un sistema **data-centric + physics-informed + continual-learning-safe** per fleet reale eterogenea.

QS gioca due ruoli distinti:

- **Modello batch:** segnale soft + diagnostico. Impatto marginale sul MAE quando lagged power presente. Correlazione QS↔MAE collassata a +0.056 nel run corrente (era −0.166 senza lagged).
- **Framework CL (deploy):** load-bearing. Gating update via `QualityGatedUpdater`, drift detection ADWIN, replay buffer DER++, diagnostica per-plant.

**Onestà narrativa:** la pipeline CL è un contributo architetturale, non un risultato sperimentale completo. Non esercitata nel run di training corrente (`qs_threshold=None`, ADWIN definito ma non attivo, replay buffer uniforme non QS-weighted).

---

## 4. Componenti del framework CL

### A. Replay buffer QS-weighted

```text
sample_weight ∝ QS_observed
loss_step = mean(sample_weight * MSE(pred, target))
```

Campioni con QS basso (rumorosi, sensori guasti, ombreggiamenti anomali) contribuiscono meno all'aggiornamento. Il dato non viene scartato, il peso è regolato dalla qualità.

Implementazione: `physiq_pv/continual/replay_buffer.py`. DER++, capacity=1000 nel run corrente.

### B. Quality-gated update rule

```text
if mean(QS_window) > qs_threshold:
    apply gradient step
else:
    skip update
```

Evita catastrofic forgetting da batch di bassa qualità. Se un mese di dati è dominato da letture SCADA disconnesse, il modello non viene degradato.

Implementazione: `physiq_pv/continual/quality_gated_update.py`. `qs_threshold` configurabile (None nel run corrente = no gate).

### C. Drift detection via QS time series

```text
ADWIN(QS_plant[t]) → flag se cambio significativo
slope_OLS(QS_plant) over months → degrado sistemico
```

Identifica impianti che stanno degradando prima che si veda nei KPI di previsione. QS = early-warning indicator.

Implementazione: `physiq_pv/agent/drift_monitor.py`.

### D. Clustering soft-DTW su trajectorie QS

```text
QS_matrix (N_plants, T) → soft-DTW TimeSeriesKMeans → labels
```

Cluster tipici:
- impianti sani stazionari
- degrado lineare
- ciclo soiling stagionale
- failure improvviso

Implementazione: `physiq_pv/agent/qs_clustering.py` (tslearn).

### E. Anomaly detection via spatial QS z-score

```text
z[plant, t] = (QS[plant, t] - mean_fleet[t]) / std_fleet[t]
```

- `z < -2` su singolo impianto → guasto isolato
- `mean_fleet[t]` drop con `std_fleet[t]` basso → evento regionale

Distinzione automatica anomalie locali vs globali.

Implementazione: `spatial_qs()` in `physiq_pv/data/quality_score.py`.

### F. Monitoraggio impianti più sani

```text
top_k_healthy = argsort(mean(QS_plant), descending)[:k]
```

Riferimento di flotta per identificare deviazioni negli altri.

---

## 5. Tabella decisione CL guidata da diagnosi

QS multi-componente classifica la causa, framework agisce di conseguenza:

| Pattern QS | Causa | Update modello | Replay | Alert |
|---|---|---|---|---|
| QS singolo plant ↓↓ improvviso, m4↓ | sensor_failure | NO | escludi plant | urgente |
| QS singolo plant ↓ lineare, slope<0 | panel_degradation | NO se transitorio, SÌ se permanente | mantieni storia | manutenzione |
| QS molti plant ↓ sincrono, m4 alto | regional_event | NO | mantieni | nessuno |
| QS oscillante stagionale | soiling | NO durante anomalia | mantieni storia | pulizia |
| QS stabile, errori ↑ | model_drift | SÌ (DER++ + QS-weighted) | replay attivo | nessuno |
| QS area geografica ↓ | local_perturbation | NO (transitorio fisico) | mantieni | nessuno |

Azione CL dipende dalla diagnosi, non dal drift flag generico. Senza classifier causale, CL ingenuo retraina ovunque QS scende → poison da sporcizia / sensori.

Implementazione: `physiq_pv/agent/causal_classifier.py`.

---

## 6. Pipeline retrain online

```text
nuovi dati arrivano (mese M+1)
   ▼
calcola m1..m5 + QS aggregato per ogni sample
   ▼
drift detector ADWIN/KS su QS(t) → flag drift?
   ▼ se drift
classifier causale (cluster soft-DTW + z-score + slope) → diagnosi
   ▼
tabella decisione: update | skip | mask | flag
   ▼ se update
soft weighting:    loss = mean(qs_i^0.2 * MSE_i)
hard gate:         if mean(qs_batch) < qs_threshold → skip
DER++ replay:      + α MSE(curr, old_pred) + β MSE(curr, ground_truth)
                   replay buffer pesato anch'esso da QS storico
   ▼
loss.backward(), optimizer.step()
```

Entry point: `online_loop.py`, `physiq_pv/agent/cycle.py:PhysiQAgent.run()`.

---

## 7. Architettura logica

```text
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
                 │   ST-GNN + lagged + m1..m5          │
                 │   LSTM, RF, XGBoost, persistence    │
                 └─────────────────────────────────────┘
```

QS vive nel layer di controllo. Modello fa predizioni; framework decide se/come/quando aggiornarlo.

---

## 8. Indipendenza framework-modello

I componenti CL non chiedono al modello di sapere cosa è il QS. Funzionano con qualsiasi forecaster:

| Modello base | Replay QS-weighted | Update gating | Drift | Clustering | Anomaly |
|---|:---:|:---:|:---:|:---:|:---:|
| ST-GNN (attuale) | ✓ | ✓ | ✓ | ✓ | ✓ |
| LSTM | ✓ | ✓ | ✓ | ✓ | ✓ |
| Random Forest | ✓ | ✓ | ✓ | ✓ | ✓ |
| XGBoost | ✓ | ✓ | ✓ | ✓ | ✓ |
| Persistence baseline | ✓ | ✓ | ✓ | ✓ | ✓ |

In deploy industriale con modello legacy, framework si applica senza ridisegnare l'architettura.

---

## 9. Sorgenti meteo per CL

Per inferenza/retrain operativo: provider meteo operativo (Open-Meteo, servizio interno) con conversione POA tramite pvlib.

Per retraining offline: reanalysis (PVGIS, ERA5) o dataset storico.

Fallback pvlib clear-sky implementato in `merge_with_weather()` se NetCDF manca: GHI clear-sky come proxy POA, temperatura stagionale, vento costante 3 m/s. Solo demo/inferenza degradata, non training accurato.

---

## 10. Requisiti minimi pipeline online

- Sorgente oraria credibile per irradiance, temperatura, vento
- Mapping lat/lon per geometria solare e grafo spaziale
- Monitoraggio esplicito qualità dati nuovi via `compute_qs`
- Storico ENERGIA per `pv_lag` autoregressive

---

## 11. Schema replay entry

```python
@dataclass
class ReplayEntry:
    x: torch.Tensor
    y_pv: torch.Tensor
    y_ghi: torch.Tensor
    qs: torch.Tensor
    eta: torch.Tensor
    timestamp: datetime
    met_source: str
```

---

## 12. Next steps possibili

1. Run formale `online_loop` con dataset multi-anno simulato (drift indotto) per validare CL infrastructure
2. Ablation isolata `pv_lag` per quantificare contributo netto vs feature meteo+QS
3. Extending dataset multi-anno reale (hook commentato `main.py`, attualmente solo 2019)
4. Decidere posizionamento finale tesi: "interpretable + deployable + safe" vs "competitive accuracy on real data"

---

## 13. Configurazione corrente

| Parametro | Valore | Note |
|---|---|---|
| `qs_threshold` | None | No hard gate attivo |
| `alpha_der` | 0.2 | DER++ MSE(curr, old_pred) |
| `beta_der` | 1.0 | DER++ MSE(curr, ground_truth) |
| `replay_capacity` | 1000 | Buffer uniforme, non QS-weighted |
| ADWIN | definito | Non attivo nel run batch |

Tutti i componenti sono in codice, ma il run di training corrente li lascia inattivi. Da rendere espliciti nella tesi come "infrastruttura presente, validazione sperimentale aperta".
